// Depth, medium, far-tier, and transmission optics. Final light/foam
// composition remains in the terminal material so imports stay one-way.

#define_import_path aqua::optics

#import bevy_pbr::{
    prepass_utils,
    mesh_view_bindings::{globals, lights, view},
}
#import bevy_pbr::mesh_view_bindings as view_bindings
#import aqua::cascade::{DEBUG_MODE_BEAUTY, DEBUG_MODE_BEER_LAMBERT, DEBUG_MODE_REFRACTION_VALIDITY, DEBUG_MODE_SEA_FLOOR, DEBUG_MODE_TRANSMISSION, DEBUG_MODE_UNREFRACTED, DEBUG_MODE_WATER_PATH, LUMINANCE_EPSILON, MIN_NORMAL_Y, capillary_resolved_weight, cascade_layout, godot_fresnel, invocation_extinction, invocation_ripple, invocation_scatter_scale, invocation_scatter_tint, invocation_scattering_asymmetry, sample_planar_reflection, screen_xz_footprint, surface}
#import aqua::medium::{N_WATER, PATH_LENGTH_MAX, attenuate_underwater_scene, fresnel_water_to_air, medium_radiance, water_leaving_radiance}
#import aqua::waves::displace::{FFT_JONSWAP_SLOPE_VARIANCE, GERSTNER_SLOPE_VARIANCE, WAVE_NORMALS_SLOPE_VARIANCE, capillary_normal_slope, detail_normal_sample}
#import aqua::foam::shade::{sample_foam_density}
#import aqua::shore::water::{blended_water_depth, caustic_bed_radiance}
#import aqua::light::incident::{GODOT_NORMAL_FADE_RATE, GODOT_NORMAL_MINIMUM_STRENGTH, GODOT_SSS_MODIFIER, GODOT_WATER_ALBEDO, LUMINANCE_WEIGHTS, filtered_primary_light_color, ggx_distribution, safe_normalize, sample_environment, smith_masking_shadowing, strongest_incident_directional_light}
#import bevy_aqua_core::material::{CameraDepthPath, FoamState, MediumState, NearSurface, PrimaryLightState, SurfaceVertexOutput, TransmissionSample, TransmissionState}

// A 2^-10 residual in the least-attenuated channel bounds body error to
// 0.0977% of scene/scatter contrast before Fresnel. At Crest's shipped minimum
// extinction (0.3 / m), the gate begins at 23.10 m.
const TRANSMISSION_OPAQUE_OPTICAL_DEPTH: f32 = 6.931471806;
// Far-tier shading only takes over once the bed is this deep, in metres.
const FAR_TIER_DEEP_METRES: f32 = 7.0;

// Dupuy et al. 2013 LEADR-style slope filtering. A pixel cannot resolve waves
// shorter than twice its projected world footprint. Integrate each shipped
// wavelength band's measured slope variance below that Nyquist cutoff. The
// logarithmic partial-band term matches the generator's octave partition.
fn unresolved_wave_roughness(
    world_xz: vec2<f32>,
    to_view: vec3<f32>,
    lod_alpha: f32,
    lighting_normal_strength: f32,
    filtered_detail_variance: f32,
) -> f32 {
    let footprint = screen_xz_footprint();
    let unresolved_wavelength = 2.0 * footprint;
    var unresolved_variance = 0.0;
    var resolved_variance = 0.0;
    for (var band = 0u; band < 5u; band++) {
        let maximum_wavelength = cascade_layout.cascades[band].max_wavelength;
        let minimum_wavelength = 0.5 * maximum_wavelength;
        let unresolved_fraction = clamp(
            log2(max(unresolved_wavelength / minimum_wavelength, 1.0)),
            0.0,
            1.0,
        );
        let fft_variance = FFT_JONSWAP_SLOPE_VARIANCE[band];
        let band_variance = mix(
            GERSTNER_SLOPE_VARIANCE[band],
            fft_variance,
            surface.reflection.x,
        );
        unresolved_variance += unresolved_fraction * band_variance;
        resolved_variance += (1.0 - unresolved_fraction) * band_variance;
    }

    let lod_blend_energy = 2.0 * (
        (1.0 - lod_alpha) * (1.0 - lod_alpha) + lod_alpha * lod_alpha
    );
    let detail_variance = WAVE_NORMALS_SLOPE_VARIANCE
        * lod_blend_energy
        * surface.detail.y * surface.detail.y
        * surface.detail.z * surface.detail.z;
    let filtered_variance = min(filtered_detail_variance, detail_variance);
    unresolved_variance += filtered_variance;
    resolved_variance += detail_variance - filtered_variance;
    let capillary_resolved = capillary_resolved_weight(world_xz);
    let capillary_variance = WAVE_NORMALS_SLOPE_VARIANCE
        * surface.capillary.y * surface.capillary.y;
    unresolved_variance += capillary_variance
        * (1.0 - capillary_resolved * capillary_resolved);
    resolved_variance += capillary_variance
        * capillary_resolved * capillary_resolved;

    // GodotOceanWaves fades all lighting slopes with distance. Move the
    // removed resolved variance into LEADR roughness instead of deleting it.
    let removed_fraction = 1.0
        - lighting_normal_strength * lighting_normal_strength;
    var slope_variance = unresolved_variance
        + removed_fraction * resolved_variance;
    let grazing_boost = mix(1.0, 1.5, 1.0 - abs(to_view.y));
    slope_variance *= surface.reflection.y * grazing_boost;
    return min(sqrt(max(slope_variance, 0.0)), surface.reflection.w);
}

fn deep_water_weight(water_depth: f32) -> f32 {
    return smoothstep(0.35, FAR_TIER_DEEP_METRES, water_depth);
}

fn surface_medium_radiance(scene: vec3<f32>, to_view: vec3<f32>, t_end: f32) -> vec3<f32> {
    return water_leaving_radiance(
        scene,
        to_view,
        t_end,
        invocation_extinction(),
        invocation_scatter_scale(),
        invocation_scatter_tint(),
        invocation_scattering_asymmetry(),
    );
}

fn far_field_water(
    world_position: vec4<f32>,
    surface_level: f32,
    geometric_normal: vec3<f32>,
    to_view: vec3<f32>,
    wave_height: f32,
    water_depth: f32,
) -> vec3<f32> {
    let geometric_slope = geometric_normal.xz / max(geometric_normal.y, MIN_NORMAL_Y);
    let lighting_normal = safe_normalize(
        vec3(
            geometric_slope.x * GODOT_NORMAL_MINIMUM_STRENGTH,
            1.0,
            geometric_slope.y * GODOT_NORMAL_MINIMUM_STRENGTH,
        ),
        vec3(0.0, 1.0, 0.0),
    );
    let t_end = min(PATH_LENGTH_MAX, water_depth / max(abs(to_view.y), 0.02));
    var body = surface_medium_radiance(vec3(0.0), to_view, t_end);

    let perceptual_roughness = max(surface.reflection.w, 0.05);
    let reflection = reflect(-to_view, lighting_normal);
    var reflected_radiance = sample_environment(
        reflection,
        lighting_normal,
        perceptual_roughness,
    );
    let planar = sample_planar_reflection(world_position.xyz, surface_level, lighting_normal);
    reflected_radiance = mix(reflected_radiance, planar.color, planar.weight);
    if lights.n_directional_lights > 0u {
        let light = lights.directional_lights[0u];
        let light_direction = safe_normalize(
            light.direction_to_light,
            vec3(0.0, 1.0, 0.0),
        );
        let filtered_light_color = filtered_primary_light_color(
            world_position,
            light.direction_to_light,
            light.sun_disk_angular_size,
            light.color.rgb,
        );
        let light_radiance = filtered_light_color * view.exposure;
        // Preserve broad SSS without near-only texture samples.
        let dot_nv = max(dot(lighting_normal, to_view), 2e-5);
        let sss_light_mask = smith_masking_shadowing(dot_nv, surface.sun.y);
        let sss_near = 0.5 * dot_nv * dot_nv;
        let sss_height = max(0.0, wave_height + 2.5)
            * pow(max(dot(light_direction, -to_view), 0.0), 4.0)
            * pow(
                0.5 - 0.5 * dot(light_direction, lighting_normal),
                3.0,
            );
        body += (sss_height + sss_near)
            * GODOT_SSS_MODIFIER / (1.0 + sss_light_mask)
            * light_radiance * GODOT_WATER_ALBEDO;
        let light_luminance = max(
            dot(filtered_light_color, LUMINANCE_WEIGHTS),
            LUMINANCE_EPSILON,
        );
        let light_color = filtered_light_color / light_luminance;
        let light_strength = clamp(light_luminance / surface.reflection.z, 0.0, 1.0);
        let sun_roughness = min(sqrt(
            surface.sun.y * surface.sun.y
                + perceptual_roughness * perceptual_roughness,
        ), 1.0);
        let halfway = safe_normalize(light_direction + to_view, lighting_normal);
        let dot_nl = max(dot(lighting_normal, light_direction), 2e-5);
        let dot_nv_sun = max(dot(lighting_normal, to_view), 2e-5);
        let light_mask = smith_masking_shadowing(dot_nv_sun, sun_roughness);
        let view_mask_sun = smith_masking_shadowing(dot_nl, sun_roughness);
        let distribution = ggx_distribution(
            clamp(dot(lighting_normal, halfway), 0.0, 1.0),
            sun_roughness,
        );
        let geometric_attenuation = 1.0 / (1.0 + light_mask + view_mask_sun);
        let sun_specular = distribution
            * geometric_attenuation / (4.0 * dot_nv_sun + 0.1);
        reflected_radiance += sun_specular
            * surface.sun.x
            * light_color
            * light_strength;
    }

    let view_alignment = clamp(dot(lighting_normal, to_view), 0.0, 1.0);
    let reflection_weight = clamp(
        godot_fresnel(view_alignment) * surface.fresnel.z,
        0.0,
        1.0,
    );
    return mix(body, reflected_radiance, reflection_weight);
}

fn camera_view_position(uv: vec2<f32>, raw_depth: f32) -> vec3<f32> {
    let ndc = vec3(uv * vec2(2.0, -2.0) + vec2(-1.0, 1.0), raw_depth);
    let position = view.view_from_clip * vec4(ndc, 1.0);
    return position.xyz / max(position.w, LUMINANCE_EPSILON);
}

fn camera_eye_depth(uv: vec2<f32>, raw_depth: f32) -> f32 {
    return max(-camera_view_position(uv, raw_depth).z, 0.0);
}

fn camera_eye_distance(uv: vec2<f32>, raw_depth: f32) -> f32 {
    return length(camera_view_position(uv, raw_depth));
}

fn reconstruct_world_from_uv(uv: vec2<f32>, raw_depth: f32) -> vec3<f32> {
    let ndc = vec3(uv * vec2(2.0, -2.0) + vec2(-1.0, 1.0), raw_depth);
    let world = view.world_from_clip * vec4(ndc, 1.0);
    return world.xyz / max(world.w, LUMINANCE_EPSILON);
}

fn empty_camera_depth_path() -> CameraDepthPath {
    var result: CameraDepthPath;
    result.path_length = 0.0;
    result.screen_uv = vec2(0.0);
    result.scene_z = 0.0;
    result.hit_y = 0.0;
    result.has_background = false;
    return result;
}

fn camera_depth_path(in: SurfaceVertexOutput) -> CameraDepthPath {
    var result = empty_camera_depth_path();
#ifdef DEPTH_PREPASS
    let viewport_origin = view.viewport.xy;
    let viewport_size = view.viewport.zw;
    result.screen_uv = clamp(
        (in.position.xy - viewport_origin) / viewport_size,
        vec2(0.0),
        vec2(1.0),
    );
    let scene_raw_depth = prepass_utils::prepass_depth(in.position, 0u);
    result.has_background = scene_raw_depth > 0.0;
    result.scene_z = camera_eye_depth(result.screen_uv, scene_raw_depth);
    let scene_distance = camera_eye_distance(result.screen_uv, scene_raw_depth);
    let surface_distance = length(in.world_position.xyz - view.world_position.xyz);
    result.path_length = max(scene_distance - surface_distance, 0.0);
    if result.has_background {
        result.hit_y = reconstruct_world_from_uv(result.screen_uv, scene_raw_depth).y;
    }
#endif
    return result;
}

// Crest `OceanEmission.hlsl:195-242`. Opaque camera depth, never SeaFloorDepth,
// is the Euclidean water path. A valid refracted sample supplies the colour UV,
// that path, and the hit height; a sample in front of the water cancels the
// offset. The unrefracted path only scales the warp.
fn resolve_transmission_sample(
    in: SurfaceVertexOutput,
    normal: vec3<f32>,
    path: CameraDepthPath,
    allow_refraction: bool,
) -> TransmissionSample {
    var result: TransmissionSample;
    result.uv = path.screen_uv;
    result.path_length = path.path_length;
    result.hit_y = path.hit_y;
    result.has_background = path.has_background;
    result.refraction_valid = false;
#ifdef DEPTH_PREPASS
    if allow_refraction {
        let shallow_gap = min(1.0, 0.5 * path.path_length);
        // Project the horizontal normal perturbation through this camera. Using
        // world XZ as UV axes makes distortion rotate incorrectly with camera yaw
        // and roll. The mean plane contributes no perturbation.
        let clip_perturbation = view.clip_from_world * vec4(normal.x, 0.0, normal.z, 0.0);
        let screen_perturbation = 0.5 * vec2(clip_perturbation.x, -clip_perturbation.y);
        let refract_offset = surface.debug.y * screen_perturbation
            * shallow_gap / max(path.scene_z, LUMINANCE_EPSILON);
        let refracted_uv = clamp(path.screen_uv + refract_offset, vec2(0.0), vec2(1.0));
        let refracted_pixel = refracted_uv * (view.viewport.zw - vec2(1.0))
            + view.viewport.xy;
        let refracted_position = vec4(refracted_pixel, in.position.zw);
        let refracted_raw_depth = prepass_utils::prepass_depth(refracted_position, 0u);
        result.refraction_valid = refracted_raw_depth > 0.0
            && refracted_raw_depth < in.position.z;
        if result.refraction_valid {
            result.uv = refracted_uv;
            let surface_distance = length(in.world_position.xyz - view.world_position.xyz);
            let scene_distance = camera_eye_distance(refracted_uv, refracted_raw_depth);
            result.path_length = max(scene_distance - surface_distance, 0.0);
            result.has_background = true;
            result.hit_y = reconstruct_world_from_uv(refracted_uv, refracted_raw_depth).y;
        }
    }
#endif
    return result;
}

// Transmission samples the opaque buffer at full resolution. Distortion
// comes only from the displacement normal; no roughness mip or blur is used.
fn opaque_background(subview_uv: vec2<f32>) -> vec3<f32> {
    let dimensions = vec2<f32>(textureDimensions(view_bindings::view_transmission_texture));
    let full_uv = (
        subview_uv * view.viewport.zw + view.viewport.xy
    ) / dimensions;
    return textureSampleLevel(
        view_bindings::view_transmission_texture,
        view_bindings::view_transmission_sampler,
        full_uv,
        0.0,
    ).rgb;
}

fn resolve_near_surface(
    in: SurfaceVertexOutput,
    surface_lod: u32,
    geometric_normal: vec3<f32>,
    far_tier: f32,
    mode: u32,
) -> NearSurface {
    var normal = geometric_normal;
    var filtered_detail_variance = 0.0;
    if mode >= DEBUG_MODE_BEAUTY {
        let near_weight = 1.0 - far_tier;
        var resolved_slope = normal.xz / max(normal.y, MIN_NORMAL_Y);
        let detail = detail_normal_sample(
            in.undisplaced_xz,
            surface_lod,
            in.sample_data.y,
            invocation_ripple(),
        );
        resolved_slope += near_weight * detail.xy;
        filtered_detail_variance = near_weight * near_weight * detail.z;
        resolved_slope += near_weight * capillary_normal_slope(in.undisplaced_xz, invocation_ripple())
            * capillary_resolved_weight(in.undisplaced_xz);
        normal = safe_normalize(
            vec3(resolved_slope.x, 1.0, resolved_slope.y),
            vec3(0.0, 1.0, 0.0),
        );
    }
    let lighting_distance = length(in.world_position.xz - view.world_position.xz);
    let lighting_normal_strength = mix(
        GODOT_NORMAL_MINIMUM_STRENGTH,
        1.0,
        exp(-lighting_distance * GODOT_NORMAL_FADE_RATE),
    );
    let full_slope = normal.xz / max(normal.y, MIN_NORMAL_Y);
    let lighting_normal = safe_normalize(
        vec3(
            full_slope.x * lighting_normal_strength,
            1.0,
            full_slope.y * lighting_normal_strength,
        ),
        vec3(0.0, 1.0, 0.0),
    );
    return NearSurface(
        normal,
        lighting_normal,
        lighting_distance,
        lighting_normal_strength,
        filtered_detail_variance,
    );
}

fn sample_water_medium(
    in: SurfaceVertexOutput,
    surface_lod: u32,
    mode: u32,
) -> MediumState {
    let water_depth = blended_water_depth(in.undisplaced_xz);
    var foam_density = 0.0;
    if mode != DEBUG_MODE_SEA_FLOOR {
        foam_density = sample_foam_density(
            in.undisplaced_xz,
            surface_lod,
            in.sample_data.y,
        );
    }
    return MediumState(
        water_depth,
        foam_density,
    );
}

fn transmitted_scene(
    uv: vec2<f32>,
    surface_y: f32,
    hit_y: f32,
    has_hit: bool,
) -> vec3<f32> {
    let scene = opaque_background(uv);
    if !has_hit {
        return scene;
    }
    return attenuate_underwater_scene(
        scene,
        hit_y,
        surface_y,
        invocation_extinction(),
    );
}

fn illuminate_bed(
    scene_colour: vec3<f32>,
    in: SurfaceVertexOutput,
    medium: MediumState,
    primary: PrimaryLightState,
) -> vec3<f32> {
    let incident = strongest_incident_directional_light(
        in,
        vec3(0.0, 1.0, 0.0),
        primary.view_z,
    );
    if !incident.valid {
        return scene_colour;
    }
    return caustic_bed_radiance(
        scene_colour,
        in.undisplaced_xz,
        medium.water_depth,
        globals.time,
        incident.direction,
        incident.color,
        incident.shadow,
    );
}

fn transmitted_body(
    in: SurfaceVertexOutput,
    medium: MediumState,
    primary: PrimaryLightState,
    to_view: vec3<f32>,
    sample: TransmissionSample,
) -> vec3<f32> {
    let scene_colour = transmitted_scene(
        sample.uv,
        in.world_position.y,
        sample.hit_y,
        sample.has_background,
    );
    let lit_scene = illuminate_bed(scene_colour, in, medium, primary);
    return surface_medium_radiance(lit_scene, to_view, sample.path_length);
}

fn resolve_transmission(
    in: SurfaceVertexOutput,
    normal: vec3<f32>,
    to_view: vec3<f32>,
    medium: MediumState,
    foam: FoamState,
    primary: PrimaryLightState,
    mode: u32,
) -> TransmissionState {
    var body = surface_medium_radiance(vec3(0.0), to_view, PATH_LENGTH_MAX);
    var shared_depth_path = foam.depth_path;
    if (mode >= DEBUG_MODE_WATER_PATH && mode <= DEBUG_MODE_BEER_LAMBERT)
        || mode == DEBUG_MODE_BEAUTY {
        if !foam.has_depth_path {
            shared_depth_path = camera_depth_path(in);
        }
        let sample = resolve_transmission_sample(
            in,
            normal,
            shared_depth_path,
            mode != DEBUG_MODE_UNREFRACTED,
        );
        if mode == DEBUG_MODE_WATER_PATH {
            let path = clamp(sample.path_length / surface.debug.z, 0.0, 1.0);
            return TransmissionState(body, vec4(vec3(path), 1.0), true);
        }
        if mode == DEBUG_MODE_REFRACTION_VALIDITY {
            let output = select(
                vec4(1.0, 0.0, 0.0, 1.0),
                vec4(0.0, 1.0, 0.0, 1.0),
                sample.refraction_valid,
            );
            return TransmissionState(body, output, true);
        }
        if mode == DEBUG_MODE_TRANSMISSION || mode == DEBUG_MODE_UNREFRACTED {
            return TransmissionState(body, vec4(opaque_background(sample.uv), 1.0), true);
        }
        if mode != DEBUG_MODE_BEAUTY {
            body = transmitted_body(in, medium, primary, to_view, sample);
            return TransmissionState(body, vec4(body, 1.0), true);
        }
        let extinction = invocation_extinction();
        let minimum_extinction = min(extinction.r, min(extinction.g, extinction.b));
        let optical_depth = minimum_extinction * sample.path_length;
        if sample.has_background
            && sample.path_length > LUMINANCE_EPSILON
            && optical_depth < TRANSMISSION_OPAQUE_OPTICAL_DEPTH {
            body = transmitted_body(in, medium, primary, to_view, sample);
        } else if sample.has_background
            && sample.path_length > LUMINANCE_EPSILON {
            body = surface_medium_radiance(vec3(0.0), to_view, sample.path_length);
        }
    }
    return TransmissionState(body, vec4(0.0), false);
}

const UNDERSIDE_AIR_MARGIN: f32 = 0.05;
const UNDERSIDE_WARP: f32 = 0.12;
const TIR_SSR_STEPS: u32 = 24u;
const TIR_SSR_STEP: f32 = 1.0;
const TIR_SSR_START: f32 = 0.3;
const TIR_SSR_THICKNESS: f32 = 2.0;
// Looking straight up, water-to-air Fresnel is ~F0. March only when the
// bounce is a visible share of the mix (TIR is 1.0).
const TIR_SSR_FRESNEL_MIN: f32 = 0.2;

fn underside_bounce_radiance(
    origin: vec3<f32>,
    bounced: vec3<f32>,
    surface_y: f32,
    march: bool,
) -> vec3<f32> {
    var scene = vec3(0.0);
    var t_end = PATH_LENGTH_MAX;
#ifdef DEPTH_PREPASS
    if march {
        let viewport_origin = view.viewport.xy;
        let viewport_size = view.viewport.zw;
        var t = TIR_SSR_START;
        for (var i = 0u; i < TIR_SSR_STEPS; i++) {
            t += TIR_SSR_STEP;
            let world = origin + bounced * t;
            if world.y > surface_y - UNDERSIDE_AIR_MARGIN {
                break;
            }
            let clip = view.clip_from_world * vec4(world, 1.0);
            if clip.w <= LUMINANCE_EPSILON {
                break;
            }
            let ndc = clip.xyz / clip.w;
            let uv = ndc.xy * vec2(0.5, -0.5) + vec2(0.5);
            if any(uv < vec2(0.0)) || any(uv > vec2(1.0)) {
                break;
            }
            let pixel = uv * (viewport_size - vec2(1.0)) + viewport_origin;
            let scene_depth = prepass_utils::prepass_depth(vec4(pixel, 0.0, 1.0), 0u);
            if scene_depth <= 0.0 {
                continue;
            }
            let ray_eye = camera_eye_depth(uv, ndc.z);
            let scene_eye = camera_eye_depth(uv, scene_depth);
            if ray_eye > scene_eye && (ray_eye - scene_eye) < TIR_SSR_THICKNESS {
                let hit = reconstruct_world_from_uv(uv, scene_depth);
                if hit.y < surface_y - UNDERSIDE_AIR_MARGIN {
                    scene = attenuate_underwater_scene(
                        opaque_background(uv),
                        hit.y,
                        surface_y,
                        invocation_extinction(),
                    );
                    t_end = clamp(dot(hit - origin, bounced), TIR_SSR_START, PATH_LENGTH_MAX);
                    break;
                }
            }
        }
    }
#endif
    return medium_radiance(
        scene,
        bounced,
        t_end,
        0.0,
        invocation_extinction(),
        invocation_scatter_scale(),
        invocation_scatter_tint(),
        invocation_scattering_asymmetry(),
    );
}

// Water-to-air interface. Scatter along camera-to-mesh is the volume pass.
// The window is air radiance * n² * T. TIR and the reflected lobe are
// in-water radiance along the bounce, with a depth-buffer march for geometry
// still on screen and the open-path integral on a miss.
fn shade_underside(
    in: SurfaceVertexOutput,
    surface_lod: u32,
    geometric_normal: vec3<f32>,
    to_view: vec3<f32>,
    mode: u32,
) -> vec4<f32> {
    let near = resolve_near_surface(in, surface_lod, geometric_normal, 0.0, mode);
    let water_normal = safe_normalize(-near.normal, vec3(0.0, -1.0, 0.0));
    let incident = -to_view;
    let bounced = reflect(incident, water_normal);
    let transmitted = refract(incident, water_normal, N_WATER);
    let cos_theta = max(dot(-incident, water_normal), 0.0);
    let fresnel = fresnel_water_to_air(cos_theta);
    let reflected = underside_bounce_radiance(
        in.world_position.xyz,
        bounced,
        in.world_position.y,
        fresnel >= TIR_SSR_FRESNEL_MIN,
    );
    if dot(transmitted, transmitted) < LUMINANCE_EPSILON {
        return vec4(reflected, 1.0);
    }

    let roughness = unresolved_wave_roughness(
        in.undisplaced_xz,
        to_view,
        in.sample_data.y,
        near.lighting_normal_strength,
        near.filtered_detail_variance,
    );
    var window = sample_environment(transmitted, geometric_normal, roughness);
    if lights.n_directional_lights > 0u {
        let light = lights.directional_lights[0u];
        let light_direction = safe_normalize(
            light.direction_to_light,
            vec3(0.0, 1.0, 0.0),
        );
        let sun_color = filtered_primary_light_color(
            in.world_position,
            light.direction_to_light,
            light.sun_disk_angular_size,
            light.color.rgb,
        ) * view.exposure;
        let disc = pow(max(dot(transmitted, light_direction), 0.0), 256.0);
        window += sun_color * disc;
    }

#ifdef DEPTH_PREPASS
    let viewport_origin = view.viewport.xy;
    let viewport_size = view.viewport.zw;
    let screen_uv = clamp(
        (in.position.xy - viewport_origin) / viewport_size,
        vec2(0.0),
        vec2(1.0),
    );
    let warped_uv = screen_uv + surface.debug.y * near.lighting_normal.xz * UNDERSIDE_WARP;
    if all(warped_uv >= vec2(0.0)) && all(warped_uv <= vec2(1.0)) {
        let warped_pixel = warped_uv * (viewport_size - vec2(1.0)) + viewport_origin;
        let warped_position = vec4(warped_pixel, in.position.zw);
        let warped_depth = prepass_utils::prepass_depth(warped_position, 0u);
        let behind_surface = warped_depth < in.position.z;
        let hit = reconstruct_world_from_uv(warped_uv, warped_depth);
        let in_air = hit.y > in.world_position.y + UNDERSIDE_AIR_MARGIN;
        if warped_depth > 0.0 && behind_surface && in_air {
            window = opaque_background(warped_uv);
        }
    }
#endif

    window *= N_WATER * N_WATER;
    return vec4(mix(window, reflected, fresnel), 1.0);
}
