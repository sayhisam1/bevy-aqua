// Depth, medium, far-tier, and transmission optics. Final light/foam
// composition remains in the terminal material so imports stay one-way.

#define_import_path aqua::optics

#import bevy_pbr::{
    prepass_utils,
    mesh_view_bindings::{globals, lights, view},
}
#import bevy_pbr::mesh_view_bindings as view_bindings
#import aqua::cascade::{DEBUG_MODE_BEAUTY, DEBUG_MODE_BEER_LAMBERT, DEBUG_MODE_REFRACTION_VALIDITY, DEBUG_MODE_SEA_FLOOR, DEBUG_MODE_TRANSMISSION, DEBUG_MODE_UNREFRACTED, DEBUG_MODE_WATER_PATH, LUMINANCE_EPSILON, MIN_NORMAL_Y, capillary_resolved_weight, cascade_layout, godot_fresnel, invocation_extinction, invocation_ripple, sample_planar_reflection, screen_xz_footprint, invocation_sun_roughness, surface}
#import aqua::waves::displace::{WAVE_NORMALS_SLOPE_VARIANCE, capillary_normal_slope, detail_normal_sample}
#import aqua::foam::shade::{sample_foam_density}
#import aqua::shore::water::{blended_water_depth, caustic_bed_radiance}
#import aqua::light::incident::{GODOT_NORMAL_FADE_RATE, GODOT_NORMAL_MINIMUM_STRENGTH, GODOT_SSS_MODIFIER, GODOT_WATER_ALBEDO, LUMINANCE_WEIGHTS, filtered_primary_light_color, ggx_distribution, safe_normalize, sample_diffuse_environment, sample_environment, smith_masking_shadowing, strongest_incident_directional_light}
#import bevy_aqua_core::material::{CameraDepthDebug, CameraDepthPath, FoamState, MediumState, NearSurface, PrimaryLightState, SurfaceVertexOutput, TransmissionState}

// A 2^-10 residual in the least-attenuated channel bounds body error to
// 0.0977% of scene/scatter contrast before Fresnel. At Crest's shipped minimum
// extinction (0.3 / m), the gate begins at 23.10 m.
const TRANSMISSION_OPAQUE_OPTICAL_DEPTH: f32 = 6.931471806;

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
    let spectral = surface.reflection.x > 0.5;
    let band_count = select(5u, 8u, spectral);
    for (var band = 0u; band < band_count; band++) {
        let cascade = cascade_layout.cascades[min(band, 4u)];
        var maximum_wavelength = cascade.max_wavelength;
        var minimum_wavelength = 0.5 * maximum_wavelength;
        if spectral && band >= 4u {
            let upper = cascade.texel_width * cascade.texture_res / 4.0;
            let octaves = log2(upper / minimum_wavelength);
            maximum_wavelength = minimum_wavelength
                * exp2(octaves * f32(band - 3u) / 4.0);
            minimum_wavelength *= exp2(octaves * f32(band - 4u) / 4.0);
        }
        let unresolved_fraction = clamp(
            log2(max(unresolved_wavelength / minimum_wavelength, 1.0))
                / log2(maximum_wavelength / minimum_wavelength),
            0.0,
            1.0,
        );
        let band_variance = surface.wave_slope_variance[band / 4u][band % 4u];
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
    return smoothstep(0.35, surface.shallow_color.a, water_depth);
}

fn depth_aware_body_albedo(
    water_depth: f32,
    deep_body_albedo: vec3<f32>,
) -> vec3<f32> {
    return mix(
        surface.shallow_color.rgb,
        deep_body_albedo,
        deep_water_weight(water_depth),
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
    let view_vertical = abs(to_view.y);
    let deep_body_albedo = mix(
        surface.grazing_color.rgb,
        surface.deep_color.rgb,
        view_vertical,
    );
    // Camera distance must not turn a shallow lake into deep ocean. Keep the
    // near path's bed-depth color classification while omitting transmission.
    let body_albedo = depth_aware_body_albedo(water_depth, deep_body_albedo);
    let diffuse_irradiance = sample_diffuse_environment(vec3(0.0, 1.0, 0.0));
    var body = diffuse_irradiance * (body_albedo + GODOT_WATER_ALBEDO);

    let perceptual_roughness = max(surface.reflection.w, 0.05);
    let reflection = reflect(-to_view, lighting_normal);
    var reflected_radiance = sample_environment(
        reflection,
        lighting_normal,
        perceptual_roughness,
    );
    let planar = sample_planar_reflection(world_position.xyz, surface_level, lighting_normal, perceptual_roughness);
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
        let lambertian = 0.5 * max(dot(lighting_normal, light_direction), 2e-5);
        body += lambertian * light_radiance * GODOT_WATER_ALBEDO;
        // Preserve broad SSS without near-only texture samples.
        let dot_nv = max(dot(lighting_normal, to_view), 2e-5);
        let sss_light_mask = smith_masking_shadowing(dot_nv, invocation_sun_roughness());
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
        let sun_roughness = min(sqrt(
            invocation_sun_roughness() * invocation_sun_roughness()
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
        // Far and near direct specular share the exposed radiance domain.
        reflected_radiance += sun_specular * surface.sun.x * light_radiance;
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

fn empty_camera_depth_path() -> CameraDepthPath {
    var result: CameraDepthPath;
    result.path_length = 0.0;
    result.screen_uv = vec2(0.0);
    result.scene_z = 0.0;
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
    if result.has_background {
        let background = camera_view_position(result.screen_uv, scene_raw_depth);
        let water = (view.view_from_world * in.world_position).xyz;
        result.scene_z = max(-background.z, 0.0);
        // Axial Z separation is shorter than the ray away from screen centre.
        // Euclidean view-space distance also works for orthographic cameras.
        if background.z < water.z {
            result.path_length = length(background - water);
        }
    }
#endif
    return result;
}

// Reimplementation of the approach in Crest `OceanEmission.hlsl:195-242`. The opaque camera depth,
// never SeaFloorDepth LodData, determines the view-ray water path and whether
// a refracted sample landed on geometry in front of the water.
fn camera_depth_debug_from_path(
    in: SurfaceVertexOutput,
    normal: vec3<f32>,
    path: CameraDepthPath,
) -> CameraDepthDebug {
    var result: CameraDepthDebug;
    result.path_length = path.path_length;
    result.refracted_path_length = path.path_length;
    result.screen_uv = path.screen_uv;
    result.refracted_uv = path.screen_uv;
    result.refracted_sample_valid = false;
    result.has_background = path.has_background;
#ifdef DEPTH_PREPASS
    let shallow_gap = min(1.0, 0.5 * path.path_length);
    // Project the horizontal normal perturbation through this camera. Using
    // world XZ as UV axes makes distortion rotate incorrectly with camera yaw
    // and roll. The mean plane contributes no perturbation.
    let clip_perturbation = view.clip_from_world * vec4(normal.x, 0.0, normal.z, 0.0);
    let screen_perturbation = 0.5 * vec2(clip_perturbation.x, -clip_perturbation.y);
    let refract_offset = surface.debug.y * screen_perturbation
        * shallow_gap / max(path.scene_z, LUMINANCE_EPSILON);
    result.refracted_uv = clamp(path.screen_uv + refract_offset, vec2(0.0), vec2(1.0));
    let refracted_pixel = result.refracted_uv * (view.viewport.zw - vec2(1.0))
        + view.viewport.xy;
    let refracted_position = vec4(refracted_pixel, in.position.zw);
    let refracted_raw_depth = prepass_utils::prepass_depth(refracted_position, 0u);
    result.refracted_sample_valid = refracted_raw_depth > 0.0
        && refracted_raw_depth < in.position.z;
    if result.refracted_sample_valid {
        let background = camera_view_position(result.refracted_uv, refracted_raw_depth);
        let water = (view.view_from_world * in.world_position).xyz;
        result.refracted_path_length = length(background - water);
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
    lighting_normal: vec3<f32>,
    to_view: vec3<f32>,
    mode: u32,
) -> MediumState {
    let water_depth = blended_water_depth(in.undisplaced_xz);
    let view_vertical = abs(to_view.y);
    let deep_body_albedo = mix(
        surface.grazing_color.rgb,
        surface.deep_color.rgb,
        view_vertical,
    );
    // Metric SeaFloorDepth shifts only the volume-scatter endpoint. Reflection,
    // foam, and camera-depth transmission remain on their existing lanes.
    let body_albedo = depth_aware_body_albedo(water_depth, deep_body_albedo);
    var diffuse_irradiance = vec3(0.0);
    if mode >= DEBUG_MODE_BEAUTY {
        diffuse_irradiance = sample_diffuse_environment(lighting_normal);
    }
    var foam_density = 0.0;
    if mode != DEBUG_MODE_SEA_FLOOR {
        foam_density = sample_foam_density(
            in.undisplaced_xz,
            surface_lod,
            in.sample_data.y,
        );
    }
    return MediumState(
        body_albedo,
        diffuse_irradiance,
        water_depth,
        foam_density,
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
        invocation_extinction(),
    );
}

// Beauty skips color sampling when there is no usable background or the accepted
// water path is opaque. Diagnostics deliberately bypass these shortcuts.
fn beauty_transmission(
    in: SurfaceVertexOutput,
    normal: vec3<f32>,
    scatter_colour: vec3<f32>,
    medium: MediumState,
    primary: PrimaryLightState,
    depth_path: CameraDepthPath,
) -> vec3<f32> {
    if !(depth_path.has_background && depth_path.path_length > LUMINANCE_EPSILON) {
        return scatter_colour;
    }

    let depth_debug = camera_depth_debug_from_path(in, normal, depth_path);
    let use_refraction = depth_debug.refracted_sample_valid;
    let water_path = select(
        depth_debug.path_length,
        depth_debug.refracted_path_length,
        use_refraction,
    );
    // Reduce extinction in the first few metres so the seabed stays visible.
    let deep_weight = smoothstep(0.35, surface.shallow_color.a, medium.water_depth);
    let shallow_extinction_scale = mix(vec3(0.52, 0.42, 0.62), vec3(1.0), deep_weight);
    let extinction = invocation_extinction() * shallow_extinction_scale;
    let minimum_extinction = min(extinction.r, min(extinction.g, extinction.b));
    // Refraction can reveal a shallower bed: test the accepted path, not the original.
    if !(minimum_extinction * water_path < TRANSMISSION_OPAQUE_OPTICAL_DEPTH) {
        return scatter_colour;
    }

    let background_uv = select(
        depth_debug.screen_uv,
        depth_debug.refracted_uv,
        use_refraction,
    );
    let scene_colour = opaque_background(background_uv);
    let lit_scene = illuminate_bed(scene_colour, in, medium, primary);
    let alpha = 1.0 - exp(-extinction * water_path);
    return mix(lit_scene, scatter_colour, alpha);
}

fn resolve_transmission(
    in: SurfaceVertexOutput,
    normal: vec3<f32>,
    scatter_colour: vec3<f32>,
    medium: MediumState,
    foam: FoamState,
    primary: PrimaryLightState,
    mode: u32,
) -> TransmissionState {
    let is_diagnostic = mode >= DEBUG_MODE_WATER_PATH && mode <= DEBUG_MODE_SEA_FLOOR;
    if mode != DEBUG_MODE_BEAUTY && !is_diagnostic {
        return TransmissionState(scatter_colour, vec4(0.0), false);
    }

    // Foam may already have fetched this pixel's depth. Reuse it for either path.
    var shared_depth_path = foam.depth_path;
    if !foam.has_depth_path {
        shared_depth_path = camera_depth_path(in);
    }
    if mode == DEBUG_MODE_BEAUTY {
        let body = beauty_transmission(in, normal, scatter_colour, medium, primary, shared_depth_path);
        return TransmissionState(body, vec4(0.0), false);
    }

    // Diagnostic modes keep the full sampling path, even for opaque water.
    var body = scatter_colour;
    let depth_debug = camera_depth_debug_from_path(in, normal, shared_depth_path);
    if mode == DEBUG_MODE_WATER_PATH {
        let path = clamp(depth_debug.path_length / surface.debug.z, 0.0, 1.0);
        return TransmissionState(body, vec4(vec3(path), 1.0), true);
    }
    if mode == DEBUG_MODE_REFRACTION_VALIDITY {
        let output = select(
            vec4(1.0, 0.0, 0.0, 1.0),
            vec4(0.0, 1.0, 0.0, 1.0),
            depth_debug.refracted_sample_valid,
        );
        return TransmissionState(body, output, true);
    }
    let refraction_enabled = mode == DEBUG_MODE_TRANSMISSION
        || mode == DEBUG_MODE_BEER_LAMBERT
        || mode == DEBUG_MODE_SEA_FLOOR;
    let use_refraction = refraction_enabled
        && depth_debug.refracted_sample_valid;
    let background_uv = select(
        depth_debug.screen_uv,
        depth_debug.refracted_uv,
        use_refraction,
    );
    let scene_colour = opaque_background(background_uv);
    if mode == DEBUG_MODE_TRANSMISSION || mode == DEBUG_MODE_UNREFRACTED {
        return TransmissionState(body, vec4(scene_colour, 1.0), true);
    }

    // Crest `OceanEmission.hlsl:254-266`: per-channel Beer-Lambert fog.
    // The colour ramp emerges from extinction; there is no authored ramp.
    let lit_scene = illuminate_bed(scene_colour, in, medium, primary);
    let water_path = select(
        depth_debug.path_length,
        depth_debug.refracted_path_length,
        use_refraction,
    );
    let alpha = 1.0 - exp(-invocation_extinction() * water_path);
    body = mix(lit_scene, scatter_colour, alpha);
    if mode == DEBUG_MODE_BEER_LAMBERT {
        return TransmissionState(body, vec4(body, 1.0), true);
    }
    return TransmissionState(body, vec4(0.0), false);
}
