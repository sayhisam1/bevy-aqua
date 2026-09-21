// Depth, medium, far-tier, and transmission optics. Final light/foam
// composition remains in the terminal material so imports stay one-way.

#define_import_path aqua::optics

#import bevy_pbr::{
    prepass_utils,
    mesh_view_bindings::{globals, lights, view},
}
#import bevy_pbr::mesh_view_bindings as view_bindings
#import aqua::cascade::{DEBUG_MODE_BEAUTY, DEBUG_MODE_BEER_LAMBERT, DEBUG_MODE_REFRACTION_VALIDITY, DEBUG_MODE_SEA_FLOOR, DEBUG_MODE_TRANSMISSION, DEBUG_MODE_UNREFRACTED, DEBUG_MODE_WATER_PATH, LUMINANCE_EPSILON, MIN_NORMAL_Y, capillary_resolved_weight, cascade_layout, godot_fresnel, field_params, sample_field_level, invocation_extinction, invocation_underwater_scatter_scale, invocation_scatter_tint, invocation_scattering_asymmetry, invocation_ripple, sample_planar_reflection, screen_xz_footprint, invocation_sun_roughness, uses_filtered_spectral_surface, get_spectral_filtered_variance, surface}
#import aqua::waves::displace::{CAPILLARY_RESOLVED_ENERGY, WAVE_NORMALS_SLOPE_VARIANCE, capillary_normal_slope, detail_normal_sample}
#import aqua::foam::shade::{sample_foam_density}
#import aqua::shore::water::{blended_water_depth, caustic_bed_radiance}
#import aqua::medium::{PATH_LENGTH_MAX, water_leaving_radiance}
#import aqua::light::incident::{GODOT_SSS_MODIFIER, GODOT_WATER_ALBEDO, LUMINANCE_WEIGHTS, filtered_primary_light_color, ggx_distribution, safe_normalize, smith_masking_shadowing, strongest_incident_directional_light}
#import aqua::light::environment::{sample_diffuse_environment, sample_environment}
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
    near_detail_weight: f32,
    filtered_detail_variance: f32,
    filtered_capillary_variance: f32,
) -> f32 {
    let footprint = screen_xz_footprint();
    let unresolved_wavelength = 2.0 * footprint;
    var unresolved_variance = 0.0;
    let spectral = surface.reflection.x > 0.5;
    var band_count = select(5u, 8u, spectral);
    if spectral && uses_filtered_spectral_surface() {
        unresolved_variance = get_spectral_filtered_variance();
        band_count = 0u;
    }
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
    }

    // Two fixed-world independent detail fields; no geometry-LOD energy pulse.
    let lod_blend_energy = 2.0;
    let ripple = invocation_ripple();
    let detail_variance = WAVE_NORMALS_SLOPE_VARIANCE
        * lod_blend_energy
        * surface.detail.y * surface.detail.y
        * surface.detail.z * surface.detail.z
        * ripple * ripple;
    let near_energy = near_detail_weight * near_detail_weight;
    // filtered_detail_variance already includes near_energy. Transfer the
    // energy removed by the far-tier fade as well as the texture mip filter.
    let filtered_variance = min(filtered_detail_variance, near_energy * detail_variance);
    unresolved_variance += filtered_variance + (1.0 - near_energy) * detail_variance;
    let capillary_resolved = capillary_resolved_weight(world_xz);
    let capillary_variance = WAVE_NORMALS_SLOPE_VARIANCE
        * surface.capillary.y * surface.capillary.y * ripple * ripple;
    let capillary_resolved_energy = near_energy
        * capillary_resolved * capillary_resolved;
    unresolved_variance += capillary_variance
        * (1.0 - capillary_resolved_energy * CAPILLARY_RESOLVED_ENERGY)
        + min(filtered_capillary_variance, capillary_resolved_energy
            * CAPILLARY_RESOLVED_ENERGY * capillary_variance);

    // Geometric wave slopes retain unit strength; only detail terms fade.
    var slope_variance = unresolved_variance;
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

fn surface_medium_radiance(scene: vec3<f32>, to_view: vec3<f32>, t_end: f32) -> vec3<f32> {
    return water_leaving_radiance(
        scene,
        to_view,
        t_end,
        invocation_extinction(),
        invocation_underwater_scatter_scale(),
        invocation_scatter_tint(),
        invocation_scattering_asymmetry(),
    );
}

fn far_field_water(
    in: SurfaceVertexOutput,
    surface_level: f32,
    near: NearSurface,
    to_view: vec3<f32>,
    water_depth: f32,
) -> vec3<f32> {
    // Reuse the accepted candidate-tier surface, including safe slope reconstruction.
    let lighting_normal = near.lighting_normal;
    let t_end = min(PATH_LENGTH_MAX, water_depth / max(abs(to_view.y), 0.02));
    var body = surface_medium_radiance(vec3(0.0), to_view, t_end);
    let diffuse_irradiance = sample_diffuse_environment(lighting_normal);
    body += diffuse_irradiance * GODOT_WATER_ALBEDO;

    let perceptual_roughness = unresolved_wave_roughness(
        in.undisplaced_xz,
        to_view,
        in.sample_data.y,
        near.near_detail_weight,
        near.filtered_detail_variance,
        near.filtered_capillary_variance,
    );
    let reflection = reflect(-to_view, lighting_normal);
    var reflected_radiance = sample_environment(
        reflection,
        lighting_normal,
        perceptual_roughness,
    );
    let planar = sample_planar_reflection(in.world_position.xyz, surface_level, lighting_normal, perceptual_roughness);
    reflected_radiance = mix(reflected_radiance, planar.color, planar.weight);
    if lights.n_directional_lights > 0u {
        let light = lights.directional_lights[0u];
        let light_direction = safe_normalize(
            light.direction_to_light,
            vec3(0.0, 1.0, 0.0),
        );
        let filtered_light_color = filtered_primary_light_color(
            in.world_position,
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
        let sss_height = max(0.0, in.sample_data.z + 2.5)
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
    result.receiver_world = vec3(0.0);
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
        result.receiver_world = (view.world_from_view * vec4(background, 1.0)).xyz;
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
    result.raw_receiver_world = path.receiver_world;
    result.refracted_receiver_world = path.receiver_world;
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
    // Select the texel containing this continuous UV; only bound the endpoint.
    let refracted_pixel = min(
        result.refracted_uv * view.viewport.zw,
        view.viewport.zw - vec2(1.0),
    ) + view.viewport.xy;
    let refracted_position = vec4(refracted_pixel, in.position.zw);
    let refracted_raw_depth = prepass_utils::prepass_depth(refracted_position, 0u);
    result.refracted_sample_valid = refracted_raw_depth > 0.0
        && refracted_raw_depth < in.position.z;
    if result.refracted_sample_valid {
        let background = camera_view_position(result.refracted_uv, refracted_raw_depth);
        let water = (view.view_from_world * in.world_position).xyz;
        result.refracted_path_length = length(background - water);
        result.refracted_receiver_world = (view.world_from_view * vec4(background, 1.0)).xyz;
    }
#endif
    return result;
}

// Transmission samples the opaque buffer at full resolution. Distortion
// comes only from the displacement normal; no roughness mip or blur is used.
fn opaque_background(subview_uv: vec2<f32>) -> vec3<f32> {
    let dimensions = vec2<f32>(textureDimensions(view_bindings::view_transmission_texture));
    // Keep the linear footprint inside this viewport, not just the backing texture.
    let color_pixel = clamp(
        subview_uv * view.viewport.zw + view.viewport.xy,
        view.viewport.xy + vec2(0.5),
        view.viewport.xy + view.viewport.zw - vec2(0.5),
    );
    let full_uv = color_pixel / dimensions;
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
    var filtered_capillary_variance = 0.0;
    if mode >= DEBUG_MODE_BEAUTY {
        let near_weight = 1.0 - far_tier;
        var resolved_slope = normal.xz / max(normal.y, MIN_NORMAL_Y);
        // At exactly zero near weight, omit only the finite detail terms.
        // Keep slope reconstruction below: a geometric normal can have Y <= 0.
        if near_weight != 0.0 {
            let detail = detail_normal_sample(
                in.undisplaced_xz,
                surface_lod,
                in.sample_data.y,
                invocation_ripple(),
            );
            resolved_slope += near_weight * detail.xy;
            filtered_detail_variance = near_weight * near_weight * detail.z;
            let capillary = capillary_normal_slope(
                in.undisplaced_xz,
                invocation_ripple(),
            );
            let capillary_weight = capillary_resolved_weight(in.undisplaced_xz);
            resolved_slope += near_weight * capillary.xy * capillary_weight;
            filtered_capillary_variance = near_weight * near_weight
                * capillary_weight * capillary_weight * capillary.z;
        }
        normal = safe_normalize(
            vec3(resolved_slope.x, 1.0, resolved_slope.y),
            vec3(0.0, 1.0, 0.0),
        );
    }
    let lighting_distance = length(in.world_position.xz - view.world_position.xz);
    let full_slope = normal.xz / max(normal.y, MIN_NORMAL_Y);
    let lighting_normal = safe_normalize(
        vec3(full_slope.x, 1.0, full_slope.y),
        vec3(0.0, 1.0, 0.0),
    );
    return NearSurface(
        normal,
        lighting_normal,
        lighting_distance,
        1.0 - far_tier,
        filtered_detail_variance,
        filtered_capillary_variance,
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

// PROTOTYPE A: frozen selected refraction offset, NOT neighbor acceptance replay.
// Negative footprint means omit caustics, never force a tiny/zero LOD.
fn receiver_caustic_footprint(
    in: SurfaceVertexOutput,
    background_uv: vec2<f32>,
    receiver_world: vec3<f32>,
    use_refraction: bool,
) -> f32 {
#ifdef DEPTH_PREPASS
    // Both depth lanes use viewport-size addressing: one UV step is one pixel.
    let span = view.viewport.zw;
    if any(span <= vec2(1.0)) { return -1.0; }
    let step_uv = vec2(1.0) / span;
    // Do not differentiate a clamped UV or load across a viewport boundary.
    if any(background_uv <= step_uv) || any(background_uv >= vec2(1.0) - step_uv) {
        return -1.0;
    }
    let center_view = (view.view_from_world * vec4(receiver_world, 1.0)).xyz;
    if !(all(abs(center_view) < vec3(1e20))) { return -1.0; }
    // Deliberately conservative experimental edge threshold in view metres.
    // Smooth steep slopes may be rejected too. Small discontinuities can pass;
    // this is not a surface-ID test and needs silhouette validation.
    let maximum_z_jump = max(0.05, abs(center_view.z) * 0.01);
    var footprint = 0.0;
    for (var i = 0u; i < 4u; i++) {
        var offset = vec2(1.0, 0.0);
        if i == 1u { offset = vec2(-1.0, 0.0); }
        if i == 2u { offset = vec2(0.0, 1.0); }
        if i == 3u { offset = vec2(0.0, -1.0); }
        let uv = background_uv + offset * step_uv;
        let pixel = select(
            uv * span,
            min(uv * span, span - vec2(1.0)),
            use_refraction,
        ) + view.viewport.xy;
        let raw_depth = prepass_utils::prepass_depth(vec4(pixel, in.position.zw), 0u);
        if !(raw_depth > 0.0 && raw_depth < in.position.z) { return -1.0; }
        let neighbor_view = camera_view_position(uv, raw_depth);
        if !(all(abs(neighbor_view) < vec3(1e20))) { return -1.0; }
        if abs(neighbor_view.z - center_view.z) > maximum_z_jump { return -1.0; }
        let neighbor_world = (view.world_from_view * vec4(neighbor_view, 1.0)).xyz;
        if !(all(abs(neighbor_world) < vec3(1e20))) { return -1.0; }
        footprint = max(footprint, length(neighbor_world.xz - receiver_world.xz));
    }
    // Degenerate footprint is invalid, rather than a spuriously sharp mip.
    if !(footprint > 0.0 && footprint < 1e20) { return -1.0; }
    return footprint;
#else
    return -1.0;
#endif
}

fn illuminate_bed(
    scene_colour: vec3<f32>,
    in: SurfaceVertexOutput,
    primary: PrimaryLightState,
    depth: CameraDepthDebug,
    use_refraction: bool,
    background_uv: vec2<f32>,
    source_slot: u32,
) -> vec3<f32> {
    // Shared by beauty and the existing Beer-Lambert/sea-floor caustic lanes.
    // No-prepass defaults fail here; transmission-only diagnostics bypass us.
    if surface.caustics.x * surface.sea_floor.w <= 0.0 || !depth.has_background {
        return scene_colour;
    }
    let selected_path = select(depth.path_length, depth.refracted_path_length, use_refraction);
    if !(selected_path > LUMINANCE_EPSILON) { return scene_colour; }
    let receiver = select(depth.raw_receiver_world, depth.refracted_receiver_world, use_refraction);
    if !(all(abs(receiver) < vec3(1e20))) { return scene_colour; }
    var receiver_slot = 0u;
    var level = cascade_layout.bed_range.z;
    if field_params.info.x > 0.5 {
        // sample_field_level uses a nearest categorical slot, filtered level.
        let field = sample_field_level(receiver.xz);
        receiver_slot = u32(field.y + 0.5);
        if receiver_slot != source_slot { return scene_colour; }
        if receiver_slot != 0u { level = field.x; }
    } else if source_slot != 0u {
        return scene_colour;
    }
    let water_depth = level - receiver.y;
    if !(water_depth > LUMINANCE_EPSILON && water_depth < surface.caustics.w) {
        return scene_colour;
    }
    // Keep shadows/sun selection at the water fragment, as before.
    let incident = strongest_incident_directional_light(
        in,
        vec3(0.0, 1.0, 0.0),
        primary.view_z,
    );
    if !incident.valid || incident.direction.y <= 0.0 || incident.shadow <= 0.0 {
        return scene_colour;
    }
    // Only enabled, submerged, same-body, sun-admitted caustics pay neighbors.
    let footprint = receiver_caustic_footprint(in, background_uv, receiver, use_refraction);
    if footprint < 0.0 { return scene_colour; }
    return caustic_bed_radiance(
        scene_colour,
        receiver.xz,
        water_depth,
        footprint,
        globals.time,
        incident.direction,
        incident.color,
        incident.shadow,
        invocation_extinction(),
    );
}

// Beauty-only attenuation. Diagnostics intentionally retain authored extinction.
fn beauty_extinction(water_depth: f32) -> vec3<f32> {
    let scale = mix(vec3(0.52, 0.42, 0.62), vec3(1.0), deep_water_weight(water_depth));
    return invocation_extinction() * scale;
}

// Beauty skips color sampling when there is no usable background or the accepted
// water path is opaque. Diagnostics deliberately bypass these shortcuts.
fn beauty_transmission(
    in: SurfaceVertexOutput,
    normal: vec3<f32>,
    to_view: vec3<f32>,
    medium: MediumState,
    primary: PrimaryLightState,
    depth_path: CameraDepthPath,
    source_slot: u32,
) -> vec3<f32> {
    if !(depth_path.has_background && depth_path.path_length > LUMINANCE_EPSILON) {
        return surface_medium_radiance(vec3(0.0), to_view, PATH_LENGTH_MAX);
    }

    let depth_debug = camera_depth_debug_from_path(in, normal, depth_path);
    let use_refraction = depth_debug.refracted_sample_valid;
    let water_path = select(
        depth_debug.path_length,
        depth_debug.refracted_path_length,
        use_refraction,
    );
    // Reduce extinction in the first few metres so the seabed stays visible.
    let extinction = beauty_extinction(medium.water_depth);
    let minimum_extinction = min(extinction.r, min(extinction.g, extinction.b));
    // Refraction can reveal a shallower bed: test the accepted path, not the original.
    if !(minimum_extinction * water_path < TRANSMISSION_OPAQUE_OPTICAL_DEPTH) {
        return surface_medium_radiance(vec3(0.0), to_view, water_path);
    }

    let background_uv = select(
        depth_debug.screen_uv,
        depth_debug.refracted_uv,
        use_refraction,
    );
    let scene_colour = opaque_background(background_uv);
    let lit_scene = illuminate_bed(
        scene_colour, in, primary, depth_debug, use_refraction,
        background_uv, source_slot,
    );
    return surface_medium_radiance(lit_scene, to_view, water_path);
}

// Admission uses the same path selection and attenuation as near beauty.
// False means keep near transmission; true also covers no transmissive
// background under the existing contract (not proof of physical opacity).
fn far_path_opaque(
    in: SurfaceVertexOutput,
    normal: vec3<f32>,
    path: CameraDepthPath,
) -> bool {
    if !path.has_background || path.path_length <= LUMINANCE_EPSILON {
        return true;
    }
    let accepted = camera_depth_debug_from_path(in, normal, path);
    let water_path = select(
        accepted.path_length,
        accepted.refracted_path_length,
        accepted.refracted_sample_valid,
    );
    let depth = blended_water_depth(in.undisplaced_xz);
    let extinction = beauty_extinction(depth);
    let minimum_extinction = min(extinction.r, min(extinction.g, extinction.b));
    return minimum_extinction * water_path >= TRANSMISSION_OPAQUE_OPTICAL_DEPTH;
}

fn resolve_transmission(
    in: SurfaceVertexOutput,
    normal: vec3<f32>,
    to_view: vec3<f32>,
    medium: MediumState,
    foam: FoamState,
    primary: PrimaryLightState,
    mode: u32,
    source_slot: u32,
) -> TransmissionState {
    let is_diagnostic = mode >= DEBUG_MODE_WATER_PATH && mode <= DEBUG_MODE_SEA_FLOOR;
    let open_body = surface_medium_radiance(vec3(0.0), to_view, PATH_LENGTH_MAX);
    if mode != DEBUG_MODE_BEAUTY && !is_diagnostic {
        return TransmissionState(open_body, vec4(0.0), false);
    }

    // Foam may already have fetched this pixel's depth. Reuse it for either path.
    var shared_depth_path = foam.depth_path;
    if !foam.has_depth_path {
        shared_depth_path = camera_depth_path(in);
    }
    if mode == DEBUG_MODE_BEAUTY {
        let body = beauty_transmission(
            in, normal, to_view, medium, primary, shared_depth_path, source_slot,
        );
        return TransmissionState(body, vec4(0.0), false);
    }

    // Diagnostic modes keep the full sampling path, even for opaque water.
    var body = open_body;
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

    let lit_scene = illuminate_bed(
        scene_colour, in, primary, depth_debug, use_refraction,
        background_uv, source_slot,
    );
    let water_path = select(
        depth_debug.path_length,
        depth_debug.refracted_path_length,
        use_refraction,
    );
    body = surface_medium_radiance(lit_scene, to_view, water_path);
    if mode == DEBUG_MODE_BEER_LAMBERT {
        return TransmissionState(body, vec4(body, 1.0), true);
    }
    return TransmissionState(body, vec4(0.0), false);
}


#ifdef UNDERWATER
#import aqua::medium::{N_WATER, fresnel_water_to_air, medium_radiance_oriented}

// Water-to-air interface. Reflected rays use the bounded homogeneous medium;
// screen-space ray marching remains omitted until it has validated quality controls.
fn shade_underside(
    in: SurfaceVertexOutput,
    surface_lod: u32,
    geometric_normal: vec3<f32>,
    to_view: vec3<f32>,
    mode: u32,
) -> vec4<f32> {
    let near = resolve_near_surface(in, surface_lod, geometric_normal, 0.0, mode);
    let incident = safe_normalize(-to_view, vec3(0.0, -1.0, 0.0));
    let initial_water_normal = safe_normalize(-near.normal, vec3(0.0, -1.0, 0.0));
    let candidate_up = -initial_water_normal;
    // Raster back-facing does not guarantee a resolved microdetail normal faces
    // the incident ray. Continuously project it into the visible hemisphere;
    // unlike faceforward this does not introduce a whole-normal sign flip.
    let visibility = dot(incident, candidate_up);
    let corrected_up = candidate_up
        + incident * max(1e-3 - visibility, 0.0);
    let facet_up = safe_normalize(corrected_up, incident);
    let water_normal = -facet_up;
    let reflected_direction = reflect(incident, water_normal);
    let transmitted_direction = refract(incident, water_normal, N_WATER);
    let cos_water = clamp(dot(-incident, water_normal), 0.0, 1.0);
    let fresnel = fresnel_water_to_air(cos_water);
    let roughness = unresolved_wave_roughness(
        in.undisplaced_xz,
        to_view,
        in.sample_data.y,
        near.near_detail_weight,
        near.filtered_detail_variance,
        near.filtered_capillary_variance,
    );
    // A reflected water-side ray cannot sample the air environment probe.
    // Evaluate the homogeneous open-path medium in the local facet frame.
    // The paired reflection normal keeps this ray in the local water halfspace
    // even when its world Y component points upward.
    let reflected = medium_radiance_oriented(
        vec3(0.0),
        reflected_direction,
        PATH_LENGTH_MAX,
        0.0,
        invocation_extinction(),
        invocation_underwater_scatter_scale(),
        invocation_scatter_tint(),
        invocation_scattering_asymmetry(),
        facet_up,
    );
    // WGSL refract returns zero at total internal reflection. Exact Fresnel
    // independently reaches one there, so this branch avoids invalid lookups.
    if dot(transmitted_direction, transmitted_direction) < LUMINANCE_EPSILON {
        return vec4(reflected, 1.0);
    }

    var window = sample_environment(
        transmitted_direction,
        -water_normal,
        roughness,
    );
#ifdef DEPTH_PREPASS
    let viewport_origin = view.viewport.xy;
    let viewport_size = view.viewport.zw;
    // Reuse the accepted front-face refraction contract: camera-projected
    // perturbation, shallow-gap scaling, viewport-local clamp, and endpoint
    // texel selection all stay identical across the interface.
    let distortion_path = camera_depth_path(in);
    let distortion = camera_depth_debug_from_path(
        in,
        near.lighting_normal,
        distortion_path,
    );
    let warped_uv = distortion.refracted_uv;
    let warped_pixel = min(
        warped_uv * viewport_size,
        viewport_size - vec2(1.0),
    ) + viewport_origin;
    let warped_position = vec4(warped_pixel, in.position.zw);
    let warped_depth = prepass_utils::prepass_depth(warped_position, 0u);
    if warped_depth > 0.0 && warped_depth < in.position.z {
        let hit = (view.world_from_view
            * vec4(camera_view_position(warped_uv, warped_depth), 1.0)).xyz;
        if hit.y > in.world_position.y + 0.02 {
            window = opaque_background(warped_uv);
        }
    }
#endif
    // Radiance invariant across the interface: L_air = L_water / n²,
    // therefore the sampled air radiance is n² larger in the water domain.
    window *= N_WATER * N_WATER;
    return vec4(mix(window, reflected, fresnel), 1.0);
}
#endif
