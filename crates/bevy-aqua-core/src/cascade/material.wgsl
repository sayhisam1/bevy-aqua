// Composed cascade surface material: vertex snap/morph/displacement, the
// transmission/reflection/foam shading pipeline, and both stage entry
// points. Imports every feature module over the aqua::cascade contract.
// Owned by bevy-aqua-core (the material is part of the published contract).

#import bevy_pbr::{
    forward_io::Vertex,
    mesh_functions,
    mesh_view_types,
    clustered_forward as clustering,
    lighting,
    prepass_utils,
    shadows,
    atmosphere::functions::{calculate_visible_sun_ratio, clamp_to_surface},
    atmosphere::bruneton_functions::transmittance_lut_r_mu_to_uv,
    mesh_view_bindings::{globals, light_probes, lights, view},
    view_transformations::position_world_to_clip,
}
#import bevy_pbr::mesh_view_bindings as view_bindings

#import aqua::cascade::{CREST_SSS_RANGE, CREST_SSS_UNCOMPRESSED, DEBUG_MODE_BEAUTY, DEBUG_MODE_BEER_LAMBERT, DEBUG_MODE_FAR_TIER, DEBUG_MODE_FOAM, DEBUG_MODE_LIGHT_RADIANCE, DEBUG_MODE_REFLECTION, DEBUG_MODE_REFLECTION_FRACTION, DEBUG_MODE_REFRACTION_VALIDITY, DEBUG_MODE_SEA_FLOOR, DEBUG_MODE_TRANSMISSION, DEBUG_MODE_UNREFRACTED, DEBUG_MODE_WATER_PATH, DEBUG_MODE_WAVE_HEIGHT, LUMINANCE_EPSILON, LocalLightSample, MIN_NORMAL_Y, SAFE_LENGTH_SQUARED, advected_world, begin_invocation, capillary_resolved_weight, cascade_layout, effective_flow, far_tier_weight, field_params, godot_fresnel, invocation_extinction, invocation_ripple, invocation_river_state, owning_body, sample_field_flow, sample_field_level, sample_planar_reflection, set_body_optics, set_effective_flow, set_effective_time, set_fragment_river, set_river_ripple, set_xz_footprint, snap_and_transition, surface}

#import aqua::waves::displace::{FFT_JONSWAP_SLOPE_VARIANCE, GERSTNER_SLOPE_VARIANCE, WAVE_NORMALS_SLOPE_VARIANCE, capillary_normal_slope, crest_sss, detail_normal_sample, far_displacement, far_normal_cross, sample_fft_normal_cross}

#import aqua::foam::contract::FOAM_PATTERN_RESOLUTION
#import aqua::foam::shade::{CREST_FOAM_NORMAL_STRENGTH, CREST_FOAM_SPECULAR_BOOST, INV_PI, CREST_FOAM_SPECULAR_FALLOFF, CREST_FOAM_WHITE_COLOR, foam_bubble_colour, local_foam_light, river_streak_density, sample_foam_density, surface_foam_mask}

#import aqua::shore::water::{blended_water_depth, caustic_bed_radiance}
#import bevy_aqua_core::deform::{deform_current}
#import bevy_aqua_core::material::{BodyLightingState, CameraDepthPath, FoamState, LocalLightingState, MediumState, NearSurface, PrimaryLightState, SurfaceVertexOutput, TransmissionState}
#import aqua::light::incident::{GODOT_NORMAL_FADE_RATE, GODOT_NORMAL_MINIMUM_STRENGTH, GODOT_SSS_MODIFIER, GODOT_WATER_ALBEDO, LUMINANCE_WEIGHTS, filtered_primary_light_color, ggx_distribution, local_light_contribution, resolve_primary_light, safe_normalize, sample_diffuse_environment, sample_environment, sample_local_light, smith_masking_shadowing, strongest_incident_directional_light, view_direction}
#import aqua::optics::{camera_depth_path, deep_water_weight, empty_camera_depth_path, far_field_water, resolve_near_surface, resolve_transmission, sample_water_medium, shade_underside, unresolved_wave_roughness}

@vertex
fn vertex(vertex: Vertex) -> SurfaceVertexOutput {
    let deformation = deform_current(vertex, globals.time);
    var out: SurfaceVertexOutput;
    out.world_position = deformation.world_position;
    out.position = position_world_to_clip(deformation.world_position.xyz);
    out.world_normal = deformation.world_normal;
    out.undisplaced_xz = deformation.undisplaced_xz;
    out.sample_data = vec3(
        f32(deformation.lod),
        deformation.sample_alpha,
        deformation.wave_height,
    );
    out.base_world_position = deformation.base_world_position;
    return out;
}

fn prepare_surface_foam(
    in: SurfaceVertexOutput,
    surface_lod: u32,
    foam_density: f32,
    lighting_distance: f32,
) -> FoamState {
    let foam_distance_fade = exp(-lighting_distance * 0.0075);
    let visible_foam_density = foam_density * foam_distance_fade;
    var white_foam_density = visible_foam_density;
    var white_foam = 0.0;
    var shared_depth_path = empty_camera_depth_path();
    var has_shared_depth_path = false;
    if visible_foam_density > 0.0 {
        shared_depth_path = camera_depth_path(in);
        has_shared_depth_path = true;
        let foam_depth = shared_depth_path;
        let shoreline_fade = select(
            1.0,
            clamp(foam_depth.path_length / 0.27, 0.0, 1.0),
            foam_depth.has_background,
        );
        white_foam_density *= shoreline_fade;
        white_foam = surface_foam_mask(
            advected_world(in.undisplaced_xz),
            surface_lod,
            in.sample_data.y,
            white_foam_density,
            vec2(0.0),
        );
    }
    // River bank streaks are independent of the persistent foam buffer:
    // they exist wherever fast water runs close to a bank.
    let streak = river_streak_density(
        invocation_river_state(),
        in.world_position.xz,
        surface_lod,
        in.sample_data.y,
    );
    if streak > 0.0 {
        white_foam_density += streak;
        white_foam += surface_foam_mask(
            advected_world(in.undisplaced_xz),
            surface_lod,
            in.sample_data.y,
            streak,
            vec2(0.0),
        );
    }
    return FoamState(
        visible_foam_density,
        white_foam_density,
        white_foam,
        shared_depth_path,
        has_shared_depth_path,
    );
}

fn directional_scatter(
    in: SurfaceVertexOutput,
    surface_lod: u32,
    near: NearSurface,
    primary: PrimaryLightState,
    to_view: vec3<f32>,
    mode: u32,
) -> vec3<f32> {
    var scatter_colour = vec3(0.0);
    // Crest `OceanEmission.hlsl::ScatterColour`: backlit subsurface tint is
    // driven by horizontal-displacement pinch, not absolute wave height.
    if mode >= DEBUG_MODE_BEAUTY
        && lights.n_directional_lights > 0u {
        let light = lights.directional_lights[0u];
        let light_direction = safe_normalize(
            light.direction_to_light,
            vec3(0.0, 1.0, 0.0),
        );
        // Crest consumes Unity's `_LightColor0` scene radiance directly. Bevy's
        // GPU light is linear RGB times lux, so one view-exposure multiplication
        // converts it to the same pre-tonemap domain as Bevy PBR and atmosphere.
        let light_radiance = primary.radiance;
        let towards_sun = pow(
            max(dot(light_direction, -to_view), 0.0),
            surface.sss.z,
        );
        let pinch = crest_sss(in.undisplaced_xz, surface_lod, in.sample_data.y);
        // Crest keeps a 0.48 SSS pedestal to hide outer-LOD transitions. With
        // physical Bevy lux this would light the entire uncompressed surface;
        // retain only its compression contrast so radiance pierces wave crests.
        let crest_transmission = clamp(
            (pinch - CREST_SSS_UNCOMPRESSED) / CREST_SSS_RANGE,
            0.0,
            1.0,
        );
        let view_vertical = abs(to_view.y);
        let grazing = max(1.0 - view_vertical * view_vertical, 0.0);
        let dot_nv = max(dot(near.lighting_normal, to_view), 2e-5);
        let sss_light_mask = smith_masking_shadowing(dot_nv, surface.sun.y);
        let sss_near = 0.5 * pow(dot_nv, 2.0);
        let sss_height = max(0.0, in.sample_data.z + 2.5)
            * pow(max(dot(light_direction, -to_view), 0.0), 4.0)
            * pow(
                0.5 - 0.5 * dot(light_direction, near.lighting_normal),
                3.0,
            );
        // GodotOceanWaves supplies broad view/body and height/backlight lanes;
        // Crest's Jacobian term remains the concentrated crest variation.
        scatter_colour += (sss_height + sss_near)
            * GODOT_SSS_MODIFIER
            / (1.0 + sss_light_mask)
            * light_radiance
            * GODOT_WATER_ALBEDO;
        scatter_colour += (surface.sss.x + surface.sss.y * towards_sun)
            * surface.sss_tint.rgb
            * light_radiance
            * grazing
            * crest_transmission
            // Budget transmission by irradiance incident on the mean water
            // plane. This preserves Crest pinch while preventing a grazing sun
            // from lighting every compressed crest at normal-incidence energy.
            * max(light_direction.y, 0.0);
    }
    return scatter_colour;
}

fn shade_water_body(
    in: SurfaceVertexOutput,
    near: NearSurface,
    primary: PrimaryLightState,
    medium: MediumState,
    foam: FoamState,
    to_view: vec3<f32>,
    input_body: vec3<f32>,
    mode: u32,
) -> BodyLightingState {
    var body = input_body;
    // Godot's engine substrate adds diffuse sky irradiance behind every
    // material light() function. Crest consumes the same scene ambient for
    // both the white-foam and sub-surface bubble lanes.
    var foam_ambient_radiance = vec3(0.0);
    if mode >= DEBUG_MODE_BEAUTY {
        // Crest consumes Unity SH L0 for foam. Bevy's up-facing diffuse
        // irradiance is the scene-driven equivalent; no constant radiance
        // floor is allowed when that environment is dark or absent.
        foam_ambient_radiance = sample_diffuse_environment(vec3(0.0, 1.0, 0.0));
        if foam.visible_density > 0.0 {
            body += foam_bubble_colour(
                in.world_position.xz,
                in.undisplaced_xz,
                u32(round(in.sample_data.x)),
                in.sample_data.y,
                foam.visible_density,
                near.normal,
                to_view,
                foam_ambient_radiance,
            );
        }
    }
    // Crest's final Fresnel composition supplies the `(1.0 - fresnel)`
    // modulation exactly once.
    let perceptual_roughness = unresolved_wave_roughness(
        in.undisplaced_xz,
        to_view,
        in.sample_data.y,
        near.lighting_normal_strength,
        near.filtered_detail_variance,
    );
    let view_alignment = clamp(dot(near.lighting_normal, to_view), 0.0, 1.0);
    let fresnel = godot_fresnel(view_alignment);
    let foam_distance_fade = exp(-near.lighting_distance * 0.0075);
    let foam_factor = smoothstep(
        0.0,
        1.0,
        medium.foam_density * 0.75,
    ) * foam_distance_fade;
    let foam_roughness = (1.0 - fresnel) * foam_factor;
    let environment_roughness = clamp(
        perceptual_roughness + foam_roughness,
        0.0,
        1.0,
    );
    let has_local_lights = primary.point_start < primary.light_end;
    // Godot uses a 0.4 alpha floor. Aqua's unresolved slope variance adds
    // in quadrature so every direct emitter softens consistently with distance.
    // Skip this emitter-only work when the clustered fragment has no lights.
    var sun_roughness = surface.sun.y;
    if lights.n_directional_lights > 0u || has_local_lights {
        let foam_surface_roughness = clamp(
            surface.sun.y + foam_roughness,
            surface.sun.y,
            1.0,
        );
        sun_roughness = min(sqrt(
            foam_surface_roughness * foam_surface_roughness
                + perceptual_roughness * perceptual_roughness,
        ), 1.0);
    }
    return BodyLightingState(
        body,
        foam_ambient_radiance,
        fresnel,
        foam_roughness,
        environment_roughness,
        sun_roughness,
    );
}

fn shade_environment_and_sun(
    world_position: vec3<f32>,
    surface_level: f32,
    near: NearSurface,
    primary: PrimaryLightState,
    body_lighting: BodyLightingState,
    to_view: vec3<f32>,
) -> vec3<f32> {
    let reflection = reflect(-to_view, near.lighting_normal);
    var reflected_radiance = sample_environment(
        reflection,
        near.lighting_normal,
        body_lighting.environment_roughness,
    );
    let planar = sample_planar_reflection(world_position, surface_level, near.lighting_normal);
    reflected_radiance = mix(reflected_radiance, planar.color, planar.weight);

    // Crest `OceanReflection.hlsl::ApplyReflectionSky`: the directional light
    // is a bounded reflection-vector lobe added before the Fresnel blend.
    if lights.n_directional_lights > 0u {
        let light = lights.directional_lights[0u];
        let light_direction = safe_normalize(
            light.direction_to_light,
            vec3(0.0, 1.0, 0.0),
        );
        let light_luminance = max(
            dot(primary.color, LUMINANCE_WEIGHTS),
            LUMINANCE_EPSILON,
        );
        let light_color = primary.color / light_luminance;
        let light_strength = clamp(light_luminance / surface.reflection.z, 0.0, 1.0);
        let halfway = safe_normalize(
            light_direction + to_view,
            near.lighting_normal,
        );
        let dot_nl = max(dot(near.lighting_normal, light_direction), 2e-5);
        let dot_nv = max(dot(near.lighting_normal, to_view), 2e-5);
        let light_mask = smith_masking_shadowing(dot_nv, body_lighting.sun_roughness);
        let view_mask = smith_masking_shadowing(dot_nl, body_lighting.sun_roughness);
        let distribution = ggx_distribution(
            clamp(dot(near.lighting_normal, halfway), 0.0, 1.0),
            body_lighting.sun_roughness,
        );
        let geometric_attenuation = 1.0 / (1.0 + light_mask + view_mask);
        let sun_specular = distribution
            * geometric_attenuation / (4.0 * dot_nv + 0.1);
        reflected_radiance += sun_specular
            * surface.sun.x
            * light_color
            * light_strength
            * primary.shadow;
    }
    return reflected_radiance;
}

fn shade_local_lights(
    in: SurfaceVertexOutput,
    near: NearSurface,
    primary: PrimaryLightState,
    body_lighting: BodyLightingState,
    foam: FoamState,
    to_view: vec3<f32>,
    reflected_input: vec3<f32>,
    mode: u32,
) -> LocalLightingState {
    let has_local_lights = primary.point_start < primary.light_end;
    let local_sss_enabled = mode >= DEBUG_MODE_BEAUTY && has_local_lights;
    var local_crest_transmission = 0.0;
    if local_sss_enabled {
        let local_pinch = crest_sss(
            in.undisplaced_xz,
            u32(round(in.sample_data.x)),
            in.sample_data.y,
        );
        local_crest_transmission = clamp(
            (local_pinch - CREST_SSS_UNCOMPRESSED) / CREST_SSS_RANGE,
            0.0,
            1.0,
        );
    }
    let view_vertical = abs(to_view.y);
    let local_grazing = max(1.0 - view_vertical * view_vertical, 0.0);
    let foam_active = foam.visible_density > 0.0;
    var foam_normal = near.normal;
    if foam_active {
        let pixel_z = max(-primary.view_z, 0.0);
        let foam_delta = 0.25 * pixel_z / FOAM_PATTERN_RESOLUTION;
        let foam_x = surface_foam_mask(
            advected_world(in.undisplaced_xz),
            u32(round(in.sample_data.x)),
            in.sample_data.y,
            foam.white_density,
            vec2(foam_delta, 0.0),
        );
        let foam_z = surface_foam_mask(
            advected_world(in.undisplaced_xz),
            u32(round(in.sample_data.x)),
            in.sample_data.y,
            foam.white_density,
            vec2(0.0, foam_delta),
        );
        let foam_gradient = vec2(foam_x - foam.white_mask, foam_z - foam.white_mask);
        foam_normal = safe_normalize(
            near.normal + CREST_FOAM_NORMAL_STRENGTH
                * vec3(-foam_gradient.x, 0.0, -foam_gradient.y),
            near.normal,
        );
    }
    var body = body_lighting.body;
    var reflected_radiance = reflected_input;
    var local_foam_radiance = vec3(0.0);
    for (
        var local_index = primary.point_start;
        local_index < primary.spot_start;
        local_index += 1u
    ) {
        let light_id = clustering::get_clusterable_object_id(local_index);
        let sample = sample_local_light(
            light_id,
            false,
            in.world_position.xyz,
            near.normal,
            in.position.xy,
        );
        let contribution = local_light_contribution(
            sample,
            near.lighting_normal,
            to_view,
            in.sample_data.z,
            body_lighting.sun_roughness,
            local_grazing,
            local_crest_transmission,
            local_sss_enabled,
        );
        body += contribution.body;
        reflected_radiance += contribution.reflection;
        if foam_active {
            local_foam_radiance += local_foam_light(sample, foam_normal, to_view);
        }
    }
    for (
        var local_index = primary.spot_start;
        local_index < primary.light_end;
        local_index += 1u
    ) {
        let light_id = clustering::get_clusterable_object_id(local_index);
        let sample = sample_local_light(
            light_id,
            true,
            in.world_position.xyz,
            near.normal,
            in.position.xy,
        );
        let contribution = local_light_contribution(
            sample,
            near.lighting_normal,
            to_view,
            in.sample_data.z,
            body_lighting.sun_roughness,
            local_grazing,
            local_crest_transmission,
            local_sss_enabled,
        );
        body += contribution.body;
        reflected_radiance += contribution.reflection;
        if foam_active {
            local_foam_radiance += local_foam_light(sample, foam_normal, to_view);
        }
    }
    return LocalLightingState(
        body,
        reflected_radiance,
        local_foam_radiance,
        foam_normal,
    );
}

fn compose_water(
    primary: PrimaryLightState,
    body_lighting: BodyLightingState,
    local: LocalLightingState,
    foam: FoamState,
    to_view: vec3<f32>,
    far_water: vec3<f32>,
    far_tier: f32,
    mode: u32,
) -> vec4<f32> {
    // Foam is dielectric diffuse froth, not a second glossy water layer.
    // Godot's foam roughness therefore damps both environment and sun glints.
    let reflected_radiance = local.reflected * (1.0 - body_lighting.foam_roughness);

    // GodotOceanWaves roughness-damped Fresnel; Crest owns final composition.
    let reflection_weight = clamp(body_lighting.fresnel * surface.fresnel.z, 0.0, 1.0);
    if mode == DEBUG_MODE_REFLECTION_FRACTION {
        return vec4(vec3(reflection_weight), 1.0);
    }
    if mode == DEBUG_MODE_REFLECTION {
        return vec4(reflected_radiance * reflection_weight, 1.0);
    }
    var water = mix(local.body, reflected_radiance, reflection_weight);
    if foam.visible_density > 0.0 {
        let mask = CREST_FOAM_WHITE_COLOR.a * foam.white_mask;

        // Crest `OceanFoam.hlsl`: shipped 3D foam lighting. Bevy's scene
        // diffuse irradiance replaces Unity SH L0; no constant ambient term
        // enters the same pre-exposed radiance domain as SSS.
        var foam_light = CREST_FOAM_WHITE_COLOR.rgb * body_lighting.foam_ambient;
        if lights.n_directional_lights > 0u {
            let light = lights.directional_lights[0u];
            let light_direction = safe_normalize(
                light.direction_to_light,
                vec3(0.0, 1.0, 0.0),
            );
            // Stay in the established pre-exposed primary-light lane, but do
            // not let a below-horizon light illuminate wave-facing foam.
            let light_radiance = primary.radiance * max(light_direction.y, 0.0);
            let foam_ndl = max(dot(local.foam_normal, light_direction), 0.0);
            foam_light += CREST_FOAM_WHITE_COLOR.rgb
                * INV_PI * surface.foam.z * light_radiance * foam_ndl;
            let foam_reflection = reflect(-to_view, local.foam_normal);
            foam_light += pow(
                max(dot(foam_reflection, light_direction), 0.0),
                CREST_FOAM_SPECULAR_FALLOFF,
            ) * CREST_FOAM_SPECULAR_BOOST * light_radiance;
        }
        foam_light += local.foam_radiance;
        water = mix(water, foam_light, mask);
    }
    return vec4(mix(water, far_water, far_tier), 1.0);
}

@fragment
fn fragment(
    in: SurfaceVertexOutput,
    @builtin(front_facing) is_front: bool,
) -> @location(0) vec4<f32> {
    // Position derivatives measure adjacent fragments in world XZ. Their
    // maximum length conservatively bounds metres per screen pixel for LOD.
    set_xz_footprint(max(
        length(dpdx(in.world_position.xz)),
        length(dpdy(in.world_position.xz)),
    ));
    // One shared tile set renders every scene: unclaimed texels discard
    // unless the Ocean resource is present, so localized scenes pay fill
    // only where their bodies (and the tiles themselves) exist.
    var slot = 0u;
    if field_params.info.x > 0.5 {
        slot = u32(sample_field_level(in.world_position.xz).y + 0.5);
    }
    let bounded = slot > 0u;
    if !bounded && field_params.info.y < 0.5 {
        discard;
    }
    let params = owning_body(slot);
    let surface_level = select(
        cascade_layout.bed_range.z,
        sample_field_level(in.undisplaced_xz).x,
        bounded,
    );
    begin_invocation(bounded, params);
    set_effective_time(globals.time);
    if bounded {
        // Bodies entirely beyond the far tier cull by extent-vs-distance.
        let distance_to_extent = length(
            params.extent.xy - view.world_position.xz,
        ) - params.extent.w;
        if distance_to_extent > surface.far_tier.y {
            discard;
        }
    }
    // River bodies clip to the baked bank distance. z is the SIGNED margin
    // in metres: negative is outside the channel; w is channel half-width.
    var fragment_river_flow = vec4(0.0, 0.0, 1024.0, 0.0);
    if bounded {
        fragment_river_flow = sample_field_flow(in.world_position.xz);
    }
    if bounded && params.flags.y > 0.5 && fragment_river_flow.z < 0.0 {
        discard;
    }
    set_fragment_river(fragment_river_flow);
    set_effective_flow(select(
        surface.advection.xy,
        fragment_river_flow.xy,
        bounded && params.flags.y > 0.5,
    ));
    // Fresh-water optics override the ocean profile when authored.
    let body_optics = bounded && params.optics_a.w > 0.5;
    set_body_optics(
        select(surface.fog_density.rgb, params.optics_a.rgb, body_optics),
        select(surface.fog_density.w, params.optics_b.x, body_optics),
        select(surface.scatter_tint.rgb, params.optics_c.rgb, body_optics),
        select(surface.far_tier.w, params.optics_b.w, body_optics),
    );
    // Discharge reads as roughness: faster narrows break up more, banks and
    // pools stay glassy. Bank fade eases the multiplier to zero at the edge.
    if bounded && params.flags.y > 0.5 {
        let speed = length(fragment_river_flow.xy);
        // Bank proximity over an 8 m band: ripples calm to the waterline.
        let bank_fade = clamp(fragment_river_flow.z / 8.0, 0.0, 1.0);
        set_river_ripple(bank_fade * clamp(0.8 + 0.75 * speed, 0.8, 2.6));
    } else {
        set_river_ripple(1.0);
    }
    let mode = u32(round(surface.debug.x));
    let surface_lod = u32(round(in.sample_data.x));
    let geometric_normal = safe_normalize(in.world_normal, vec3(0.0, 1.0, 0.0));
    let to_view = view_direction(in.world_position.xyz);
    if !is_front {
        return shade_underside(in, surface_lod, geometric_normal, to_view, mode);
    }
    let far_diagnostic = mode == DEBUG_MODE_FAR_TIER;
    var far_tier = select(
        0.0,
        far_tier_weight(in.base_world_position),
        mode == DEBUG_MODE_BEAUTY || far_diagnostic,
    );
    if far_diagnostic {
        return vec4(vec3(far_tier), 1.0);
    }
    var far_water = vec3(0.0);
    var far_water_depth = 0.0;
    if far_tier > 0.0 {
        // Shallow water keeps the depth-aware near optics. As the bed becomes
        // deep enough to disappear, the cheap far tier becomes fully active.
        far_water_depth = blended_water_depth(in.world_position.xz);
        far_tier *= deep_water_weight(far_water_depth);
    }
    if far_tier > 0.0 {
        far_water = far_field_water(
            in.world_position,
            surface_level,
            geometric_normal,
            to_view,
            in.sample_data.z,
            far_water_depth,
        );
        if far_tier >= 1.0 {
            // Fully far beauty skips detail, clusters, shadows, foam, sampled
            // SSS, local lights, and every transmission/camera-depth read.
            return vec4(far_water, 1.0);
        }
    }

    let near = resolve_near_surface(in, surface_lod, geometric_normal, far_tier, mode);
    let primary = resolve_primary_light(in, near.normal);
    let medium = sample_water_medium(in, surface_lod, mode);
    if mode == DEBUG_MODE_SEA_FLOOR {
        let depth = clamp(medium.water_depth / surface.sea_floor.y, 0.0, 1.0);
        return vec4(1.0 - depth, 0.0, depth, 1.0);
    }
    if mode == DEBUG_MODE_LIGHT_RADIANCE {
        if lights.n_directional_lights == 0u {
            return vec4(0.0, 0.0, 0.0, 1.0);
        }
        let light_radiance = lights.directional_lights[0u].color.rgb * view.exposure;
        return vec4(light_radiance / 16.0, 1.0);
    }
    if mode == DEBUG_MODE_WAVE_HEIGHT {
        let height = clamp(0.5 + 0.5 * in.world_position.y, 0.0, 1.0);
        return vec4(vec3(height), 1.0);
    }
    if mode == DEBUG_MODE_FOAM {
        return vec4(vec3(medium.foam_density), 1.0);
    }

    let foam = prepare_surface_foam(
        in,
        surface_lod,
        medium.foam_density,
        near.lighting_distance,
    );
    let scatter = directional_scatter(
        in,
        surface_lod,
        near,
        primary,
        to_view,
        mode,
    );
    let transmission =
        resolve_transmission(in, near.normal, to_view, medium, foam, primary, mode);
    if transmission.handled {
        return transmission.output;
    }
    let body_lighting = shade_water_body(
        in,
        near,
        primary,
        medium,
        foam,
        to_view,
        transmission.body + scatter,
        mode,
    );
    let reflected = shade_environment_and_sun(
        in.world_position.xyz,
        surface_level,
        near,
        primary,
        body_lighting,
        to_view,
    );
    let local = shade_local_lights(
        in,
        near,
        primary,
        body_lighting,
        foam,
        to_view,
        reflected,
        mode,
    );
    return compose_water(
        primary,
        body_lighting,
        local,
        foam,
        to_view,
        far_water,
        far_tier,
        mode,
    );
}
