// Incident-light, BRDF, and environment-sampling primitives. Final water
// composition remains in the terminal material to keep imports one-way.

#define_import_path aqua::light::incident

#import bevy_pbr::{
    clustered_forward as clustering,
    lighting,
    mesh_view_types,
    shadows,
    atmosphere::functions::{calculate_visible_sun_ratio, clamp_to_surface},
    atmosphere::bruneton_functions::transmittance_lut_r_mu_to_uv,
    mesh_view_bindings::{light_probes, lights, view},
}
#import bevy_pbr::mesh_view_bindings as view_bindings
#import aqua::cascade::{LUMINANCE_EPSILON, LocalLightSample, SAFE_LENGTH_SQUARED, invocation_sun_roughness, surface}
#import bevy_aqua_core::material::{PrimaryLightState, SurfaceVertexOutput}

const PI: f32 = 3.14159265359;

const LUMINANCE_WEIGHTS: vec3<f32> = vec3(0.2126, 0.7152, 0.0722);

const GODOT_NORMAL_FADE_RATE: f32 = 0.0175;
const GODOT_NORMAL_MINIMUM_STRENGTH: f32 = 0.015;
const GODOT_WATER_ALBEDO: vec3<f32> = vec3(
    0.010022826, 0.01960665, 0.02721178,
);
const GODOT_SSS_MODIFIER: vec3<f32> = vec3(0.9, 1.15, 0.85);

// Smith GGX Lambda, used by height-correlated G2 = 1 / (1 + LambdaV + LambdaL).
// Match ggx_distribution's alpha and argument order; a Beckmann approximation
// would under-mask the long tails of the GGX distribution.
fn smith_masking_shadowing(cos_theta: f32, alpha: f32) -> f32 {
    let cosine = clamp(cos_theta, 1e-5, 1.0);
    let tangent_squared = max(1.0 - cosine * cosine, 0.0) / (cosine * cosine);
    return 0.5 * (sqrt(1.0 + alpha * alpha * tangent_squared) - 1.0);
}

fn ggx_distribution(cos_theta: f32, alpha: f32) -> f32 {
    let alpha_squared = alpha * alpha;
    let denominator = 1.0
        + (alpha_squared - 1.0) * cos_theta * cos_theta;
    return alpha_squared / (PI * denominator * denominator);
}

fn safe_normalize(value: vec3<f32>, fallback: vec3<f32>) -> vec3<f32> {
    let length_squared = dot(value, value);
    let normalized = value * inverseSqrt(max(length_squared, SAFE_LENGTH_SQUARED));
    return select(fallback, normalized, length_squared > SAFE_LENGTH_SQUARED);
}

struct LocalLightContribution {
    body: vec3<f32>,
    reflection: vec3<f32>,
}

fn sample_local_light(
    light_id: u32,
    is_spot: bool,
    world_position: vec3<f32>,
    surface_normal: vec3<f32>,
    frag_coord: vec2<f32>,
) -> LocalLightSample {
    let light = view_bindings::clustered_lights.data[light_id];
    let to_light = light.position_radius.xyz - world_position;
    let distance_squared = dot(to_light, to_light);
    let direction = safe_normalize(to_light, vec3(0.0, 1.0, 0.0));
    var attenuation = lighting::getDistanceAttenuation(
        distance_squared,
        light.color_inverse_square_range.w,
    );
    if is_spot {
        var spot_direction = vec3(
            light.light_custom_data.x,
            0.0,
            light.light_custom_data.y,
        );
        spot_direction.y = sqrt(max(
            0.0,
            1.0 - spot_direction.x * spot_direction.x
                - spot_direction.z * spot_direction.z,
        ));
        if (light.flags
            & mesh_view_types::POINT_LIGHT_FLAGS_SPOT_LIGHT_Y_NEGATIVE) != 0u {
            spot_direction.y = -spot_direction.y;
        }
        let cosine = dot(-spot_direction, direction);
        let cone = clamp(
            cosine * light.light_custom_data.z + light.light_custom_data.w,
            0.0,
            1.0,
        );
        attenuation *= cone * cone;
    }
    var shadow = 1.0;
    if (light.flags
        & mesh_view_types::POINT_LIGHT_FLAGS_SHADOWS_ENABLED_BIT) != 0u {
        if is_spot {
            shadow = shadows::fetch_spot_shadow(
                light_id,
                vec4(world_position, 1.0),
                surface_normal,
                light.shadow_map_near_z,
                frag_coord,
            );
        } else {
            shadow = shadows::fetch_point_shadow(
                light_id,
                vec4(world_position, 1.0),
                surface_normal,
                frag_coord,
            );
        }
    }
    return LocalLightSample(
        direction,
        light.color_inverse_square_range.rgb
            * attenuation * shadow * view.exposure,
    );
}

fn local_light_contribution(
    sample: LocalLightSample,
    lighting_normal: vec3<f32>,
    to_view: vec3<f32>,
    wave_height: f32,
    sun_roughness: f32,
    grazing: f32,
    crest_transmission: f32,
    sss_enabled: bool,
) -> LocalLightContribution {
    let dot_nl = max(dot(lighting_normal, sample.direction), 0.0);
    let dot_nv = max(dot(lighting_normal, to_view), 2e-5);
    var body = 0.5 * dot_nl * sample.radiance * GODOT_WATER_ALBEDO;
    if sss_enabled {
        let towards_light = pow(
            max(dot(sample.direction, -to_view), 0.0),
            surface.sss.z,
        );
        let light_mask = smith_masking_shadowing(dot_nv, invocation_sun_roughness());
        let sss_near = 0.5 * pow(dot_nv, 2.0);
        let sss_height = max(0.0, wave_height + 2.5)
            * pow(max(dot(sample.direction, -to_view), 0.0), 4.0)
            * pow(
                0.5 - 0.5 * dot(sample.direction, lighting_normal),
                3.0,
            );
        body += (sss_height + sss_near)
            * GODOT_SSS_MODIFIER
            / (1.0 + light_mask)
            * sample.radiance
            * GODOT_WATER_ALBEDO;
        body += (surface.sss.x + surface.sss.y * towards_light)
            * surface.sss_tint.rgb
            * sample.radiance
            * grazing
            * crest_transmission
            * max(sample.direction.y, 0.0);
    }

    var reflection = vec3(0.0);
    if dot_nl > 0.0 {
        let halfway = safe_normalize(
            sample.direction + to_view,
            lighting_normal,
        );
        let light_mask = smith_masking_shadowing(dot_nv, sun_roughness);
        let view_mask = smith_masking_shadowing(max(dot_nl, 2e-5), sun_roughness);
        let distribution = ggx_distribution(
            clamp(dot(lighting_normal, halfway), 0.0, 1.0),
            sun_roughness,
        );
        let geometric_attenuation = 1.0 / (1.0 + light_mask + view_mask);
        reflection = distribution
            * geometric_attenuation
            / (4.0 * dot_nv + 0.1)
            * surface.sun.x
            * sample.radiance;
    }
    return LocalLightContribution(body, reflection);
}



fn view_direction(world_position: vec3<f32>) -> vec3<f32> {
    let orthographic = view.clip_from_view[3].w == 1.0;
    let orthographic_direction = vec3(
        view.clip_from_world[0].z,
        view.clip_from_world[1].z,
        view.clip_from_world[2].z,
    );
    let perspective_direction = view.world_position.xyz - world_position;
    return safe_normalize(
        select(perspective_direction, orthographic_direction, orthographic),
        vec3(0.0, 1.0, 0.0),
    );
}





fn filtered_primary_light_color(
    world_position: vec4<f32>,
    direction_to_light: vec3<f32>,
    sun_disk_angular_size: f32,
    raw_color: vec3<f32>,
) -> vec3<f32> {
    var color = raw_color;
#ifdef ATMOSPHERE
    if surface.sun.z > 0.5 {
        let atmosphere = view_bindings::atmosphere;
        let atmosphere_position = (
            atmosphere.world_to_atmosphere * world_position
        ).xyz;
        let clamped_position = clamp_to_surface(
            atmosphere,
            atmosphere_position,
        );
        let radius = length(clamped_position);
        let local_up = normalize(clamped_position);
        let light_direction = safe_normalize(
            direction_to_light,
            vec3(0.0, 1.0, 0.0),
        );
        let mu_light = dot(light_direction, local_up);
        let uv = transmittance_lut_r_mu_to_uv(
            atmosphere,
            radius,
            mu_light,
        );
        let transmittance = textureSampleLevel(
            view_bindings::atmosphere_transmittance_texture,
            view_bindings::atmosphere_transmittance_sampler,
            uv,
            0.0,
        ).rgb;
        let sun_visibility = calculate_visible_sun_ratio(
            atmosphere,
            radius,
            mu_light,
            sun_disk_angular_size,
        );
        color *= transmittance * sun_visibility;
    }
#endif
    return color;
}

fn resolve_primary_light(
    in: SurfaceVertexOutput,
    normal: vec3<f32>,
) -> PrimaryLightState {
    let view_z = (view.view_from_world * in.world_position).z;
    let is_orthographic = view.clip_from_view[3].w == 1.0;
    let cluster_index = clustering::view_fragment_cluster_index(
        in.position.xy,
        view_z,
        is_orthographic,
    );
    let ranges = clustering::unpack_clusterable_object_index_ranges(cluster_index);
    var primary_light_shadow = 1.0;
    var primary_light_color = vec3(0.0);
    if lights.n_directional_lights > 0u {
        let light = lights.directional_lights[0u];
        primary_light_color = filtered_primary_light_color(
            in.world_position,
            light.direction_to_light,
            light.sun_disk_angular_size,
            light.color.rgb,
        );
        if (light.flags
            & mesh_view_types::DIRECTIONAL_LIGHT_FLAGS_SHADOWS_ENABLED_BIT) != 0u {
            primary_light_shadow = shadows::fetch_directional_shadow(
                0u,
                in.world_position,
                normal,
                view_z,
                in.position.xy,
            );
        }
    }
    let primary_light_radiance =
        primary_light_color * view.exposure * primary_light_shadow;
    return PrimaryLightState(
        view_z,
        ranges.first_point_light_index_offset,
        ranges.first_spot_light_index_offset,
        ranges.first_reflection_probe_index_offset,
        primary_light_shadow,
        primary_light_color,
        primary_light_radiance,
    );
}

struct IncidentDirectionalLight {
    valid: bool,
    direction: vec3<f32>,
    color: vec3<f32>,
    shadow: f32,
}

fn strongest_incident_directional_light(
    in: SurfaceVertexOutput,
    shadow_normal: vec3<f32>,
    view_z: f32,
) -> IncidentDirectionalLight {
    var selected = false;
    var selected_index = 0u;
    var selected_direction = vec3(0.0, -1.0, 0.0);
    var best_surface_illuminance = 0.0;
    for (var index = 0u; index < lights.n_directional_lights; index += 1u) {
        let light = lights.directional_lights[index];
        let direction = safe_normalize(
            light.direction_to_light,
            vec3(0.0, -1.0, 0.0),
        );
        if direction.y <= 0.0 {
            continue;
        }
        let surface_illuminance = direction.y * dot(
            max(light.color.rgb, vec3(0.0)),
            vec3(0.2126, 0.7152, 0.0722),
        );
        if surface_illuminance > best_surface_illuminance {
            selected = true;
            selected_index = index;
            selected_direction = direction;
            best_surface_illuminance = surface_illuminance;
        }
    }
    if !selected {
        return IncidentDirectionalLight(
            false,
            selected_direction,
            vec3(0.0),
            0.0,
        );
    }
    let light = lights.directional_lights[selected_index];
    let color = filtered_primary_light_color(
        in.world_position,
        light.direction_to_light,
        light.sun_disk_angular_size,
        light.color.rgb,
    );
    var shadow = 1.0;
    if (light.flags
        & mesh_view_types::DIRECTIONAL_LIGHT_FLAGS_SHADOWS_ENABLED_BIT) != 0u {
        shadow = shadows::fetch_directional_shadow(
            selected_index,
            in.world_position,
            shadow_normal,
            view_z,
            in.position.xy,
        );
    }
    return IncidentDirectionalLight(
        true,
        selected_direction,
        color,
        shadow,
    );
}

