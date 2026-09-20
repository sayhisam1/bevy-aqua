// Environment-probe sampling isolated from Aqua material bindings so fullscreen
// passes can reuse the same water interface radiance contract.

#define_import_path aqua::light::environment

#import bevy_pbr::mesh_view_bindings::{light_probes, view}
#import bevy_pbr::mesh_view_bindings as view_bindings

const ENV_SAFE_LENGTH_SQUARED: f32 = 1e-8;
const ENV_LUMINANCE_EPSILON: f32 = 0.0001;

fn environment_safe_normalize(value: vec3<f32>, fallback: vec3<f32>) -> vec3<f32> {
    let length_squared = dot(value, value);
    let normalized = value * inverseSqrt(max(length_squared, ENV_SAFE_LENGTH_SQUARED));
    return select(fallback, normalized, length_squared > ENV_SAFE_LENGTH_SQUARED);
}

fn quat_rotate(rotation: vec4<f32>, value: vec3<f32>) -> vec3<f32> {
    return value + 2.0 * cross(rotation.xyz, cross(rotation.xyz, value) + rotation.w * value);
}

fn sample_environment(
    reflection: vec3<f32>,
    surface_normal: vec3<f32>,
    perceptual_roughness: f32,
) -> vec3<f32> {
    // Godot renderer substrate: bend rough reflections toward the normal and
    // suppress the below-horizon lobe before sampling the environment.
    let bent_reflection = mix(
        reflection,
        surface_normal,
        perceptual_roughness * perceptual_roughness,
    );
    let horizon = min(1.0 + dot(bent_reflection, surface_normal), 1.0);
    // With no scene environment there is no radiance to reflect.
    var radiance = vec3(0.0);
#ifdef ENVIRONMENT_MAP
    if light_probes.view_cubemap_index >= 0 {
        // Work in probe-local space: arbitrary probe rotation must not tilt this
        // guard back into the cubemap's ground hemisphere. Raise the rough lobe
        // by its RMS slope cone. Where the original cone crosses the horizon,
        // blend to a ground-free mip-0 sample (or zero if no probe exists)
        // instead of a contaminated rough mip.
        var probe_reflection = quat_rotate(
            light_probes.view_rotation,
            environment_safe_normalize(bent_reflection, surface_normal),
        );
        let horizon_sine = perceptual_roughness
            / sqrt(1.0 + perceptual_roughness * perceptual_roughness);
        let roughness_enabled = perceptual_roughness > 0.0;
        let guard_threshold = max(horizon_sine, ENV_LUMINANCE_EPSILON);
        let ground_risk = select(
            0.0,
            1.0 - smoothstep(
                guard_threshold,
                2.0 * guard_threshold,
                probe_reflection.y,
            ),
            roughness_enabled,
        );
        probe_reflection.y = select(
            probe_reflection.y,
            max(probe_reflection.y, horizon_sine),
            roughness_enabled,
        );
        var sample_direction = environment_safe_normalize(
            probe_reflection,
            vec3(0.0, 1.0, 0.0),
        );
        sample_direction.z = -sample_direction.z;
        let mip = sqrt(perceptual_roughness)
            * f32(light_probes.smallest_specular_mip_level_for_view);
#ifdef MULTIPLE_LIGHT_PROBES_IN_ARRAY
        radiance = textureSampleLevel(
            view_bindings::specular_environment_maps[u32(light_probes.view_cubemap_index)],
            view_bindings::environment_map_sampler,
            sample_direction,
            mip,
        ).rgb * light_probes.intensity_for_view * view.exposure;
#else
        radiance = textureSampleLevel(
            view_bindings::specular_environment_map,
            view_bindings::environment_map_sampler,
            sample_direction,
            mip,
        ).rgb * light_probes.intensity_for_view * view.exposure;
#endif
        if ground_risk > 0.0 {
            var ground_free_sky = vec3(0.0);
#ifdef MULTIPLE_LIGHT_PROBES_IN_ARRAY
            ground_free_sky = textureSampleLevel(
                view_bindings::specular_environment_maps[u32(light_probes.view_cubemap_index)],
                view_bindings::environment_map_sampler,
                sample_direction,
                0.0,
            ).rgb * light_probes.intensity_for_view * view.exposure;
#else
            ground_free_sky = textureSampleLevel(
                view_bindings::specular_environment_map,
                view_bindings::environment_map_sampler,
                sample_direction,
                0.0,
            ).rgb * light_probes.intensity_for_view * view.exposure;
#endif
            radiance = mix(radiance, ground_free_sky, ground_risk);
        }
    }
#endif
    return radiance * horizon * horizon;
}

fn sample_diffuse_environment(surface_normal: vec3<f32>) -> vec3<f32> {
    // Material colours are reflectance coefficients, never fallback emitters.
    var irradiance = vec3(0.0);
#ifdef ENVIRONMENT_MAP
    if light_probes.view_cubemap_index >= 0 {
        var sample_direction = quat_rotate(
            light_probes.view_rotation,
            surface_normal,
        );
        sample_direction.y = max(sample_direction.y, 0.0);
        sample_direction = environment_safe_normalize(
            sample_direction,
            vec3(0.0, 1.0, 0.0),
        );
        sample_direction.z = -sample_direction.z;
#ifdef MULTIPLE_LIGHT_PROBES_IN_ARRAY
        irradiance = textureSampleLevel(
            view_bindings::diffuse_environment_maps[u32(light_probes.view_cubemap_index)],
            view_bindings::environment_map_sampler,
            sample_direction,
            0.0,
        ).rgb * light_probes.intensity_for_view * view.exposure;
#else
        irradiance = textureSampleLevel(
            view_bindings::diffuse_environment_map,
            view_bindings::environment_map_sampler,
            sample_direction,
            0.0,
        ).rgb * light_probes.intensity_for_view * view.exposure;
#endif
    }
#endif
    return irradiance;
}
