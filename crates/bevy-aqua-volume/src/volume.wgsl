#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput
#import aqua::medium::{fresnel_water_to_air, medium_radiance, medium_radiance_oriented, mesh_incident_transmittance, N_WATER, PATH_LENGTH_MAX}
#import aqua::light::environment::sample_environment
#import bevy_pbr::mesh_view_bindings::view
#import bevy_pbr::view_transformations::{
    frag_coord_to_ndc,
    position_ndc_to_world,
}

struct VolumeUniform {
    extinction: vec4<f32>,
    scatter: vec4<f32>,
    environment: vec4<f32>,
    sea: vec4<f32>,
    interface_normal: vec4<f32>,
}

#ifdef MULTISAMPLED
@group(2) @binding(0) var screen_texture: texture_2d<f32>;
@group(2) @binding(1) var depth_texture: texture_depth_multisampled_2d;
#else
@group(2) @binding(0) var screen_texture: texture_2d<f32>;
@group(2) @binding(1) var depth_texture: texture_depth_2d;
#endif
@group(2) @binding(2) var screen_sampler: sampler;
@group(2) @binding(3) var<uniform> volume: VolumeUniform;

struct ViewRay {
    direction: vec3<f32>,
    near_distance: f32,
}

fn view_ray(frag_xy: vec2<f32>) -> ViewRay {
    // Near plane (NDC z = 1). Reverse-Z infinite perspective puts the far
    // plane at infinity, so a depth-0 reconstruct is Inf/NaN.
    let near_world = position_ndc_to_world(frag_coord_to_ndc(vec4(frag_xy, 1.0, 1.0)));
    let to_near = near_world - view.world_position;
    let near_distance = max(length(to_near), 1e-4);
    return ViewRay(to_near / near_distance, near_distance);
}

fn intersect_local_surface_metres(
    origin: vec3<f32>,
    rd: vec3<f32>,
    surface: f32,
    facet_up: vec3<f32>,
) -> f32 {
    // The sampled tangent plane passes through (camera.x, surface, camera.z).
    let numerator = (surface - origin.y) * facet_up.y;
    let denominator = dot(rd, facet_up);
    if abs(denominator) <= 1e-6 {
        return -1.0;
    }
    return numerator / denominator;
}

fn terminal_underside_radiance(incident: vec3<f32>, facet_up: vec3<f32>) -> vec3<f32> {
    let water_normal = -facet_up;
    let reflected_direction = reflect(incident, water_normal);
    let transmitted_direction = refract(incident, water_normal, N_WATER);
    let cos_water = clamp(dot(-incident, water_normal), 0.0, 1.0);
    let fresnel = fresnel_water_to_air(cos_water);
    let reflected = medium_radiance_oriented(
        vec3(0.0),
        reflected_direction,
        PATH_LENGTH_MAX,
        0.0,
        volume.extinction.rgb,
        volume.extinction.w,
        volume.scatter.rgb,
        volume.environment.x,
        facet_up,
    );
    // Do not sample an environment with WGSL's zero TIR direction.
    if dot(transmitted_direction, transmitted_direction) < 1e-8 {
        return reflected;
    }
    let window = sample_environment(transmitted_direction, facet_up, 0.0)
        * (N_WATER * N_WATER);
    return mix(window, reflected, fresnel);
}

@fragment
fn fragment(
    in: FullscreenVertexOutput,
) -> @location(0) vec4<f32> {
    // Screen color is the full backing target, while this pass is clipped to
    // the active camera viewport. Derive backing-texture UV from frag coords.
    let screen_size = vec2<f32>(textureDimensions(screen_texture));
    let screen_uv = in.position.xy / screen_size;
    var scene = textureSample(screen_texture, screen_sampler, screen_uv).rgb;
    let viewport_min = view.viewport.xy;
    let viewport_max = viewport_min + view.viewport.zw;
    if any(in.position.xy < viewport_min) || any(in.position.xy >= viewport_max) {
        // Initialize the complete ping-pong destination while preserving
        // pixels belonging to other camera viewports.
        return vec4(scene, 1.0);
    }
#ifdef MULTISAMPLED
    // The post-process target is single-sampled. Resolve reverse-Z depth to
    // the nearest covered sample rather than using an invalid sample index.
    var raw_depth = 0.0;
    let sample_count = textureNumSamples(depth_texture);
    for (var sample = 0u; sample < sample_count; sample += 1u) {
        raw_depth = max(raw_depth, textureLoad(
            depth_texture,
            vec2<i32>(in.position.xy),
            i32(sample),
        ));
    }
#else
    let raw_depth = textureLoad(depth_texture, vec2<i32>(in.position.xy), 0);
#endif

    // Orthographic rays require per-pixel parallel origins. Until the pass
    // carries those explicitly, leave the image unchanged rather than fan
    // rays from the camera position.
    if view.clip_from_view[3].w == 1.0 {
        return vec4(scene, 1.0);
    }

    let plane = volume.sea.x;
    let camera_y = volume.sea.y;
    if camera_y >= plane {
        return vec4(scene, 1.0);
    }

    let camera = view.world_position;
    let ray = view_ray(in.position.xy);
    let rd_world = ray.direction;
    var t_scene = PATH_LENGTH_MAX;
    if raw_depth > 0.0 {
        let world = position_ndc_to_world(frag_coord_to_ndc(vec4(in.position.xy, raw_depth, 1.0)));
        t_scene = min(length(world - camera), PATH_LENGTH_MAX);
        if volume.environment.y > 0.5 {
            scene *= mesh_incident_transmittance(
                volume.extinction.rgb,
                max(plane - world.y, 0.0),
            );
        }
    }
    var t_end = min(t_scene, PATH_LENGTH_MAX);
    let facet_up = normalize(volume.interface_normal.xyz);
    if dot(rd_world, facet_up) > 1e-6 && raw_depth <= 0.0 {
        // Reconstruct a clipped exit from the camera's sampled local tangent
        // plane. Every upward ray in the unbounded single-sheet ocean must
        // eventually leave the water, so its local plane also closes later
        // no-depth gaps. Bounded bodies retain the stricter sub-near-only path
        // because their open side walls need explicit lateral clipping.
        let t_surface = intersect_local_surface_metres(camera, rd_world, plane, facet_up);
        scene = vec3(0.0);
        if t_surface > 0.0 {
            t_end = min(t_end, t_surface);
            let clipped_before_near = t_surface <= ray.near_distance + 1e-4;
            if clipped_before_near || volume.environment.z > 0.5 {
                scene = terminal_underside_radiance(rd_world, facet_up);
            }
        }
    }
    let d0 = max(plane - camera_y, 0.0);
    return vec4(
        medium_radiance(
            scene,
            rd_world,
            t_end,
            d0,
            volume.extinction.rgb,
            volume.extinction.w,
            volume.scatter.rgb,
            volume.environment.x,
        ),
        1.0,
    );
}
