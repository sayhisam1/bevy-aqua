#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput
#import aqua::medium::{medium_radiance, mesh_incident_transmittance, PATH_LENGTH_MAX}
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
}

#ifdef MULTISAMPLED
@group(1) @binding(0) var screen_texture: texture_2d<f32>;
@group(1) @binding(1) var depth_texture: texture_depth_multisampled_2d;
#else
@group(1) @binding(0) var screen_texture: texture_2d<f32>;
@group(1) @binding(1) var depth_texture: texture_depth_2d;
#endif
@group(1) @binding(2) var screen_sampler: sampler;
@group(1) @binding(3) var<uniform> volume: VolumeUniform;

fn view_ray_direction(frag_xy: vec2<f32>) -> vec3<f32> {
    // Near plane (NDC z = 1). Reverse-Z infinite perspective puts the far
    // plane at infinity, so a depth-0 reconstruct is Inf/NaN.
    let near_world = position_ndc_to_world(frag_coord_to_ndc(vec4(frag_xy, 1.0, 1.0)));
    let dir = near_world - view.world_position;
    return dir / max(length(dir), 1e-4);
}

fn intersect_surface_metres(origin: vec3<f32>, rd: vec3<f32>, t_max: f32, surface: f32) -> f32 {
    if rd.y <= 1e-5 {
        return t_max;
    }
    return clamp((surface - origin.y) / rd.y, 0.0, t_max);
}

@fragment
fn fragment(
#ifdef MULTISAMPLED
    @builtin(sample_index) sample_index: u32,
#endif
    in: FullscreenVertexOutput,
) -> @location(0) vec4<f32> {
    var scene = textureSample(screen_texture, screen_sampler, in.uv).rgb;
#ifdef MULTISAMPLED
    let raw_depth = textureLoad(depth_texture, vec2<i32>(in.position.xy), i32(sample_index));
#else
    let raw_depth = textureLoad(depth_texture, vec2<i32>(in.position.xy), 0);
#endif

    let plane = volume.sea.x;
    let camera_y = volume.sea.y;
    if camera_y >= plane {
        return vec4(scene, 1.0);
    }

    let camera = view.world_position;
    let rd_world = view_ray_direction(in.position.xy);
    var t_scene = PATH_LENGTH_MAX;
    if raw_depth > 0.0 {
        let world = position_ndc_to_world(frag_coord_to_ndc(vec4(in.position.xy, raw_depth, 1.0)));
        t_scene = min(length(world - camera), PATH_LENGTH_MAX);
        scene *= mesh_incident_transmittance(
            volume.extinction.rgb,
            max(plane - world.y, 0.0),
        );
    }
    let t_surface = intersect_surface_metres(camera, rd_world, PATH_LENGTH_MAX, plane);
    var t_end = min(t_scene, PATH_LENGTH_MAX);
    if rd_world.y > 0.0 && raw_depth <= 0.0 {
        // No mesh hit looking up: integrate to the mean plane and do not
        // treat the sky as in-water radiance. A surface hit keeps its
        // underside colour and path length.
        t_end = min(t_end, t_surface);
        scene = vec3(0.0);
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
