// Shared data transferred between Aqua's terminal material and layered
// lighting/optics modules. This file contains types only.

#define_import_path bevy_aqua_core::material

struct SurfaceVertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) world_position: vec4<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) undisplaced_xz: vec2<f32>,
    @location(3) sample_data: vec3<f32>,
    @location(4) base_world_position: vec3<f32>,
}

struct CameraDepthPath {
    path_length: f32,
    screen_uv: vec2<f32>,
    scene_z: f32,
    hit_y: f32,
    has_background: bool,
}

struct TransmissionSample {
    uv: vec2<f32>,
    path_length: f32,
    hit_y: f32,
    has_background: bool,
    refraction_valid: bool,
}

struct NearSurface {
    normal: vec3<f32>,
    lighting_normal: vec3<f32>,
    lighting_distance: f32,
    lighting_normal_strength: f32,
    filtered_detail_variance: f32,
}

struct PrimaryLightState {
    view_z: f32,
    point_start: u32,
    spot_start: u32,
    light_end: u32,
    shadow: f32,
    color: vec3<f32>,
    radiance: vec3<f32>,
}

struct MediumState {
    water_depth: f32,
    foam_density: f32,
}

struct FoamState {
    visible_density: f32,
    white_density: f32,
    white_mask: f32,
    depth_path: CameraDepthPath,
    has_depth_path: bool,
}

struct TransmissionState {
    body: vec3<f32>,
    output: vec4<f32>,
    handled: bool,
}

struct BodyLightingState {
    body: vec3<f32>,
    foam_ambient: vec3<f32>,
    fresnel: f32,
    foam_roughness: f32,
    environment_roughness: f32,
    sun_roughness: f32,
}

struct LocalLightingState {
    body: vec3<f32>,
    reflected: vec3<f32>,
    foam_radiance: vec3<f32>,
    foam_normal: vec3<f32>,
}

