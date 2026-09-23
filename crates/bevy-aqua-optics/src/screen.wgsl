// Screen-space reconstruction and opaque-scene sampling shared by
// transmission and screen-space reflections.

#define_import_path aqua::screen

#import bevy_pbr::mesh_view_bindings::view
#import bevy_pbr::mesh_view_bindings as view_bindings
#import aqua::cascade::LUMINANCE_EPSILON

// Reconstructs a view-space position from viewport UV and reverse-Z depth.
fn camera_view_position(uv: vec2<f32>, raw_depth: f32) -> vec3<f32> {
    let ndc = vec3(uv * vec2(2.0, -2.0) + vec2(-1.0, 1.0), raw_depth);
    let position = view.view_from_clip * vec4(ndc, 1.0);
    return position.xyz / max(position.w, LUMINANCE_EPSILON);
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
