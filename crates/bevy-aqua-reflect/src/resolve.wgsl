@group(0) @binding(0) var source: texture_2d<f32>;
@group(0) @binding(1) var depth: texture_depth_2d;
@group(0) @binding(2) var output: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(output);
    if any(id.xy >= size) { return; }
    let p = vec2<i32>(id.xy);
    // Bevy uses reversed Z. Zero is the cleared background, including sky.
    // Do NOT use source alpha, brightness, or an arbitrary positive epsilon.
    let valid = textureLoad(depth, p, 0) > 0.0;
    let color = textureLoad(source, p, 0).rgb;
    // Explicit branch keeps invalid RGB zero even if source contains NaN.
    var result = vec4(0.0);
    if valid { result = vec4(color, 1.0); }
    textureStore(output, p, result);
}
