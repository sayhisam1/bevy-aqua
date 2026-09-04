// Average premultiplied RGB AND depth-derived coverage. Never unpremultiply here.
@group(0) @binding(0) var source: texture_2d<f32>;
@group(0) @binding(1) var output: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let dst_size = textureDimensions(output);
    if any(id.xy >= dst_size) { return; }
    let src_size = textureDimensions(source);
    // Native mip sizes floor-divide. A literal p*2 box drops the final row /
    // column when odd. Integrate the full source footprint instead: exactly
    // four equally weighted texels for even 2D sizes, at most 3x3 for odd sizes.
    // A collapsed axis (size 1) contributes its one texel, not an out-of-bounds load.
    let scale = vec2<f32>(src_size) / vec2<f32>(dst_size);
    let lo = vec2<f32>(id.xy) * scale;
    let hi = vec2<f32>(id.xy + vec2(1u)) * scale;
    let first = vec2<u32>(floor(lo));
    let end = min(vec2<u32>(ceil(hi)), src_size);
    var sum = vec4(0.0);
    for (var y = first.y; y < end.y; y += 1u) {
        for (var x = first.x; x < end.x; x += 1u) {
            let p = vec2<f32>(f32(x), f32(y));
            let overlap = max(min(hi, p + vec2(1.0)) - max(lo, p), vec2(0.0));
            sum += textureLoad(source, vec2<i32>(i32(x), i32(y)), 0)
                * overlap.x * overlap.y;
        }
    }
    textureStore(output, vec2<i32>(id.xy), sum / (scale.x * scale.y));
}
