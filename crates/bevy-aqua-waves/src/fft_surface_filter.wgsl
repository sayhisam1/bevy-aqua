// Dense channel averages preserve derivatives and their second moments.
// Do not normalize either lane. Each dispatch reads one mip view
// and writes the next. No sparse large-footprint sampling or extra FFT.
@group(0) @binding(0) var source: texture_2d_array<f32>;
@group(0) @binding(1) var output: texture_storage_2d_array<rgba16float, write>;
@compute @workgroup_size(8, 8, 1)
fn filter_surface(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(output);
    if any(id.xy >= size) || id.z >= textureNumLayers(output) { return; }
    let xy = vec2<i32>(2u * id.xy);
    let layer = i32(id.z);
    let mean = 0.25 * (
        textureLoad(source, xy, layer, 0)
        + textureLoad(source, xy + vec2(1, 0), layer, 0)
        + textureLoad(source, xy + vec2(0, 1), layer, 0)
        + textureLoad(source, xy + vec2(1, 1), layer, 0)
    );
    textureStore(output, vec2<i32>(id.xy), layer, mean);
}
