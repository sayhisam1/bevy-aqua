// Consolidates the alternating ShapeCombine outputs into AnimWaves.

const LOD_COUNT: u32 = 5u;

@group(0) @binding(0) var scratch_a: texture_2d_array<f32>;
@group(0) @binding(1) var scratch_b: texture_2d_array<f32>;
@group(0) @binding(2) var output: texture_storage_2d_array<rgba16float, write>;

@compute @workgroup_size(8, 8, 1)
fn gather(@builtin(global_invocation_id) id: vec3<u32>) {
    if any(id.xy >= vec2<u32>(256u)) || id.z >= LOD_COUNT {
        return;
    }
    let value = select(
        textureLoad(scratch_a, vec2<i32>(id.xy), i32(id.z), 0),
        textureLoad(scratch_b, vec2<i32>(id.xy), i32(id.z), 0),
        id.z % 2u == 0u,
    );
    textureStore(output, vec2<i32>(id.xy), i32(id.z), value);
}
