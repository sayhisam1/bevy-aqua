// Reimplementation of the approach in Crest Shaders/Resources/ShapeCombine.compute.

const LOD_COUNT: u32 = 5u;
const CASCADE_COUNT: u32 = LOD_COUNT + 1u;
const WAVE_COUNT: u32 = 40u;

struct CascadeParams {
    center: vec2<f32>,
    scale: f32,
    texture_res: f32,
    inv_texture_res: f32,
    texel_width: f32,
    weight: f32,
    max_wavelength: f32,
}
struct CascadeLayout {
    cascades: array<CascadeParams, CASCADE_COUNT>,
    center: vec4<f32>,
    // XY bed-map first-texel world origin, ZW inverse world extent.
    bed_transform: vec4<f32>,
    // X height minimum, Y height span (negative = no bed map), Z sea level.
    bed_range: vec4<f32>,
}
struct GerstnerWave {
    direction: vec2<f32>,
    amplitude: f32,
    wave_number: f32,
    angular_frequency: f32,
    phase: f32,
    chop_amplitude: f32,
}
struct AnimWavesUniform {
    cascade_layout: CascadeLayout,
    waves: array<GerstnerWave, WAVE_COUNT>,
    ranges: array<vec4<u32>, LOD_COUNT>,
    time: vec4<f32>,
    // xy: world-space current in m/s; kept identical to the other
    // AnimWavesUniform declarations so one upload serves every consumer.
    flow: vec4<f32>,
}

@group(0) @binding(0) var raw_waves: texture_2d_array<f32>;
@group(0) @binding(1) var previous: texture_2d_array<f32>;
@group(0) @binding(2) var linear_sampler: sampler;
@group(0) @binding(3) var output: texture_storage_2d_array<rgba16float, write>;
@group(0) @binding(4) var<uniform> params: AnimWavesUniform;

fn combine_lod(id: vec3<u32>, slice: u32) {
    let cascade = params.cascade_layout.cascades[slice];
    if id.x >= u32(cascade.texture_res) || id.y >= u32(cascade.texture_res) {
        return;
    }
    var displacement = textureLoad(raw_waves, vec2<i32>(id.xy), i32(slice), 0).xyz;
    if slice + 1u < LOD_COUNT {
        let uv = (vec2<f32>(id.xy) + vec2(0.5)) * cascade.inv_texture_res;
        let coverage = cascade.texel_width * cascade.texture_res;
        let world_xz = coverage * (uv - vec2(0.5)) + cascade.center;
        let next = params.cascade_layout.cascades[slice + 1u];
        let next_coverage = next.texel_width * next.texture_res;
        let next_uv = (world_xz - next.center) / next_coverage + vec2(0.5);
        displacement += textureSampleLevel(
            previous,
            linear_sampler,
            next_uv,
            i32(slice + 1u),
            0.0,
        ).xyz;
    }
    textureStore(output, vec2<i32>(id.xy), i32(slice), vec4(displacement, 0.0));
}

#ifdef COMBINE_1
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) { combine_lod(id, 1u); }
#endif
#ifdef COMBINE_2
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) { combine_lod(id, 2u); }
#endif
#ifdef COMBINE_3
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) { combine_lod(id, 3u); }
#endif
#ifdef COMBINE_4
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) { combine_lod(id, 4u); }
#endif
#ifdef COMBINE_0
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) { combine_lod(id, 0u); }
#endif
