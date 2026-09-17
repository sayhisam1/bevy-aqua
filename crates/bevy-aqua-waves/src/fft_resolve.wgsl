#import bevy_aqua_core::waves_sample::{CascadeLayout, LOD_COUNT}

const FFT_RESOLUTION: u32 = 256u;
const ATTENUATION_BINS: u32 = 4u;
struct FftUniform {
    cascade_layout: CascadeLayout,
    params: vec4<f32>,
    // x: active attenuation-bin count (1 or ATTENUATION_BINS).
    mode: vec4<f32>,
}

// Decoded "no bed data" depth: matches a cleared full-depth capture texel.
const NO_BED_DEPTH: f32 = 256.0;

/// Water depth under the sea level from the game's bed height map. Outside
/// the mapped area (or with no map at all) the sample keeps the deep default,
/// mirroring a full-depth capture texel. Nearest-texel read: the shoaling
/// profile is smooth enough that one bed texel per cascade texel suffices.
fn bed_water_depth(world_xz: vec2<f32>) -> f32 {
    let range = fft.cascade_layout.bed_range;
    if range.y < 0.0 {
        return NO_BED_DEPTH;
    }
    let uv = (world_xz - fft.cascade_layout.bed_transform.xy)
        * fft.cascade_layout.bed_transform.zw;
    if any(uv < vec2(0.0)) || any(uv > vec2(1.0)) {
        return NO_BED_DEPTH;
    }
    let dimensions = vec2<u32>(textureDimensions(bed_height));
    let maximum = vec2<i32>(dimensions - vec2<u32>(1u));
    let texel = clamp(vec2<i32>(round(uv * vec2<f32>(maximum))), vec2(i32(0)), maximum);
    let height = f32(textureLoad(bed_height, texel, 0).r) * range.y + range.x;
    return max(range.z - height, 0.0);
}

@group(0) @binding(0) var height_x: texture_2d_array<f32>;
@group(0) @binding(1) var z_field: texture_2d_array<f32>;
@group(0) @binding(2) var bed_height: texture_2d<f32>;
@group(0) @binding(3) var output: texture_storage_2d_array<rgba16float, write>;
@group(0) @binding(4) var<uniform> fft: FftUniform;

fn shoaling_weights(depth: f32, wave_number: f32) -> vec2<f32> {
    let relative_depth = clamp(depth * wave_number / 3.141592653589793, 0.0, 1.0);
    let vertical_base = smoothstep(0.0, 1.0, relative_depth);
    let breaker = 4.0 * vertical_base * (1.0 - vertical_base);
    let vertical = vertical_base * (1.0 + 0.18 * breaker);
    let chop_ratio = mix(0.55, 1.0, smoothstep(0.15, 0.85, vertical_base));
    return vec2(vertical * chop_ratio, vertical);
}

@compute @workgroup_size(8, 8, 1)
fn resolve(@builtin(global_invocation_id) id: vec3<u32>) {
    if any(id.xy >= vec2<u32>(FFT_RESOLUTION)) || id.z >= LOD_COUNT { return; }
    let bins = u32(fft.mode.x);
    let normalization = 1.0 / f32(FFT_RESOLUTION * FFT_RESOLUTION);
    let cascade = fft.cascade_layout.cascades[id.z];
    var displacement = vec3(0.0);
    if bins == 1u {
        // Deep-water fast path: shoaling is exactly vec2(1.0) for every bin,
        // so the four-bin sum reduces to this one layer with no depth read.
        let packed = textureLoad(height_x, vec2<i32>(id.xy), i32(id.z), 0);
        let z = textureLoad(z_field, vec2<i32>(id.xy), i32(id.z), 0).x;
        displacement = vec3(packed.z, packed.x, z) * normalization;
    } else {
        let coverage = cascade.texel_width * cascade.texture_res;
        let uv = (vec2<f32>(id.xy) + vec2(0.5)) / vec2<f32>(cascade.texture_res);
        let depth = bed_water_depth(coverage * (uv - vec2(0.5)) + cascade.center);
        for (var bin = 0u; bin < bins; bin += 1u) {
            let layer = id.z * bins + bin;
            let packed = textureLoad(height_x, vec2<i32>(id.xy), i32(layer), 0);
            let z = textureLoad(z_field, vec2<i32>(id.xy), i32(layer), 0).x;
            let octave_fraction = (f32(bin) + 0.5) / f32(bins);
            let representative_wavelength = 0.5 * cascade.max_wavelength
                * exp2(octave_fraction);
            let wave_number = 2.0 * 3.141592653589793 / representative_wavelength;
            let shoaling = shoaling_weights(depth, wave_number);
            let attenuation = mix(vec2(1.0), shoaling, fft.params.y);
            displacement += vec3(
                packed.z * attenuation.x,
                packed.x * attenuation.y,
                z * attenuation.x,
            ) * normalization;
        }
    }
    textureStore(output, vec2<i32>(id.xy), i32(id.z), vec4(displacement, 0.0));
}
