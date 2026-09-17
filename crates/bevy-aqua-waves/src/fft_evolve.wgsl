#import bevy_aqua_core::waves_sample::{CascadeLayout, LOD_COUNT}

const FFT_RESOLUTION: u32 = 256u;
const ATTENUATION_BINS: u32 = 4u;
const FIELD_LAYERS: u32 = LOD_COUNT * ATTENUATION_BINS;
const GRAVITY: f32 = 9.81;
const CHOP: f32 = 0.8;

struct FftUniform {
    cascade_layout: CascadeLayout,
    params: vec4<f32>,
    // x: active attenuation-bin count (1 or ATTENUATION_BINS).
    mode: vec4<f32>,
}

@group(0) @binding(0) var initial_spectrum: texture_2d_array<f32>;
@group(0) @binding(1) var height_x: texture_storage_2d_array<rgba32float, write>;
@group(0) @binding(2) var z_field: texture_storage_2d_array<rgba32float, write>;
@group(0) @binding(3) var<uniform> fft: FftUniform;

fn complex_multiply(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}
fn conjugate(value: vec2<f32>) -> vec2<f32> { return vec2(value.x, -value.y); }
fn signed_frequency(value: u32) -> i32 {
    return select(i32(value), i32(value) - i32(FFT_RESOLUTION), value > FFT_RESOLUTION / 2u);
}

@compute @workgroup_size(8, 8, 1)
fn evolve(@builtin(global_invocation_id) id: vec3<u32>) {
    let bins = u32(fft.mode.x);
    if any(id.xy >= vec2<u32>(FFT_RESOLUTION)) || id.z >= LOD_COUNT * bins { return; }
    let cascade_index = id.z / bins;
    let attenuation_bin = id.z % bins;
    let cascade = fft.cascade_layout.cascades[cascade_index];
    let period = cascade.texel_width * cascade.texture_res;
    let k = (2.0 * 3.141592653589793 / period)
        * vec2<f32>(f32(signed_frequency(id.x)), f32(signed_frequency(id.y)));
    let k_length = length(k);
    let mirror = vec2<i32>(
        i32((FFT_RESOLUTION - id.x) % FFT_RESOLUTION),
        i32((FFT_RESOLUTION - id.y) % FFT_RESOLUTION),
    );
    let h0 = textureLoad(initial_spectrum, vec2<i32>(id.xy), i32(cascade_index), 0).xy;
    let h0_mirror = textureLoad(initial_spectrum, mirror, i32(cascade_index), 0).xy;
    let omega = sqrt(GRAVITY * k_length);
    let phase = omega * fft.params.x;
    let negative_phase = vec2(cos(phase), -sin(phase));
    let positive_phase = vec2(cos(phase), sin(phase));
    var height = complex_multiply(h0, negative_phase)
        + complex_multiply(conjugate(h0_mirror), positive_phase);
    // Keep each Fourier component in a narrow quarter-octave field. Resolve can then
    // apply local half-wavelength attenuation before summing the fields.
    if bins == 1u {
        // Single-bin deep-water path keeps every component; DC stays suppressed.
        if k_length <= 0.0 { height = vec2(0.0); }
    } else if k_length > 0.0 {
        let wavelength = 2.0 * 3.141592653589793 / k_length;
        let octave = log2(wavelength / (0.5 * cascade.max_wavelength));
        let component_bin = min(u32(max(floor(octave * f32(ATTENUATION_BINS)), 0.0)),
            ATTENUATION_BINS - 1u);
        if component_bin != attenuation_bin { height = vec2(0.0); }
    } else {
        height = vec2(0.0);
    }
    let origin = cascade.center - vec2(0.5 * period)
        + vec2(0.5 * cascade.texel_width);
    let origin_phase = dot(k, origin);
    height = complex_multiply(height, vec2(cos(origin_phase), sin(origin_phase)));
    height *= fft.params.z;
    var horizontal_x = vec2(0.0);
    var horizontal_z = vec2(0.0);
    if k_length > 0.0 {
        let i_height = vec2(-height.y, height.x);
        let chop = CHOP * (1.0 - fft.params.w);
        horizontal_x = chop * k.x / k_length * i_height;
        horizontal_z = chop * k.y / k_length * i_height;
    }
    textureStore(height_x, vec2<i32>(id.xy), i32(id.z), vec4(height, horizontal_x));
    textureStore(z_field, vec2<i32>(id.xy), i32(id.z), vec4(horizontal_z, 0.0, 0.0));
}
