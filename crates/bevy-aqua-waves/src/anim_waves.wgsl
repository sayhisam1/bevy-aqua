// Reimplementation of the approach in Crest Scripts/Shapes/ShapeGerstnerBatched.cs and
// Shaders/OceanInputs/GerstnerShared.hlsl.

const LOD_COUNT: u32 = 5u;
const CASCADE_COUNT: u32 = LOD_COUNT + 1u;
const WAVE_COUNT: u32 = 40u;
const TAU: f32 = 6.283185307179586;
const PI: f32 = 3.141592653589793;

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
    flow: vec4<f32>,
}

@group(0) @binding(0) var output: texture_storage_2d_array<rgba16float, write>;
@group(0) @binding(1) var<uniform> params: AnimWavesUniform;
@group(0) @binding(2) var bed_height: texture_2d<f32>;

// Decoded "no bed data" depth: matches a cleared full-depth capture texel.
const NO_BED_DEPTH: f32 = 256.0;

/// Water depth under the sea level from the game's bed height map. Outside
/// the mapped area (or with no map at all) the sample keeps the deep default,
/// mirroring a full-depth capture texel. Nearest-texel read: the shoaling
/// profile is smooth enough that one 5 m-class bed texel suffices.
fn bed_water_depth(world_xz: vec2<f32>) -> f32 {
    let range = params.cascade_layout.bed_range;
    if range.y < 0.0 {
        return NO_BED_DEPTH;
    }
    let uv = (world_xz - params.cascade_layout.bed_transform.xy)
        * params.cascade_layout.bed_transform.zw;
    if any(uv < vec2(0.0)) || any(uv > vec2(1.0)) {
        return NO_BED_DEPTH;
    }
    let dimensions = vec2<u32>(textureDimensions(bed_height));
    let maximum = vec2<i32>(dimensions - vec2<u32>(1u));
    let texel = clamp(vec2<i32>(round(uv * vec2<f32>(maximum))), vec2(i32(0)), maximum);
    let height = f32(textureLoad(bed_height, texel, 0).r) * range.y + range.x;
    return max(range.z - height, 0.0);
}

// A bounded finite-depth shoaling profile. The vertical lane retains more
// energy than horizontal chop through the breaker zone, producing steeper
// near-shore crests without changing deep-water phase or world anchoring.
fn shoaling_weights(depth: f32, wave_number: f32) -> vec2<f32> {
    let relative_depth = clamp(depth * wave_number / PI, 0.0, 1.0);
    let vertical_base = smoothstep(0.0, 1.0, relative_depth);
    let breaker = 4.0 * vertical_base * (1.0 - vertical_base);
    let vertical = vertical_base * (1.0 + 0.18 * breaker);
    let chop_ratio = mix(0.55, 1.0, smoothstep(0.15, 0.85, vertical_base));
    return vec2(vertical * chop_ratio, vertical);
}

@compute @workgroup_size(8, 8, 1)
fn generate(@builtin(global_invocation_id) id: vec3<u32>) {
    let cascade = params.cascade_layout.cascades[id.z];
    if any(id.xy >= vec2<u32>(u32(cascade.texture_res))) || id.z >= LOD_COUNT {
        return;
    }
    let uv = (vec2<f32>(id.xy) + vec2(0.5)) * cascade.inv_texture_res;
    let coverage = cascade.texel_width * cascade.texture_res;
    let world_xz = coverage * (uv - vec2(0.5)) + cascade.center;
    let range = params.ranges[id.z];
    let water_depth = bed_water_depth(world_xz);
    var displacement = vec3(0.0);

    for (var index = range.x; index < range.y; index += 1u) {
        let wave = params.waves[index];
        let temporal_phase = wave.phase + wave.angular_frequency * params.time.x;
        let wrapped_phase = temporal_phase - floor(temporal_phase / TAU) * TAU;
        let angle = wave.wave_number * dot(wave.direction, world_xz) + wrapped_phase;
        // Extend Crest's per-wave attenuation with a smooth breaker profile.
        // Apply it before coarse-to-fine combination so every owned wavelength
        // responds to its own finite-depth ratio.
        let shoaling = shoaling_weights(water_depth, wave.wave_number);
        let attenuation = params.time.z * mix(vec2(1.0), shoaling, params.time.y);
        let horizontal = attenuation.x * wave.chop_amplitude * sin(angle);
        displacement += vec3(
            horizontal * wave.direction.x,
            attenuation.y * wave.amplitude * cos(angle),
            horizontal * wave.direction.y,
        );
    }
    textureStore(output, vec2<i32>(id.xy), i32(id.z), vec4(displacement, 0.0));
}
