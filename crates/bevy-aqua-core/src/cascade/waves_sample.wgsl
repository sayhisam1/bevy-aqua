// Cascade layout ABI and LOD displacement blend. Wave producers, foam
// simulation, and GPU query import this module.

#define_import_path bevy_aqua_core::waves_sample

const LOD_COUNT: u32 = 5u;
const CASCADE_COUNT: u32 = LOD_COUNT + 1u;
const WAVE_COUNT: u32 = 40u;
const LOD_TRANSITION_START: f32 = 1.0;
// Crest 0.4 morph fade at eight vertices per 64-vertex tile.
const MORPH_BLACK_POINT: f32 = 0.05;
const MORPH_FADE_SIDES: f32 = 2.0;
const UV_CENTER: f32 = 0.5;
const MIN_SAMPLE_WEIGHT: f32 = 0.001;

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

struct LodSample {
    lod: u32,
    alpha: f32,
}

fn lod_count(cascade_layout: CascadeLayout) -> u32 {
    return u32(cascade_layout.center.w);
}

fn lod_alpha(world_xz: vec2<f32>, cascade: CascadeParams, cascade_layout: CascadeLayout) -> f32 {
    let offset = abs(world_xz - cascade_layout.center.xy);
    // Chebyshev distance matches the square LOD rings; Euclidean `length` is wrong here.
    let chebyshev_distance = max(offset.x, offset.y);
    let raw_alpha = chebyshev_distance / cascade.scale - LOD_TRANSITION_START;
    let fade_width = 1.0 - MORPH_FADE_SIDES * MORPH_BLACK_POINT;
    return clamp((raw_alpha - MORPH_BLACK_POINT) / fade_width, 0.0, 1.0);
}

fn world_to_uv(world_xz: vec2<f32>, cascade: CascadeParams) -> vec2<f32> {
    let coverage = cascade.texel_width * cascade.texture_res;
    return (world_xz - cascade.center) / coverage + vec2(UV_CENTER);
}

fn sample_displacement(
    lod_data: texture_2d_array<f32>,
    lod_sampler: sampler,
    cascade_layout: CascadeLayout,
    world_xz: vec2<f32>,
    lod: u32,
    alpha: f32,
) -> vec3<f32> {
    let smaller = cascade_layout.cascades[lod];
    let bigger = cascade_layout.cascades[lod + 1u];
    let smaller_weight = (1.0 - alpha) * smaller.weight;
    let bigger_weight = (1.0 - smaller_weight) * bigger.weight;
    var displacement = vec3(0.0);

    if smaller_weight > MIN_SAMPLE_WEIGHT {
        let uv = world_to_uv(world_xz, smaller);
        displacement += smaller_weight
            * textureSampleLevel(lod_data, lod_sampler, uv, i32(lod), 0.0).xyz;
    }
    if bigger_weight > MIN_SAMPLE_WEIGHT {
        let uv = world_to_uv(world_xz, bigger);
        displacement += bigger_weight
            * textureSampleLevel(lod_data, lod_sampler, uv, i32(lod + 1u), 0.0).xyz;
    }
    return displacement;
}

fn select_lod(cascade_layout: CascadeLayout, world_xz: vec2<f32>) -> u32 {
    // Use square rings; beyond the coarsest ring its sentinel fades to zero.
    let count = lod_count(cascade_layout);
    for (var lod = 0u; lod < count; lod += 1u) {
        let cascade = cascade_layout.cascades[lod];
        let offset = abs(world_xz - cascade.center);
        if max(offset.x, offset.y) < cascade.scale {
            return lod;
        }
    }
    return count - 1u;
}

fn resolve_lod(cascade_layout: CascadeLayout, world_xz: vec2<f32>) -> LodSample {
    let count = lod_count(cascade_layout);
    let detail_lod = clamp(cascade_layout.center.z, 0.0, f32(count - 1u));
    let minimum_lod = u32(floor(detail_lod));
    let detail_alpha = fract(detail_lod);
    var lod = select_lod(cascade_layout, world_xz);
    var alpha = lod_alpha(world_xz, cascade_layout.cascades[lod], cascade_layout);
    if lod < minimum_lod {
        lod = minimum_lod;
        alpha = detail_alpha;
    } else if lod == minimum_lod {
        alpha = max(alpha, detail_alpha);
    }
    return LodSample(lod, alpha);
}
