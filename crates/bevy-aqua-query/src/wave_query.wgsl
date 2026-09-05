// Samples the rendered displacement field at gameplay probe positions.
// LOD selection and cascade blending match the vertex path.

#import bevy_aqua_core::river::{river_analytic_displacement}

const LOD_COUNT: u32 = 5u;
const CASCADE_COUNT: u32 = LOD_COUNT + 1u;
const WAVE_COUNT: u32 = 40u;
const MAX_QUERIES: u32 = 256u;
const LOD_TRANSITION_START: f32 = 1.0;
const MORPH_BLACK_POINT: f32 = 0.05;
const MORPH_FADE_SIDES: f32 = 2.0;
const UV_CENTER: f32 = 0.5;
const MIN_SAMPLE_WEIGHT: f32 = 0.001;
const MIN_NORMAL_Y: f32 = 0.0001;

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

// `flow`: xy current (m/s), z signed bank margin (m), w channel half-width
// (m). Zero current selects cascade sampling.
struct QueryRequest {
    world_xz: vec2<f32>,
    slot: f32,
    kind: f32,
    flow: vec4<f32>,
}

struct QueryResult {
    displacement_slot: vec4<f32>,
    normal_validity: vec4<f32>,
    signals: vec4<f32>,
}

@group(0) @binding(0) var lod_data: texture_2d_array<f32>;
@group(0) @binding(1) var lod_sampler: sampler;
@group(0) @binding(2) var<uniform> params: AnimWavesUniform;
@group(0) @binding(3) var<storage, read> requests: array<QueryRequest>;
@group(0) @binding(4) var<storage, read_write> results: array<QueryResult>;

fn lod_count() -> u32 {
    return u32(params.cascade_layout.center.w);
}

fn lod_alpha(world_xz: vec2<f32>, cascade: CascadeParams) -> f32 {
    let offset = abs(world_xz - params.cascade_layout.center.xy);
    // Chebyshev distance matches the square LOD rings; Euclidean `length` is wrong here.
    let chebyshev_distance = max(offset.x, offset.y);
    let raw_alpha = chebyshev_distance / cascade.scale - LOD_TRANSITION_START;
    let black_point = MORPH_BLACK_POINT;
    let fade_width = 1.0 - MORPH_FADE_SIDES * black_point;
    return clamp((raw_alpha - black_point) / fade_width, 0.0, 1.0);
}

fn world_to_uv(world_xz: vec2<f32>, cascade: CascadeParams) -> vec2<f32> {
    let coverage = cascade.texel_width * cascade.texture_res;
    return (world_xz - cascade.center) / coverage + vec2(UV_CENTER);
}

fn sample_displacement(world_xz: vec2<f32>, lod: u32, alpha: f32) -> vec3<f32> {
    let smaller = params.cascade_layout.cascades[lod];
    let bigger = params.cascade_layout.cascades[lod + 1u];
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

fn select_lod(world_xz: vec2<f32>) -> u32 {
    // Use square rings; beyond the coarsest ring its sentinel fades to zero.
    for (var lod = 0u; lod < lod_count(); lod += 1u) {
        let cascade = params.cascade_layout.cascades[lod];
        let offset = abs(world_xz - cascade.center);
        if max(offset.x, offset.y) < cascade.scale {
            return lod;
        }
    }
    return lod_count() - 1u;
}

fn river_surface(world_xz: vec2<f32>, request: QueryRequest) -> vec3<f32> {
    return river_analytic_displacement(
        world_xz,
        request.flow,
        params.time.x,
    );
}

@compute @workgroup_size(64u)
fn sample(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    let count = arrayLength(&requests);
    if index >= count || index >= MAX_QUERIES {
        return;
    }
    let request = requests[index];

    if request.flow.w > 0.0 {
        let analytic = river_surface(request.world_xz, request);
        // Central differences of the same closed form for the normal.
        let epsilon = 0.35;
        let height_x = river_analytic_displacement(
            request.world_xz + vec2(epsilon, 0.0),
            request.flow,
            params.time.x,
        ).x;
        let height_z = river_analytic_displacement(
            request.world_xz + vec2(0.0, epsilon),
            request.flow,
            params.time.x,
        ).x;
        var normal_cross = vec3(
            -(height_x - analytic.x) / epsilon,
            1.0,
            -(height_z - analytic.x) / epsilon,
        );
        normal_cross.y = max(normal_cross.y, MIN_NORMAL_Y);
        results[index] = QueryResult(
            vec4(vec3(0.0, analytic.x, 0.0), request.slot),
            vec4(normalize(normal_cross), 1.0),
            vec4(0.0),
        );
        return;
    }

    if request.kind > 0.5 {
        results[index] = QueryResult(
            vec4(0.0, 0.0, 0.0, request.slot),
            vec4(0.0, 1.0, 0.0, 1.0),
            vec4(0.0),
        );
        return;
    }

    // Mirror cascade.wgsl's advection so probes ride the same moving water
    // the camera sees; the shift is applied once before any wave sampling.
    let world_xz = request.world_xz - params.flow.xy * params.time.x;
    let detail_lod = clamp(params.cascade_layout.center.z, 0.0, f32(lod_count() - 1u));
    let minimum_lod = u32(floor(detail_lod));
    let detail_alpha = fract(detail_lod);
    var lod = select_lod(request.world_xz);
    var alpha = lod_alpha(request.world_xz, params.cascade_layout.cascades[lod]);
    if lod < minimum_lod {
        lod = minimum_lod;
        alpha = detail_alpha;
    } else if lod == minimum_lod {
        alpha = max(alpha, detail_alpha);
    }

    let displacement = sample_displacement(world_xz, lod, alpha);
    let texel_width = params.cascade_layout.cascades[lod].texel_width;
    let displacement_x = sample_displacement(
        world_xz + vec2(texel_width, 0.0),
        lod,
        alpha,
    );
    let displacement_z = sample_displacement(
        world_xz + vec2(0.0, texel_width),
        lod,
        alpha,
    );
    let tangent_x = vec3(texel_width, 0.0, 0.0) + displacement_x - displacement;
    let tangent_z = vec3(0.0, 0.0, texel_width) + displacement_z - displacement;
    var normal_cross = cross(tangent_z, tangent_x);
    normal_cross.y = max(normal_cross.y, MIN_NORMAL_Y);
    let normal = normalize(normal_cross);
    let horizontal_x = vec2(texel_width, 0.0)
        + displacement_x.xz - displacement.xz;
    let horizontal_z = vec2(0.0, texel_width)
        + displacement_z.xz - displacement.xz;
    let determinant = (horizontal_x.x * horizontal_z.y
        - horizontal_x.y * horizontal_z.x) / (texel_width * texel_width);
    let crest = clamp(0.55 - determinant, 0.0, 1.0);

    results[index] = QueryResult(
        vec4(displacement, request.slot),
        vec4(normal, 1.0),
        vec4(crest, 0.0, 0.0, 0.0),
    );
}
