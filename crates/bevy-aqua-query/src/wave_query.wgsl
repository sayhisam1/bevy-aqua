// Samples the rendered displacement field at gameplay probe positions.
// LOD selection and cascade blending match the vertex path.

#import bevy_aqua_core::river::{river_analytic_displacement}
#import bevy_aqua_core::waves_sample::{
    AnimWavesUniform,
    resolve_lod,
    sample_displacement,
}

const MAX_QUERIES: u32 = 256u;
const MIN_NORMAL_Y: f32 = 0.0001;

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

@compute @workgroup_size(64u)
fn sample(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    let count = arrayLength(&requests);
    if index >= count || index >= MAX_QUERIES {
        return;
    }
    let request = requests[index];

    if request.flow.w > 0.0 {
        let analytic = river_analytic_displacement(
            request.world_xz,
            request.flow,
            params.time.x,
        );
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
    let lod = resolve_lod(params.cascade_layout, world_xz);
    let displacement = sample_displacement(
        lod_data,
        lod_sampler,
        params.cascade_layout,
        world_xz,
        lod.lod,
        lod.alpha,
    );
    let texel_width = params.cascade_layout.cascades[lod.lod].texel_width;
    let displacement_x = sample_displacement(
        lod_data,
        lod_sampler,
        params.cascade_layout,
        world_xz + vec2(texel_width, 0.0),
        lod.lod,
        lod.alpha,
    );
    let displacement_z = sample_displacement(
        lod_data,
        lod_sampler,
        params.cascade_layout,
        world_xz + vec2(0.0, texel_width),
        lod.lod,
        lod.alpha,
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
