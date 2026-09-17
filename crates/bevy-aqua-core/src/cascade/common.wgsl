// Aqua cascade sampling boundary: ABI structs, material bindings, shared
// constants, and the field/level/flow/depth sampling entry points. Feature
// modules (#import-ed by the composed material) may only reach shared state
// through this file; per-invocation private state is read/written here and
// handed to feature modules as plain values or small snapshots.
//
// Reimplementation of the approach in Crest OceanVertHelpers.hlsl, OceanHelpersNew.hlsl, and Ocean.shader.
// AnimWaves sampling follows Scripts/LodData/LodDataMgrAnimWaves.cs.

#define_import_path aqua::cascade

#import bevy_pbr::{
    forward_io::Vertex,
    mesh_functions,
    mesh_view_types,
    clustered_forward as clustering,
    lighting,
    prepass_utils,
    shadows,
    mesh_view_bindings::{light_probes, lights, view},
    view_transformations::position_world_to_clip,
}
#import bevy_pbr::mesh_view_bindings as view_bindings
#import bevy_aqua_core::waves_sample::{
    CascadeLayout,
    CascadeParams,
    lod_alpha,
}

const VERTEX_SNAP_MULTIPLIER: f32 = 2.0;
const COARSE_GRID_MULTIPLIER: f32 = 4.0;
const MORPH_INNER_RADIUS: f32 = 0.375;

const GRID_CELL_CENTER: f32 = 0.5;
const MIN_SAMPLE_WEIGHT: f32 = 0.001;
const MIN_NORMAL_Y: f32 = 0.0001;
const SAFE_LENGTH_SQUARED: f32 = 1e-8;
const LUMINANCE_EPSILON: f32 = 0.0001;

const DEBUG_MODE_WATER_PATH: u32 = 1u;
const DEBUG_MODE_REFRACTION_VALIDITY: u32 = 2u;
const DEBUG_MODE_TRANSMISSION: u32 = 3u;
const DEBUG_MODE_UNREFRACTED: u32 = 4u;
const DEBUG_MODE_BEER_LAMBERT: u32 = 5u;
const DEBUG_MODE_SEA_FLOOR: u32 = 6u;
const DEBUG_MODE_BEAUTY: u32 = 7u;
const DEBUG_MODE_REFLECTION: u32 = 8u;
const DEBUG_MODE_FOAM: u32 = 9u;
const DEBUG_MODE_WAVE_HEIGHT: u32 = 10u;
const DEBUG_MODE_LIGHT_RADIANCE: u32 = 11u;
const DEBUG_MODE_REFLECTION_FRACTION: u32 = 12u;
const DEBUG_MODE_FAR_TIER: u32 = 13u;

const CREST_SSS_MAXIMUM: f32 = 0.6;
const CREST_SSS_RANGE: f32 = 0.12;
const CREST_SSS_UNCOMPRESSED: f32 = CREST_SSS_MAXIMUM - CREST_SSS_RANGE;

struct PlanarReflectionView {
    view_projection: mat4x4<f32>,
    level: f32,
}

struct PlanarReflectionParams {
    views: array<PlanarReflectionView, 2>,
    view_count: u32,
    distortion: f32,
}

struct PlanarReflectionSample {
    color: vec3<f32>,
    weight: f32,
}

// Allows displaced surface positions to feather just beyond the projected target.
const PLANAR_PROJECTION_GUARD: f32 = 0.03;

struct SurfaceParams {
    fresnel: vec4<f32>,
    reflection: vec4<f32>,
    sun: vec4<f32>,
    debug: vec4<f32>,
    fog_density: vec4<f32>,
    scatter_tint: vec4<f32>,
    sea_floor: vec4<f32>,
    sss_tint: vec4<f32>,
    sss: vec4<f32>,
    detail: vec4<f32>,
    capillary: vec4<f32>,
    foam: vec4<f32>,
    advection: vec4<f32>,
    /// x/y: configurable far-tier start/end distances in metres.
    /// z reserved; w: Henyey-Greenstein `g`.
    far_tier: vec4<f32>,
    /// Strength, metres per cell, metres per second, and maximum depth in metres.
    caustics: vec4<f32>,
}

/// Localized-water extent controls; mirrors lod::BodyParams. flags.x is 1.0
/// for bounded bodies: the vertex stage skips camera snap/morph and the
/// fragment stage culls against extent.xy (centre) and extent.w (radius).
struct BodyParams {
    flags: vec4<f32>,
    extent: vec4<f32>,
    aabb_min: vec4<f32>,
    aabb_size: vec4<f32>,
    /// rgb: per-channel Beer-Lambert extinction in 1/m; w: optics enable.
    optics_a: vec4<f32>,
    /// x: scatter-scale for particle σs; y: sun roughness; z: plain Schlick
    /// flag; w: Henyey-Greenstein `g`.
    optics_b: vec4<f32>,
    /// rgb: particle scatter chromaticity; w reserved.
    optics_c: vec4<f32>,
}

@group(#{MATERIAL_BIND_GROUP}) @binding(0) var lod_data: texture_2d_array<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(1) var lod_sampler: sampler;
@group(#{MATERIAL_BIND_GROUP}) @binding(2) var<uniform> cascade_layout: CascadeLayout;
@group(#{MATERIAL_BIND_GROUP}) @binding(3) var<uniform> surface: SurfaceParams;
@group(#{MATERIAL_BIND_GROUP}) @binding(4) var bed_height: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(5) var detail_normal: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(6) var detail_sampler: sampler;
@group(#{MATERIAL_BIND_GROUP}) @binding(7) var foam_data: texture_2d_array<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(8) var foam_sampler: sampler;
@group(#{MATERIAL_BIND_GROUP}) @binding(9) var foam_pattern: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(10) var foam_pattern_sampler: sampler;
/// Global water fields: region mapping, per-slot body parameters, and the
/// baked level/slot + flow textures. Mirrors fields::FieldParams.
const MAX_BODIES: u32 = 16u;

struct FieldParams {
    /// xy: region minimum in metres; zw: region size in metres.
    region: vec4<f32>,
    /// x: bounded body count; y: 1.0 when the Ocean resource is present;
    /// z: metres per texel; w: reserved.
    info: vec4<f32>,
    bodies: array<BodyParams, MAX_BODIES>,
}

@group(#{MATERIAL_BIND_GROUP}) @binding(15) var<uniform> field_params: FieldParams;
@group(#{MATERIAL_BIND_GROUP}) @binding(16) var field_maps: texture_2d_array<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(17) var field_sampler: sampler;
@group(#{MATERIAL_BIND_GROUP}) @binding(19) var reflection_a: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(20) var reflection_sampler: sampler;
@group(#{MATERIAL_BIND_GROUP}) @binding(21) var reflection_b: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(22) var<uniform> planar_reflections: PlanarReflectionParams;

fn sample_planar_reflection(
    world_position: vec3<f32>,
    surface_level: f32,
    surface_normal: vec3<f32>,
) -> PlanarReflectionSample {
    if planar_reflections.view_count == 0u {
        return PlanarReflectionSample(vec3(0.0), 0.0);
    }
    var index = 0u;
    if planar_reflections.view_count > 1u
        && abs(surface_level - planar_reflections.views[1].level)
            < abs(surface_level - planar_reflections.views[0].level) {
        index = 1u;
    }
    let view = planar_reflections.views[index];
    let clip = view.view_projection * vec4(world_position, 1.0);
    if clip.w <= 0.0 {
        return PlanarReflectionSample(vec3(0.0), 0.0);
    }
    let ndc = clip.xy / clip.w;
    let projected_uv = vec2(ndc.x, -ndc.y) * 0.5 + 0.5;
    let outside = max(
        max(-projected_uv.x, projected_uv.x - 1.0),
        max(-projected_uv.y, projected_uv.y - 1.0),
    );
    if outside >= PLANAR_PROJECTION_GUARD {
        return PlanarReflectionSample(vec3(0.0), 0.0);
    }
    let dimensions = vec2<f32>(textureDimensions(reflection_a));
    let half_texel = 0.5 / dimensions;
    let projected_edge = min(
        min(projected_uv.x, projected_uv.y),
        min(1.0 - projected_uv.x, 1.0 - projected_uv.y),
    );
    // Preserve full reflection coverage: withdraw distortion near the edge instead
    // of moving an otherwise valid projected sample outside the render target.
    let distortion_guard = smoothstep(
        0.0,
        planar_reflections.distortion + max(half_texel.x, half_texel.y),
        projected_edge,
    );
    var uv = projected_uv
        + vec2(surface_normal.x, -surface_normal.z)
            * planar_reflections.distortion * distortion_guard;
    uv = clamp(uv, half_texel, vec2(1.0) - half_texel);
    var sample = textureSampleLevel(
        reflection_a,
        reflection_sampler,
        uv,
        0.0,
    );
    if index == 1u {
        sample = textureSampleLevel(
            reflection_b,
            reflection_sampler,
            uv,
            0.0,
        );
    }
    // Deferred HDR alpha is not a validity signal. In-bounds projected pixels
    // remain fully planar; only displaced projections beyond the target feather.
    let weight = 1.0 - smoothstep(0.0, PLANAR_PROJECTION_GUARD, max(outside, 0.0));
    return PlanarReflectionSample(sample.rgb, weight);
}

// Effective current for wave advection at the current invocation: the
// global uniform by default, the river's local flow inside bounded bodies
// that bake one. Vertex and fragment stages set it before sampling.
var<private> effective_flow: vec2<f32> = vec2(0.0, 0.0);
var<private> effective_time: f32 = 0.0;
// Conservative world-XZ metres covered by one screen pixel; nonnegative.
var<private> xz_footprint: f32 = 0.0;

fn set_xz_footprint(value: f32) {
    xz_footprint = value;
}

fn screen_xz_footprint() -> f32 {
    return xz_footprint;
}

// Analytic texture LOD: dUV/dpixel is the world-XZ footprint divided by
// metres per UV repeat, and mip LOD is log2(texels covered per pixel).
fn screen_texture_lod(metres_per_repeat: f32, texture_width: u32) -> f32 {
    let texels_per_pixel =
        screen_xz_footprint() * f32(texture_width) / max(metres_per_repeat, 1e-6);
    return max(log2(max(texels_per_pixel, 1.0)), 0.0);
}

/// Ripple-strength multiplier at the current fragment: 1.0 everywhere except
/// inside river bodies, where faster narrows read rougher and banks calm.
var<private> river_ripple_scale: f32 = 1.0;

/// Per-invocation body state, set at stage entry from the owning slot:
/// bounded flag, river flag, optics_a, optics_b. Defaults are inert.
var<private> invocation_bounded: f32 = 0.0;
var<private> invocation_river: f32 = 0.0;
var<private> invocation_optics_a: vec4<f32> = vec4(0.0);
var<private> invocation_optics_b: vec4<f32> = vec4(0.0);

/// Per-body water optics: extinction replaces the ocean Beer-Lambert
/// coefficients, scatter_scale is particle load, and scatter_tint is haze
/// chromaticity for the shared medium.
var<private> body_extinction: vec3<f32> = vec3(0.0);
var<private> body_scatter_scale: f32 = 1.0;
var<private> body_scatter_tint: vec3<f32> = vec3(0.85, 1.0, 1.22);
var<private> body_scattering_asymmetry: f32 = 0.8;

/// Baked flow sample at the current fragment (xy: current m/s, z: signed
/// bank margin, w: channel half-width in metres).
var<private> fragment_river: vec4<f32> = vec4(0.0);


struct LocalLightSample {
    direction: vec3<f32>,
    radiance: vec3<f32>,
}

/// One snapshot of this invocation's river state, handed to imported
/// modules that cannot read this file's private globals directly.
struct RiverState {
    /// True when the fragment sits inside a bounded river body.
    enabled: bool,
    /// The baked flow sample: xy current m/s, z signed bank margin,
    /// w channel half-width in metres.
    sample: vec4<f32>,
    /// Effective advection current for wave-content sampling.
    flow: vec2<f32>,
}

/// Stage entry: records the owning body slot's parameters.
fn begin_invocation(bounded: bool, params: BodyParams) {
    invocation_bounded = select(0.0, 1.0, bounded);
    invocation_river = params.flags.y;
    invocation_optics_a = params.optics_a;
    invocation_optics_b = params.optics_b;
}

/// Stage entry: selects the advection current for wave-content sampling.
fn set_effective_flow(flow: vec2<f32>) {
    effective_flow = flow;
}

fn set_effective_time(time: f32) {
    effective_time = time;
}

/// Fragment entry: records the baked river-flow sample under the fragment.
fn set_fragment_river(sample: vec4<f32>) {
    fragment_river = sample;
}

/// Fragment entry: records the effective Beer-Lambert extinction, particle
/// scatter scale and tint, and Henyey-Greenstein `g` after fresh-water optics
/// override.
fn set_body_optics(
    extinction: vec3<f32>,
    scatter_scale: f32,
    scatter_tint: vec3<f32>,
    scattering_asymmetry: f32,
) {
    body_extinction = extinction;
    body_scatter_scale = scatter_scale;
    body_scatter_tint = scatter_tint;
    body_scattering_asymmetry = scattering_asymmetry;
}

/// Fragment entry: records the river ripple-strength multiplier.
fn set_river_ripple(scale: f32) {
    river_ripple_scale = scale;
}

fn invocation_extinction() -> vec3<f32> {
    return body_extinction;
}

fn invocation_scatter_scale() -> f32 {
    return body_scatter_scale;
}

fn invocation_scatter_tint() -> vec3<f32> {
    return body_scatter_tint;
}

fn invocation_scattering_asymmetry() -> f32 {
    return body_scattering_asymmetry;
}

fn invocation_river_state() -> RiverState {
    return RiverState(
        invocation_bounded > 0.5 && invocation_river > 0.5,
        fragment_river,
        effective_flow,
    );
}

fn invocation_ripple() -> f32 {
    return river_ripple_scale;
}


// Wave content advects at `surface.advection.xy` metres per second. A
// sampling-space shift is exact Doppler advection for both spectra:
// sampling at `x - u * t` turns every component's phase into
// `(k . x) - (omega + k . u) * t`. Snap/transition, depth lookups, and foam
// gating stay world-anchored.
fn advected_world(world_xz: vec2<f32>) -> vec2<f32> {
    return world_xz - effective_flow * effective_time;
}

/// Samples the baked river field; returns zero flow and a huge bank margin
/// when this material has no flow texture.
fn field_uv(world_xz: vec2<f32>) -> vec2<f32> {
    return (world_xz - field_params.region.xy)
        / max(field_params.region.zw, vec2(1e-4));
}

/// rg: surface level, one-based body slot (0 = unclaimed).
fn sample_field_level(world_xz: vec2<f32>) -> vec2<f32> {
    return textureSampleLevel(field_maps, field_sampler, field_uv(world_xz), 0, 0.0).xy;
}

/// rgb: flow m/s; z: signed bank margin in metres; w: speed m/s.
fn sample_field_flow(world_xz: vec2<f32>) -> vec4<f32> {
    return textureSampleLevel(field_maps, field_sampler, field_uv(world_xz), 1, 0.0);
}

/// Parameters of the body owning a point; slot 0 falls back to the
/// inactive ocean defaults so unguarded reads stay harmless.
fn owning_body(slot: u32) -> BodyParams {
    if slot > 0u {
        return field_params.bodies[slot - 1u];
    }
    return field_params.bodies[0];
}

fn lod_count() -> u32 {
    return u32(cascade_layout.center.w);
}

fn smootherstep01(value: f32) -> f32 {
    let x = clamp(value, 0.0, 1.0);
    return x * x * x * (x * (x * 6.0 - 15.0) + 10.0);
}

fn far_tier_weight(base_world_position: vec3<f32>) -> f32 {
    let distance = length(base_world_position - view.world_position.xyz);
    return smootherstep01(
        (distance - surface.far_tier.x)
            / max(surface.far_tier.y - surface.far_tier.x, 1.0),
    );
}

fn snap_and_transition(
    world_xz: vec2<f32>,
    object_xz: vec2<f32>,
    cascade: CascadeParams,
) -> vec3<f32> {
    let grid_width = cascade.texel_width;
    let snap_width = VERTEX_SNAP_MULTIPLIER * grid_width;
    var transitioned = world_xz - fract(object_xz / snap_width) * snap_width;
    let alpha = lod_alpha(transitioned, cascade, cascade_layout);

    let coarse_grid = COARSE_GRID_MULTIPLIER * grid_width;
    let offset = fract(transitioned / coarse_grid) - vec2(GRID_CELL_CENTER);
    if abs(offset.x) < MORPH_INNER_RADIUS {
        transitioned.x += offset.x * alpha * coarse_grid;
    }
    if abs(offset.y) < MORPH_INNER_RADIUS {
        transitioned.y += offset.y * alpha * coarse_grid;
    }
    return vec3(transitioned, alpha);
}

fn flow_frame(world_xz: vec2<f32>) -> vec2<f32> {
    let speed = length(effective_flow);
    if !(invocation_bounded > 0.5 && invocation_river > 0.5) || speed < 0.05 {
        return world_xz;
    }
    let dir = effective_flow / speed;
    let along = dot(dir, world_xz) / (1.0 + 0.55 * min(speed, 4.5));
    let across = dot(vec2(-dir.y, dir.x), world_xz);
    return vec2(along, across * 1.35);
}

fn capillary_resolved_weight(world_xz: vec2<f32>) -> f32 {
    let distance_to_view = length(world_xz - view.world_position.xz);
    return 1.0 - smoothstep(
        surface.capillary.z,
        surface.capillary.w,
        distance_to_view,
    );
}

// GodotOceanWaves `water.gdshader`: bounded GGX distribution and its Smith
// masking-shadowing approximation. Aqua applies Fresnel later in Crest's
// reflection composition, so this returns the remaining direct-sun factor.
fn godot_fresnel(view_alignment: f32) -> f32 {
    // Cubemap-only oceans preserve the accepted roughness-damped Godot curve.
    // Planar mode uses physical dielectric Schlick: roughness broadens the
    // reflected lobe but must not cap grazing-angle energy, or a bright sky
    // leaves distant water saturated navy. Calm authored bodies use the same
    // plain response in either reflection mode.
    let body_active = invocation_bounded > 0.5 && invocation_optics_a.w > 0.5;
    let sun_roughness = select(
        surface.sun.y,
        invocation_optics_b.y,
        body_active && invocation_optics_b.y >= 0.0,
    );
    let plain_schlick = planar_reflections.view_count > 0u
        || (body_active && invocation_optics_b.z > 0.5);
    let exponent = select(
        surface.fresnel.y * exp(-2.69 * sun_roughness),
        surface.fresnel.y,
        plain_schlick,
    );
    let damping = select(
        1.0 + 22.7 * pow(sun_roughness, 1.5),
        1.0,
        plain_schlick,
    );
    let rough = pow(max(0.0, 1.0 - view_alignment), exponent) / damping;
    return mix(rough, 1.0, surface.fresnel.x);
}
