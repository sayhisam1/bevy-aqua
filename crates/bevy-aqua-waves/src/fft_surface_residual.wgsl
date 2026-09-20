// FD prediction for the bounded shallow residual. Read the FD prefix before
// analytic overwrite; write a disjoint four-layer view (physical21..24).
// XY: one native-grid band's slope increment over its coarser context.
// ZW: midpoint curvature for footprint retention of that band. The increments
// telescope at full retention; a quadratic matches retention0,0.5,1.
// Actual legacy normals remain filtered in the fragment. Caching their finished
// residual instead would blur coarse terms again on each finer cache's grid.
const FFT_RESOLUTION: u32 = 256u;
const LOD_COUNT: u32 = 5u;
const ATTENUATION_BINS: u32 = 4u;

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
    cascades: array<CascadeParams, 6>,
    center: vec4<f32>,
    // XY bed-map first-texel world origin, ZW inverse world extent.
    bed_transform: vec4<f32>,
    // X height minimum, Y height span (negative = no bed map), Z sea level.
    bed_range: vec4<f32>,
}
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


@group(0) @binding(0) var source: texture_2d_array<f32>;
@group(0) @binding(1) var linear_sampler: sampler;
@group(0) @binding(2) var residuals: texture_storage_2d_array<rgba16float, write>;
@group(0) @binding(3) var<uniform> fft: FftUniform;
@group(0) @binding(4) var bed_height: texture_2d<f32>;

fn uv_at(world: vec2<f32>, c: CascadeParams) -> vec2<f32> {
    return (world - c.center) / (c.texel_width * c.texture_res) + vec2(0.5);
}
fn depth_weights(depth: f32, wavelength: f32) -> vec2<f32> {
    let base = smoothstep(0.0, 1.0, clamp(2.0 * depth / wavelength, 0.0, 1.0));
    let breaker = 4.0 * base * (1.0 - base);
    let vertical = base * (1.0 + 0.18 * breaker);
    let chop = mix(0.55, 1.0, smoothstep(0.15, 0.85, base));
    return mix(vec2(1.0), vec2(vertical * chop, vertical), fft.params.y);
}
fn slope(dx: vec3<f32>, dz: vec3<f32>) -> vec2<f32> {
    let n = cross(vec3(0.0, 0.0, 1.0) + dz, vec3(1.0, 0.0, 0.0) + dx);
    return n.xz / max(n.y, 0.0001);
}
@compute @workgroup_size(8, 8, 1)
fn resolve_residual(@builtin(global_invocation_id) id: vec3<u32>) {
    if any(id.xy >= vec2<u32>(FFT_RESOLUTION)) || id.z >= 4u { return; }
    let xy = vec2<i32>(id.xy);
    let c = fft.cascade_layout.cascades[id.z];
    let uv = (vec2<f32>(id.xy) + vec2(0.5)) / c.texture_res;
    let world = c.center + (uv - vec2(0.5)) * c.texel_width * c.texture_res;
    let depth = bed_water_depth(world);
    // A global all-deep bound is safe. A local depth cutoff is not: pixels
    // across a bed step can need this prediction through interpolation.
    let range = fft.cascade_layout.bed_range;
    if range.z - (range.x + range.y) >= 6.0 {
        textureStore(residuals, xy, i32(id.z), vec4(0.0));
        return;
    }
    var dx = vec3(0.0);
    var dz = vec3(0.0);
    var fine_dx = vec3(0.0);
    var fine_dz = vec3(0.0);
    for (var field = id.z; field < 8u; field++) {
        let band = fft.cascade_layout.cascades[min(field, 4u)];
        let period = band.texel_width * band.texture_res;
        var minimum = 0.5 * band.max_wavelength;
        var maximum = select(band.max_wavelength, period / 4.0, field >= 4u);
        if field >= 4u {
            let octaves = log2(maximum / minimum);
            maximum = minimum * exp2(octaves * f32(field - 3u) / 4.0);
            minimum *= exp2(octaves * f32(field - 4u) / 4.0);
        }
        let at = depth_weights(depth, sqrt(minimum * maximum));
        let weights = vec3(at.x, at.y, at.x);
        let band_uv = fract(uv_at(world, band));
        let layer = i32(LOD_COUNT + 2u * field);
        let field_dx = weights * textureSampleLevel(source, linear_sampler, band_uv, layer, 0.0).xyz;
        let field_dz = weights * textureSampleLevel(source, linear_sampler, band_uv, layer + 1, 0.0).xyz;
        dx += field_dx;
        dz += field_dz;
        if field == id.z { fine_dx = field_dx; fine_dz = field_dz; }
    }
    let coarse = slope(dx - fine_dx, dz - fine_dz);
    let increment = slope(dx, dz) - coarse;
    let curvature = slope(dx - 0.5 * fine_dx, dz - 0.5 * fine_dz)
        - coarse - 0.5 * increment;
    // Nearly folded Jacobians can produce large slope ratios. Keep the
    // approximate prediction finite in RGBA16F; displacement and analytic
    // derivative/moment fields are not clipped by this storage safety bound.
    let prediction = clamp(vec4(increment, curvature), vec4(-60000.0), vec4(60000.0));
    textureStore(residuals, xy, i32(id.z), prediction);
}
