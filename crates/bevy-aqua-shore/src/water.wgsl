// Bed-depth and river-body contributions to the composed cascade material.

#define_import_path aqua::shore::water

#import aqua::cascade::{bed_height, cascade_layout, foam_pattern, foam_pattern_sampler, surface}

// Decoded "no bed data" depth: matches a cleared full-depth capture texel.

const NO_BED_DEPTH: f32 = 256.0;
// Must match `PATTERN_CAUSTIC_CELLS` packed into foam_pattern green.
const CAUSTIC_CELLS_PER_TILE: f32 = 16.0;
// Restores bright focused ridges after multiplying two sparse samples.
const CAUSTIC_FOCUS_GAIN: f32 = 12.0;
// Suppress moonlight-scale directional sources while smoothly admitting daylight.
const CAUSTIC_DAYLIGHT_MIN_LUX: f32 = 64.0;
const CAUSTIC_REFERENCE_SUN_LUX: f32 = 120000.0;
const CAUSTIC_LUMINANCE_WEIGHTS: vec3<f32> = vec3(0.2126, 0.7152, 0.0722);

// River-wave synthesis lives in bevy_aqua_core::river (shared with the
// wave-query pass).


/// Water depth under the sea level from the game's bed height map. Outside
/// the mapped area (or with no map at all) the sample keeps the deep default,
/// mirroring a full-depth capture texel.
fn bed_water_depth(world_xz: vec2<f32>) -> f32 {
    let range = cascade_layout.bed_range;
    if range.y < 0.0 {
        return NO_BED_DEPTH;
    }
    let uv = (world_xz - cascade_layout.bed_transform.xy)
        * cascade_layout.bed_transform.zw;
    if any(uv < vec2(0.0)) || any(uv > vec2(1.0)) {
        return NO_BED_DEPTH;
    }
    let dimensions = vec2<f32>(textureDimensions(bed_height));
    let texel_position = uv * (dimensions - vec2(1.0));
    let base = vec2<i32>(floor(texel_position));
    let fraction = fract(texel_position);
    let maximum = i32(dimensions.x) - 1;
    let maximum_y = i32(dimensions.y) - 1;
    let p00 = clamp(base, vec2(0), vec2(maximum, maximum_y));
    let p10 = clamp(base + vec2(1, 0), vec2(0), vec2(maximum, maximum_y));
    let p01 = clamp(base + vec2(0, 1), vec2(0), vec2(maximum, maximum_y));
    let p11 = clamp(base + vec2(1), vec2(0), vec2(maximum, maximum_y));
    let row_0 = mix(
        textureLoad(bed_height, p00, 0).r,
        textureLoad(bed_height, p10, 0).r,
        fraction.x,
    );
    let row_1 = mix(
        textureLoad(bed_height, p01, 0).r,
        textureLoad(bed_height, p11, 0).r,
        fraction.x,
    );
    let height = mix(row_0, row_1, fraction.y) * range.y + range.x;
    return max(range.z - height, 0.0);
}

// The bed map is world-static, so Crest's cascade blending for SeaFloorDepth
// collapses to one lookup; only the shallow-attenuation cap remains.
fn blended_water_depth(world_xz: vec2<f32>) -> f32 {
    return min(bed_water_depth(world_xz), surface.sea_floor.x);
}

// Direct sunlight reaches the bed before Beer-Lambert mixing.
fn caustic_bed_radiance(
    scene_colour: vec3<f32>,
    world_xz: vec2<f32>,
    water_depth: f32,
    time: f32,
    sun_direction: vec3<f32>,
    sun_unexposed_radiance: vec3<f32>,
    sun_shadow: f32,
) -> vec3<f32> {
    let strength = surface.caustics.x * surface.sea_floor.w;
    if strength <= 0.0 {
        return scene_colour;
    }
    let maximum_depth = surface.caustics.w;
    if water_depth >= maximum_depth || sun_direction.y <= 0.0 || sun_shadow <= 0.0 {
        return scene_colour;
    }
    let scale = surface.caustics.y * CAUSTIC_CELLS_PER_TILE;
    let scroll = surface.caustics.z * time / scale;
    // Incommensurate scale and directions prevent the two layers from locking.
    let uv_a = world_xz / scale + vec2(scroll, 0.63 * scroll);
    let uv_b = world_xz * 1.37 / scale + vec2(-0.71 * scroll, 0.43 * scroll);
    let a = textureSampleLevel(
        foam_pattern,
        foam_pattern_sampler,
        uv_a,
        0.0,
    ).g;
    let b = textureSampleLevel(
        foam_pattern,
        foam_pattern_sampler,
        uv_b,
        0.0,
    ).g;
    let pattern = CAUSTIC_FOCUS_GAIN * a * b;
    let depth_gate = 1.0 - smoothstep(0.0, maximum_depth, water_depth);
    let direct_lux = dot(
        max(sun_unexposed_radiance, vec3(0.0)),
        CAUSTIC_LUMINANCE_WEIGHTS,
    );
    let daylight = clamp(
        (direct_lux - CAUSTIC_DAYLIGHT_MIN_LUX)
            / (CAUSTIC_REFERENCE_SUN_LUX - CAUSTIC_DAYLIGHT_MIN_LUX),
        0.0,
        1.0,
    );
    // The incoming sunlight already sits in the opaque colour after the
    // screen-space light-path scale. Focus that attenuated bed rather than
    // applying a second Beer-Lambert path here.
    let bed_incidence = max(sun_direction.y, 0.0);
    let sun_chroma = max(sun_unexposed_radiance, vec3(0.0))
        / max(direct_lux, CAUSTIC_DAYLIGHT_MIN_LUX);
    let focus = strength * depth_gate * pattern * daylight
        * bed_incidence * sun_shadow;
    return scene_colour + scene_colour * sun_chroma * focus;
}
