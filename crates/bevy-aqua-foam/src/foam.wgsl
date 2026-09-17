// Persistent reprojection, decay, whitecaps, and shore source.
#import aqua::foam::contract::{FOAM_LOD_COUNT, FOAM_TEXTURE_RESOLUTION, FOAM_TEXTURE_RESOLUTION_U32}
#import bevy_aqua_core::waves_sample::{
    CascadeLayout,
    CascadeParams,
    UV_CENTER,
    world_to_uv,
}

// Decoded "no bed data" depth: matches a cleared full-depth capture texel.
const NO_BED_DEPTH: f32 = 256.0;

struct FoamUniform {
    source_layout: CascadeLayout,
    target_layout: CascadeLayout,
    step: vec4<u32>,
    wave: vec4<f32>,
    shore: vec4<f32>,
}

@group(0) @binding(0) var source_foam: texture_2d_array<f32>;
@group(0) @binding(1) var source_sampler: sampler;
@group(0) @binding(2) var anim_waves: texture_2d_array<f32>;
@group(0) @binding(3) var waves_sampler: sampler;
@group(0) @binding(4) var bed_height: texture_2d<f32>;
@group(0) @binding(5) var target_foam: texture_storage_2d_array<rgba16float, write>;
@group(0) @binding(6) var<uniform> foam: FoamUniform;

fn texel_world(id: vec2<u32>, cascade: CascadeParams) -> vec2<f32> {
    let uv = (vec2<f32>(id) + vec2(0.5)) / FOAM_TEXTURE_RESOLUTION;
    let coverage = cascade.texel_width * cascade.texture_res;
    return cascade.center + (uv - vec2(UV_CENTER)) * coverage;
}

fn inside_source(uv: vec2<f32>) -> bool {
    let radius = UV_CENTER - 1.0 / FOAM_TEXTURE_RESOLUTION;
    // Chebyshev distance preserves the square cascade footprint.
    return max(abs(uv.x - UV_CENTER), abs(uv.y - UV_CENTER)) <= radius;
}

fn reproject(world_xz: vec2<f32>, slice: u32, source: CascadeLayout) -> f32 {
    let cascade = source.cascades[slice];
    let uv = world_to_uv(world_xz, cascade);
    if inside_source(uv) {
        return textureSampleLevel(source_foam, source_sampler, uv, i32(slice), 0.0).r;
    }
    if slice + 1u < FOAM_LOD_COUNT {
        let coarse = source.cascades[slice + 1u];
        let coarse_uv = world_to_uv(world_xz, coarse);
        if inside_source(coarse_uv) {
            return textureSampleLevel(
                source_foam,
                source_sampler,
                coarse_uv,
                i32(slice + 1u),
                0.0,
            ).r;
        }
    }
    return 0.0;
}

fn displacement(world_xz: vec2<f32>, slice: u32) -> vec4<f32> {
    let cascade = foam.target_layout.cascades[slice];
    return textureSampleLevel(
        anim_waves,
        waves_sampler,
        world_to_uv(world_xz, cascade),
        i32(slice),
        0.0,
    );
}

fn jacobian_foam_source(
    world_xz: vec2<f32>,
    slice: u32,
    cascade: CascadeParams,
) -> f32 {
    if foam.step.z != 0u {
        let determinant = textureSampleLevel(
            anim_waves,
            waves_sampler,
            world_to_uv(world_xz, cascade),
            i32(slice + FOAM_LOD_COUNT),
            0.0,
        ).w;
        return clamp(foam.wave.w - determinant, 0.0, 1.0);
    }
    let center = displacement(world_xz, slice);
    let offset_x = displacement(
        world_xz + vec2(cascade.texel_width, 0.0),
        slice,
    );
    let offset_z = displacement(
        world_xz + vec2(0.0, cascade.texel_width),
        slice,
    );
    let tangent_x = vec2(cascade.texel_width, 0.0)
        + offset_x.xz - center.xz;
    let tangent_z = vec2(0.0, cascade.texel_width)
        + offset_z.xz - center.xz;
    let determinant = (tangent_x.x * tangent_z.y - tangent_x.y * tangent_z.x)
        / (cascade.texel_width * cascade.texel_width);
    return clamp(foam.wave.w - determinant + 0.7 * center.w, 0.0, 1.0);
}

fn averaged_foam_source(
    world_xz: vec2<f32>,
    slice: u32,
    cascade: CascadeParams,
) -> f32 {
    if foam.step.z == 0u {
        return jacobian_foam_source(world_xz, slice, cascade);
    }
    let foam_texel_width = cascade.texel_width
        * cascade.texture_res / FOAM_TEXTURE_RESOLUTION;
    let quarter_texel = 0.25 * foam_texel_width;
    let offset = vec2(quarter_texel);
    return 0.25 * (
        jacobian_foam_source(world_xz - offset, slice, cascade)
            + jacobian_foam_source(
                world_xz + vec2(offset.x, -offset.y),
                slice,
                cascade,
            )
            + jacobian_foam_source(
                world_xz + vec2(-offset.x, offset.y),
                slice,
                cascade,
            )
            + jacobian_foam_source(world_xz + offset, slice, cascade)
    );
}

// Water depth below sea level. Unmapped samples use the cleared full-depth value.
fn bed_water_depth(world_xz: vec2<f32>) -> f32 {
    let range = foam.target_layout.bed_range;
    if range.y < 0.0 {
        return NO_BED_DEPTH;
    }
    let uv = (world_xz - foam.target_layout.bed_transform.xy)
        * foam.target_layout.bed_transform.zw;
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

fn update(id: vec3<u32>, use_previous_layout: bool, dt: f32) {
    if any(id.xy >= vec2<u32>(FOAM_TEXTURE_RESOLUTION_U32)) || id.z >= FOAM_LOD_COUNT {
        return;
    }
    let slice = id.z;
    let cascade = foam.target_layout.cascades[slice];
    let world_xz = texel_world(id.xy, cascade);
    var density = reproject(world_xz, slice, foam.target_layout);
    if use_previous_layout {
        density = reproject(world_xz, slice, foam.source_layout);
    }

    density *= max(0.0, 1.0 - foam.wave.y * dt);

    let center = displacement(world_xz, slice);
    density += 5.0 * dt * foam.wave.z
        * averaged_foam_source(world_xz, slice, cascade);

    var depth = bed_water_depth(world_xz + center.xz) + center.y;
    // Two world-anchored finite bands: a wet edge and a breaking-wave band.
    // Persistent reprojection and decay filter both under camera motion.
    let wet_edge = 1.0 - smoothstep(foam.shore.z, 2.0 * foam.shore.z, depth);
    let breaker_rise = smoothstep(foam.shore.z, foam.shore.w, depth);
    let breaker_fall = 1.0 - smoothstep(foam.shore.w, foam.shore.x, depth);
    let shore_source = max(wet_edge, 0.18 * breaker_rise * breaker_fall);
    density += foam.shore.y * dt * shore_source;

    density = clamp(density, 0.0, 1.0);
    textureStore(target_foam, vec2<i32>(id.xy), i32(slice), vec4(density, 0.0, 0.0, 0.0));
}

#ifdef FOAM_REPROJECT_PREVIOUS
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    update(id, true, 0.0);
}
#endif

#ifdef FOAM_UPDATE_CURRENT
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    update(id, false, foam.wave.x);
}
#endif

#ifndef FOAM_REPROJECT_PREVIOUS
#ifndef FOAM_UPDATE_CURRENT
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    update(id, true, foam.wave.x);
}
#endif
#endif
