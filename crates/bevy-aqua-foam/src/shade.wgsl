// Foam contributions to the composed cascade material: bicubic cascade
// sampling, breakup masks, bank streaks, bubble tint, and foam lighting.

#define_import_path aqua::foam::shade

#import bevy_pbr::mesh_view_bindings::globals

#import bevy_aqua_core::waves_sample::{CascadeParams, world_to_uv}
#import aqua::cascade::{LUMINANCE_EPSILON, LocalLightSample, MIN_SAMPLE_WEIGHT, RiverState, advected_world, capillary_resolved_weight, cascade_layout, foam_data, foam_pattern, foam_pattern_sampler, foam_sampler, lod_count, screen_texture_lod, surface}
#import aqua::foam::contract::{FOAM_PATTERN_RESOLUTION, FOAM_TEXTURE_RESOLUTION}
const INV_PI: f32 = 0.31830988618;
const CREST_FOAM_WHITE_COLOR: vec4<f32> = vec4(
    1.0, 0.99634784, 0.97199994, 0.6862745,
);
// The blue tint and 0.14 parallax preserve the subsurface foam cue.
const CREST_FOAM_BUBBLE_COLOR: vec3<f32> = vec3(0.64, 0.83, 0.82);
const CREST_FOAM_BUBBLE_PARALLAX: f32 = 0.14;
const CREST_FOAM_BUBBLE_COVERAGE: f32 = 1.68;
const CREST_FOAM_NORMAL_STRENGTH: f32 = 3.5;
const CREST_FOAM_SPECULAR_FALLOFF: f32 = 293.0;
const CREST_FOAM_SPECULAR_BOOST: f32 = 0.15;

// Rotation and independent scrolling avoid a close-range beat with the two main taps.
const FINE_FOAM_ROTATION: mat2x2<f32> = mat2x2(
    vec2(0.798636, 0.601815),
    vec2(-0.601815, 0.798636),
);
const FINE_FOAM_SCROLL_DIRECTION: vec2<f32> = vec2(-0.819152, 0.573576);
const FINE_FOAM_SCALE: f32 = 0.55;
// Linear-light mean of the Unity-style 512px Foam2 import.
const CREST_FOAM_LINEAR_MEAN: f32 = 0.426906;

// GodotOceanWaves `water.gdshader`: cubic B-spline weights for the GPU Gems 2
// four-bilinear-tap approximation to bicubic filtering.
fn cubic_weights(value: f32) -> vec4<f32> {
    let squared = value * value;
    let cubed = squared * value;
    return vec4(
        -cubed + 3.0 * squared - 3.0 * value + 1.0,
        3.0 * cubed - 6.0 * squared + 4.0,
        -3.0 * cubed + 3.0 * squared + 3.0 * value + 1.0,
        cubed,
    ) / 6.0;

}
fn sample_foam_bicubic(uv: vec2<f32>, layer: i32, resolution: f32) -> f32 {
    let texel_position = uv * resolution + vec2(0.5);
    let fraction = fract(texel_position);
    let wx = cubic_weights(fraction.x);
    let wy = cubic_weights(fraction.y);
    let groups = vec4(
        wx.x + wx.y,
        wx.z + wx.w,
        wy.x + wy.y,
        wy.z + wy.w,
    );
    let positions = (
        vec4(wx.y, wx.w, wy.y, wy.w) / groups
            + vec4(-1.5, 0.5, -1.5, 0.5)
            + floor(texel_position).xxyy
    ) / resolution;
    let blend = vec2(
        groups.x / (groups.x + groups.y),
        groups.z / (groups.z + groups.w),
    );
    let lower = mix(
        textureSampleLevel(
            foam_data,
            foam_sampler,
            positions.yw,
            layer,
            0.0,
        ).r,
        textureSampleLevel(
            foam_data,
            foam_sampler,
            positions.xw,
            layer,
            0.0,
        ).r,
        blend.x,
    );
    let upper = mix(
        textureSampleLevel(
            foam_data,
            foam_sampler,
            positions.yz,
            layer,
            0.0,
        ).r,
        textureSampleLevel(
            foam_data,
            foam_sampler,
            positions.xz,
            layer,
            0.0,
        ).r,
        blend.x,
    );
    return mix(lower, upper, blend.y);
}

fn sample_foam_cascade(world_xz: vec2<f32>, cascade: CascadeParams, layer: i32) -> f32 {
    let uv = world_to_uv(world_xz, cascade);
    let bilinear = textureSampleLevel(
        foam_data,
        foam_sampler,
        uv,
        layer,
        0.0,
    ).r;
    // Gerstner's smooth source already hides its texels; bicubic taps only help FFT.
    if surface.reflection.x < 0.5 || surface.debug.w > 0.5 {
        return bilinear;
    }
    let bicubic = sample_foam_bicubic(uv, layer, FOAM_TEXTURE_RESOLUTION);
    let coverage = cascade.texel_width * cascade.texture_res;
    let pixels_per_meter = FOAM_TEXTURE_RESOLUTION / coverage;
    return mix(bicubic, bilinear, min(1.0, pixels_per_meter * 0.1));

}
fn sample_foam_density(world_xz: vec2<f32>, lod: u32, alpha: f32) -> f32 {
    let smaller = cascade_layout.cascades[lod];
    let bigger = cascade_layout.cascades[lod + 1u];
    let smaller_weight = (1.0 - alpha) * smaller.weight;
    let bigger_weight = (1.0 - smaller_weight) * bigger.weight;
    var density = 0.0;
    if smaller_weight > MIN_SAMPLE_WEIGHT {
        density += smaller_weight
            * sample_foam_cascade(world_xz, smaller, i32(lod));
    }
    if bigger_weight > MIN_SAMPLE_WEIGHT && lod + 1u < lod_count() {
        density += bigger_weight
            * sample_foam_cascade(world_xz, bigger, i32(lod + 1u));
    }
    return clamp(density, 0.0, 1.0);

}
fn surface_foam_mask(
    advected_xz: vec2<f32>,
    lod: u32,
    alpha: f32,
    density: f32,
    sample_offset: vec2<f32>,
) -> f32 {
    let cascade = cascade_layout.cascades[lod];
    let texture_scale = surface.foam.x * cascade.scale / 25.0;
    // The breakup pattern rides the current so foam streaks do not slide
    // against the water carrying them; callers hand over the advected
    // position so this module never touches per-invocation state.
    let offset = vec2(globals.time / 10.0) + sample_offset;
    let pattern_xz = advected_xz;
    let texture_width = textureDimensions(foam_pattern).x;
    let near_lod = screen_texture_lod(texture_scale / 1.25, texture_width);
    let near = textureSampleLevel(
        foam_pattern,
        foam_pattern_sampler,
        (1.25 * pattern_xz + offset) / texture_scale,
        near_lod,
    ).r;
    let far_scale = 2.0 * texture_scale;
    let far_lod = screen_texture_lod(far_scale / 1.25, texture_width);
    let far = textureSampleLevel(
        foam_pattern,
        foam_pattern_sampler,
        (1.25 * pattern_xz + offset) / far_scale,
        far_lod,
    ).r;
    var pattern = mix(near, far, alpha);
    let fine_scroll = FINE_FOAM_SCROLL_DIRECTION * globals.time / 17.0;
    let fine_coordinates =
        FINE_FOAM_ROTATION * (1.25 * pattern_xz + sample_offset) + fine_scroll;
    let fine_scale = FINE_FOAM_SCALE * texture_scale;
    let fine_lod = screen_texture_lod(fine_scale / 1.25, texture_width);
    let fine = textureSampleLevel(
        foam_pattern,
        foam_pattern_sampler,
        fine_coordinates / fine_scale,
        fine_lod,
    ).r;
    // Mean-one near-field modulation hides the 512-grid footprint without
    // changing the far-field pattern.
    let fine_modulation = 1.0 + 0.65 * (fine - CREST_FOAM_LINEAR_MEAN);
    let fine_weight = capillary_resolved_weight(advected_xz);
    let modulated_pattern = pattern * mix(1.0, fine_modulation, fine_weight);

    pattern = select(pattern, modulated_pattern, fine_weight > 0.0);
    let black_point = clamp(1.0 - density, 0.0, 1.0);
    return smoothstep(black_point, black_point + surface.foam.y, pattern);
}

// Add elongated, flow-aligned foam where fast water meets a bank.
fn river_streak_density(state: RiverState, world_xz: vec2<f32>, lod: u32, alpha: f32) -> f32 {
    if !state.enabled {
        return 0.0;
    }
    let speed = length(state.sample.xy);
    // Same 8 m near-bank band as the ripple scale.
    let margin_ratio = clamp(state.sample.z / 8.0, 0.0, 1.0);
    let strength = (1.0 - margin_ratio) * clamp(speed / 1.6 - 0.2, 0.0, 1.0);
    if strength <= 0.001 {
        return 0.0;
    }
    let dir = state.flow / max(speed, 1e-4);
    let along = dot(dir, world_xz) / (1.0 + 0.9 * min(speed, 4.0));
    let across = dot(vec2(-dir.y, dir.x), world_xz);
    // The mask input is a synthetic along/across coordinate; advecting it
    // here matches the previous in-mask advection of that coordinate.
    return surface_foam_mask(
        advected_world(vec2(along, across * 1.6)),
        lod,
        alpha,
        strength * 1.4,
        vec2(0.0),
    );
}

fn foam_bubble_colour(
    displaced_xz: vec2<f32>,
    undisplaced_xz: vec2<f32>,
    lod: u32,
    alpha: f32,
    density: f32,
    surface_normal: vec3<f32>,
    to_view: vec3<f32>,
    ambient_radiance: vec3<f32>,
) -> vec3<f32> {
    let wind_direction = vec2(0.866, 0.5);
    let bubble_world = mix(undisplaced_xz, displaced_xz, 0.7)
        + 0.5 * globals.time * wind_direction;
    let bubble_uv = bubble_world / surface.foam.x
        + 0.125 * surface_normal.xz;
    let parallax = -CREST_FOAM_BUBBLE_PARALLAX * to_view.xz
        / max(dot(surface_normal, to_view), LUMINANCE_EPSILON);
    let smaller = cascade_layout.cascades[lod];
    let bigger = cascade_layout.cascades[min(lod + 1u, lod_count())];
    let smaller_sample = textureSampleLevel(
        foam_pattern,
        foam_pattern_sampler,
        (0.74 * bubble_uv + parallax) / (smaller.scale / 25.0),
        3.0,
    ).r;
    let bigger_sample = textureSampleLevel(
        foam_pattern,
        foam_pattern_sampler,
        (0.74 * bubble_uv + parallax) / (bigger.scale / 25.0),
        3.0,
    ).r;
    let bubble_texture = mix(smaller_sample, bigger_sample, alpha);
    let coverage = clamp(density * CREST_FOAM_BUBBLE_COVERAGE, 0.0, 1.0);
    return bubble_texture * CREST_FOAM_BUBBLE_COLOR
        * coverage * ambient_radiance;
}

fn local_foam_light(
    sample: LocalLightSample,
    foam_normal: vec3<f32>,
    to_view: vec3<f32>,
) -> vec3<f32> {
    let foam_ndl = max(dot(foam_normal, sample.direction), 0.0);
    let foam_reflection = reflect(-to_view, foam_normal);
    let diffuse = CREST_FOAM_WHITE_COLOR.rgb
        * INV_PI * surface.foam.z * sample.radiance * foam_ndl;
    let specular = pow(
        max(dot(foam_reflection, sample.direction), 0.0),
        CREST_FOAM_SPECULAR_FALLOFF,
    ) * CREST_FOAM_SPECULAR_BOOST * sample.radiance;
    return diffuse + specular;
}
