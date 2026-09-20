// Wave-surface sampling for the composed cascade material: cascade-space
// displacement and FFT normal-cross reads, far-tier fallbacks, the Jacobian
// SSS pinch, and the scrolled detail-normal slopes. Owned by bevy-aqua-waves.

#define_import_path aqua::waves::displace

#import bevy_pbr::mesh_view_bindings::globals

#import aqua::cascade::{CREST_SSS_MAXIMUM, CREST_SSS_RANGE, MIN_NORMAL_Y, MIN_SAMPLE_WEIGHT, advected_world, cascade_layout, detail_normal, detail_sampler, fft_surface, flow_frame, lod_alpha, lod_count, lod_data, lod_sampler, screen_texture_lod, screen_xz_footprint, set_spectral_filtered_variance, surface, world_to_uv}

// Both animated layers travel with the authored dominant wave heading. Their
// small angular spread prevents lockstep motion without the old counter-moving
// texture. Spatial rotation and irrational scale, rather than opposite travel,
// break up the common repeat lattice.
const DETAIL_TRAVEL_0: vec2<f32> = vec2(0.98480775, -0.17364818);
const DETAIL_TRAVEL_1: vec2<f32> = vec2(0.97437006, 0.22495105);
const DETAIL_B_SCALE: f32 = 1.41421356;
const DETAIL_B_ROTATION: mat2x2<f32> = mat2x2(
    vec2(0.7313537, 0.6819984),
    vec2(-0.6819984, 0.7313537),
);
const CAPILLARY_A_SCALE: f32 = 0.0625;
const CAPILLARY_B_SCALE: f32 = 0.09730085;
const CAPILLARY_RESOLVED_STRENGTH: f32 = 0.45;
const CAPILLARY_RESOLVED_ENERGY: f32 =
    CAPILLARY_RESOLVED_STRENGTH * CAPILLARY_RESOLVED_STRENGTH;
const CAPILLARY_A_ROTATION: mat2x2<f32> = mat2x2(
    vec2(0.819152, 0.573576),
    vec2(-0.573576, 0.819152),
);
const CAPILLARY_B_ROTATION: mat2x2<f32> = mat2x2(
    vec2(0.4539905, -0.8910065),
    vec2(0.8910065, 0.4539905),
);

// Mean square of the decoded WaveNormals.png XY slope. Used to account for
// mip-filtered detail and separately distance-faded capillary variance.
const WAVE_NORMALS_SLOPE_VARIANCE: f32 = 0.05602466;
const NORMAL_SCROLL_MULTIPLIER: f32 = 1.875;
const NORMAL_SCROLL_POWER: f32 = 1.4;
fn direct_displacement(world_xz: vec2<f32>, lod: u32) -> vec3<f32> {
    let sampled_xz = advected_world(world_xz);
    let cascade = cascade_layout.cascades[lod];
    return textureSampleLevel(
        lod_data,
        lod_sampler,
        world_to_uv(sampled_xz, cascade),
        i32(lod),
        0.0,
    ).xyz;
}

fn direct_fft_normal_cross(world_xz: vec2<f32>, lod: u32) -> vec3<f32> {
    let sampled_xz = advected_world(world_xz);
    let cascade = cascade_layout.cascades[lod];
    return textureSampleLevel(
        fft_surface,
        lod_sampler,
        world_to_uv(sampled_xz, cascade),
        i32(lod),
        0.0,
    ).xyz;
}

fn sample_fft_normal_cross(world_xz: vec2<f32>, lod: u32, alpha: f32) -> vec3<f32> {
    let smaller = cascade_layout.cascades[lod];
    let bigger = cascade_layout.cascades[lod + 1u];
    let smaller_weight = (1.0 - alpha) * smaller.weight;
    let bigger_weight = (1.0 - smaller_weight) * bigger.weight;
    var normal_cross = vec3(0.0);
    var sampled_weight = 0.0;
    if smaller_weight > MIN_SAMPLE_WEIGHT {
        normal_cross += smaller_weight * direct_fft_normal_cross(world_xz, lod);
        sampled_weight += smaller_weight;
    }
    if bigger_weight > MIN_SAMPLE_WEIGHT && lod + 1u < lod_count() {
        normal_cross += bigger_weight * direct_fft_normal_cross(world_xz, lod + 1u);
        sampled_weight += bigger_weight;
    }
    normal_cross.y += max(1.0 - sampled_weight, 0.0);
    return normal_cross;
}

// Exclusive periodic fields have fixed world-space frequencies. Mesh-ring
// coverage must not change the lighting spectrum or turn it into roughness.
// XYZ stores a 3D displacement derivative; W stores vertical-slope E[g²].
fn spectral_depth_weights(depth: f32, wavelength: f32) -> vec2<f32> {
    let relative_depth = clamp(2.0 * depth / wavelength, 0.0, 1.0);
    let base = smoothstep(0.0, 1.0, relative_depth);
    let breaker = 4.0 * base * (1.0 - base);
    let vertical = base * (1.0 + 0.18 * breaker);
    let chop_ratio = mix(0.55, 1.0, smoothstep(0.15, 0.85, base));
    return mix(vec2(1.0), vec2(vertical * chop_ratio, vertical), surface.detail.w);
}

fn derivative_slope(dx: vec3<f32>, dz: vec3<f32>) -> vec2<f32> {
    let n = cross(vec3(0.0, 0.0, 1.0) + dz, vec3(1.0, 0.0, 0.0) + dx);
    return n.xz / max(n.y, MIN_NORMAL_Y);
}

fn local_fft_slope(world_xz: vec2<f32>, layer: u32) -> vec3<f32> {
    let cascade = cascade_layout.cascades[layer];
    let uv = world_to_uv(advected_world(world_xz), cascade);
    let level = clamp(log2(max(screen_xz_footprint() / cascade.texel_width, 1.0)),
        0.0, f32(textureNumLevels(fft_surface) - 1u));
    let edge = min(min(uv.x, uv.y), min(1.0 - uv.x, 1.0 - uv.y));
    let guard = max(2.0 * exp2(level) / cascade.texture_res, 2.0 * cascade.inv_texture_res);
    let valid = smoothstep(guard, 2.0 * guard, edge);
    let n = mix(
        textureSampleLevel(fft_surface, lod_sampler, uv, i32(layer), floor(level)).xyz,
        textureSampleLevel(fft_surface, lod_sampler, uv, i32(layer), ceil(level)).xyz,
        fract(level));
    return vec3(n.xz / max(n.y, MIN_NORMAL_Y), valid);
}

fn filtered_fft_normal(world_xz: vec2<f32>, water_depth: f32) -> vec3<f32> {
    let split_long_band = cascade_layout.bed_range.y >= 0.0 && surface.detail.w > 0.0;
    let field_count = select(5u, 8u, split_long_band);
    var derivative_x = vec3(0.0);
    var derivative_z = vec3(0.0);
    var expected_small = vec2(0.0);
    var expected_large = vec2(0.0);
    var filtered_variance = 0.0;
    let footprint = screen_xz_footprint();
    let shallow = (1.0 - smoothstep(4.0, 6.0, water_depth))
        * (1.0 - smoothstep(6.0, 12.0, 2.0 * footprint));
    var local_layer = 4u;
    var local_retained = 0.0;
    if split_long_band && shallow > 0.0 {
        for (var layer = 0u; layer < 4u; layer++) {
            let retained = 1.0 - smoothstep(0.0, 1.0,
                lod_alpha(world_xz, cascade_layout.cascades[layer]));
            if retained > 0.0 {
                local_layer = layer;
                local_retained = retained;
                break;
            }
        }
    }
    for (var field = 0u; field < field_count; field++) {
        let cascade = cascade_layout.cascades[min(field, 4u)];
        let period = cascade.texel_width * cascade.texture_res;
        var minimum = 0.5 * cascade.max_wavelength;
        var maximum = select(cascade.max_wavelength, period / 4.0, field >= 4u);
        if split_long_band && field >= 4u {
            let octaves = log2(maximum / minimum);
            maximum = minimum * exp2(octaves * f32(field - 3u) / 4.0);
            minimum *= exp2(octaves * f32(field - 4u) / 4.0);
        }
        let level = clamp(log2(max(footprint / cascade.texel_width, 1.0)),
            0.0, f32(textureNumLevels(fft_surface) - 1u));
        let uv = fract(world_to_uv(advected_world(world_xz), cascade));
        let layer = i32(lod_count() + 2u * field);
        let dx = textureSampleLevel(fft_surface, detail_sampler, uv, layer, level);
        let dz = textureSampleLevel(fft_surface, detail_sampler, uv, layer + 1, level);
        let retained = 1.0 - smoothstep(0.5 * maximum, maximum, 2.0 * footprint);
        // Long-band bins match the producer's representative wavelengths.
        // Fine octaves use one geometric-mean wavelength for local shading;
        // their displacement still retains the producer's four-bin shoaling.
        let attenuation = spectral_depth_weights(water_depth, sqrt(minimum * maximum));
        let lane_weights = vec3(attenuation.x, attenuation.y, attenuation.x);
        let resolved_dx = retained * lane_weights * dx.xyz;
        let resolved_dz = retained * lane_weights * dz.xyz;
        derivative_x += resolved_dx;
        derivative_z += resolved_dz;
        if local_layer < 4u && field < 4u && field >= local_layer && retained > 0.0 {
            // This prediction is bounded, not periodic. The legacy endpoint
            // support masks below reject its cache edges before wrap can leak.
            let local_uv = world_to_uv(advected_world(world_xz), cascade);
            let prediction = textureSampleLevel(fft_surface, detail_sampler,
                local_uv, i32(21u + field), level);
            let predicted = retained * prediction.xy
                + 4.0 * retained * (1.0 - retained) * prediction.zw;
            expected_small += predicted;
            if field > local_layer { expected_large += predicted; }
        }
        let resolved_energy = retained * retained * (dx.y * dx.y + dz.y * dz.y);
        filtered_variance += attenuation.y * attenuation.y
            * max(dx.w + dz.w - resolved_energy, 0.0);
    }
    set_spectral_filtered_variance(filtered_variance);
    var slope = derivative_slope(derivative_x, derivative_z);
    // Preserve the producer's accurate local short-wave shallow response.
    // This is only a small residual below6m depth, not a spectral-band fade.
    // It is never periodic and vanishes before its local filter loses support.
    if local_layer < 4u {
        let local_outer = local_fft_slope(world_xz, 4u);
        let small = local_fft_slope(world_xz, local_layer);
        let small_residual = (small.xy - local_outer.xy - expected_small)
            * small.z * local_outer.z;
        var large_residual = vec2(0.0);
        if local_layer + 1u < 4u {
            let large = local_fft_slope(world_xz, local_layer + 1u);
            large_residual = (large.xy - local_outer.xy - expected_large)
                * large.z * local_outer.z;
        }
        slope += shallow * mix(large_residual, small_residual, local_retained);
    }
    return normalize(vec3(slope.x, 1.0, slope.y));
}

fn outer_cascade_weight(world_xz: vec2<f32>) -> f32 {
    let outer_lod = lod_count() - 1u;
    return 1.0 - lod_alpha(world_xz, cascade_layout.cascades[outer_lod]);

}
fn far_displacement(world_xz: vec2<f32>) -> vec3<f32> {
    let weight = outer_cascade_weight(world_xz);
    if weight <= MIN_SAMPLE_WEIGHT {
        return vec3(0.0);
    }
    return weight * direct_displacement(world_xz, lod_count() - 1u);

}
fn far_normal_cross(world_xz: vec2<f32>) -> vec3<f32> {
    let outer_lod = lod_count() - 1u;
    let cascade = cascade_layout.cascades[outer_lod];
    let weight = outer_cascade_weight(world_xz);
    if weight <= MIN_SAMPLE_WEIGHT {
        return vec3(0.0, 1.0, 0.0);
    }
    if surface.reflection.x > 0.5 {
        let cached = direct_fft_normal_cross(world_xz, outer_lod);
        return mix(vec3(0.0, 1.0, 0.0), cached, weight);
    }
    let center = weight * direct_displacement(world_xz, outer_lod);
    let offset_x = weight * direct_displacement(
        world_xz + vec2(cascade.texel_width, 0.0),
        outer_lod,
    );
    let offset_z = weight * direct_displacement(
        world_xz + vec2(0.0, cascade.texel_width),
        outer_lod,
    );
    let tangent_x = vec3(cascade.texel_width, 0.0, 0.0) + offset_x - center;
    let tangent_z = vec3(0.0, 0.0, cascade.texel_width) + offset_z - center;
    var normal_cross = cross(tangent_z, tangent_x);
    normal_cross.y = max(normal_cross.y, MIN_NORMAL_Y);
    return normal_cross;

// Crest `OceanHelpersNew.hlsl::SampleDisplacementsNormals`: horizontal
// displacement Jacobian determinant. Compression/pinch has determinant < 1.
}
fn displacement_jacobian(world_xz: vec2<f32>, lod: u32) -> f32 {
    if surface.reflection.x > 0.5 {
        let cascade = cascade_layout.cascades[lod];
        return textureSampleLevel(
            fft_surface,
            lod_sampler,
            world_to_uv(advected_world(world_xz), cascade),
            i32(lod),
            0.0,
        ).w;
    }
    let texel_width = cascade_layout.cascades[lod].texel_width;
    let center = direct_displacement(world_xz, lod);
    let offset_x = direct_displacement(world_xz + vec2(texel_width, 0.0), lod);
    let offset_z = direct_displacement(world_xz + vec2(0.0, texel_width), lod);
    let tangent_x = vec2(texel_width, 0.0) + offset_x.xz - center.xz;
    let tangent_z = vec2(0.0, texel_width) + offset_z.xz - center.xz;
    return (tangent_x.x * tangent_z.y - tangent_x.y * tangent_z.x)
        / (texel_width * texel_width);

}
fn crest_sss(world_xz: vec2<f32>, lod: u32, alpha: f32) -> f32 {
    let smaller = cascade_layout.cascades[lod];
    let bigger = cascade_layout.cascades[lod + 1u];
    let smaller_weight = (1.0 - alpha) * smaller.weight;
    let bigger_weight = (1.0 - smaller_weight) * bigger.weight;
    var determinant = 0.0;
    if smaller_weight > MIN_SAMPLE_WEIGHT {
        determinant += smaller_weight * displacement_jacobian(world_xz, lod);
    }
    if bigger_weight > MIN_SAMPLE_WEIGHT && lod + 1u < lod_count() {
        determinant += bigger_weight * displacement_jacobian(world_xz, lod + 1u);
    }
    if lod + 1u >= lod_count() {
        determinant += 1.0 - smaller_weight;
    }
    return clamp(CREST_SSS_MAXIMUM - CREST_SSS_RANGE * determinant, 0.0, 1.0);
}

fn sample_detail_normal(uv: vec2<f32>, lod: f32) -> vec3<f32> {
    let packed = textureSampleLevel(detail_normal, detail_sampler, uv, lod);
    let slope = 2.0 * packed.xy - vec2(1.0);
    let second_moment = 2.0 * packed.z;
    return vec3(slope, max(second_moment - dot(slope, slope), 0.0));
}

// Rotate a local spread direction into the authored dominant travel frame.
fn heading_frame(local_direction: vec2<f32>) -> vec2<f32> {
    var heading = surface.advection.zw;
    if dot(heading, heading) < 1e-6 {
        heading = vec2(1.0, 0.0);
    } else {
        heading = normalize(heading);
    }
    return vec2(
        heading.x * local_direction.x - heading.y * local_direction.y,
        heading.y * local_direction.x + heading.x * local_direction.y,
    );
}

fn transformed_detail_normal(
    world_xz: vec2<f32>,
    rotation: mat2x2<f32>,
    metres_per_repeat: f32,
) -> vec3<f32> {
    let lod = screen_texture_lod(metres_per_repeat, textureDimensions(detail_normal).x);
    let sampled = sample_detail_normal(rotation * world_xz / metres_per_repeat, lod);
    return vec3(transpose(rotation) * sampled.xy, sampled.z);
}

// Two co-travelling layers at fixed world scales. Geometry LOD must not
// enlarge painted normal-map features; texture mips handle minification.
fn detail_normal_sample(world_xz: vec2<f32>, lod: u32, alpha: f32, ripple: f32) -> vec3<f32> {
    let advected_xz = advected_world(world_xz);
    let cascade = cascade_layout.cascades[0u];
    let stretch = surface.detail.x * cascade.scale / 100.0;
    let speed = pow(
        log(1.0 + 2.0 * cascade.texel_width) * NORMAL_SCROLL_MULTIPLIER,
        NORMAL_SCROLL_POWER,
    );
    let a = transformed_detail_normal(
        flow_frame(advected_xz - heading_frame(DETAIL_TRAVEL_0) * globals.time * speed),
        mat2x2(vec2(1.0, 0.0), vec2(0.0, 1.0)), stretch,
    );
    let b = transformed_detail_normal(
        flow_frame(advected_xz - heading_frame(DETAIL_TRAVEL_1) * globals.time * speed),
        DETAIL_B_ROTATION, DETAIL_B_SCALE * stretch,
    );
    let strength = ripple * surface.detail.y * surface.detail.z;
    return vec3(strength * (a.xy + b.xy), strength * strength * (a.z + b.z));
}

// GodotOceanWaves anchors this as a displacement-free fine normal cascade.
// DIVERGENCE: Aqua reuses Crest's trilinear WaveNormals at 16x frequency and
// its shipped 0.08 strength because minification makes it weaker than an FFT
// normal field; rotation avoids aligning its period with the authored layers.
fn capillary_normal_slope(world_xz: vec2<f32>, ripple: f32) -> vec3<f32> {
    let cascade = cascade_layout.cascades[0u];
    let base_stretch = surface.detail.x * cascade.scale / 100.0;
    let speed = pow(
        log(1.0 + 2.0 * cascade.texel_width) * NORMAL_SCROLL_MULTIPLIER,
        NORMAL_SCROLL_POWER,
    );
    let direction = heading_frame(vec2(1.0, 0.0));
    let scrolled = flow_frame(
        advected_world(world_xz) - direction * globals.time * speed,
    );
    let sample_a = transformed_detail_normal(
        scrolled,
        CAPILLARY_A_ROTATION,
        CAPILLARY_A_SCALE * base_stretch,
    );
    let sample_b = transformed_detail_normal(
        scrolled,
        CAPILLARY_B_ROTATION,
        CAPILLARY_B_SCALE * base_stretch,
    );
    // Average two independent fields with unit-variance normalization, then
    // reduce only resolved amplitude. The LEAN roughness lane retains the
    // authored full capillary energy as this detail becomes sub-pixel.
    let decorrelated = 0.70710678 * (sample_a.xy + sample_b.xy);
    let filtered_variance = 0.5 * (sample_a.z + sample_b.z);
    let strength = ripple * CAPILLARY_RESOLVED_STRENGTH * surface.capillary.y;
    return vec3(strength * decorrelated, strength * strength * filtered_variance);
}
