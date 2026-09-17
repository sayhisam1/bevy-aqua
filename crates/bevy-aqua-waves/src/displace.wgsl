// Wave-surface sampling for the composed cascade material: cascade-space
// displacement and FFT normal-cross reads, far-tier fallbacks, the Jacobian
// SSS pinch, and the scrolled detail-normal slopes. Owned by bevy-aqua-waves.

#define_import_path aqua::waves::displace

#import bevy_pbr::mesh_view_bindings::globals

#import bevy_aqua_core::waves_sample::{lod_alpha, world_to_uv}
#import aqua::cascade::{CREST_SSS_MAXIMUM, CREST_SSS_RANGE, MIN_NORMAL_Y, MIN_SAMPLE_WEIGHT, advected_world, cascade_layout, detail_normal, detail_sampler, flow_frame, lod_count, lod_data, lod_sampler, screen_texture_lod, surface}

const NORMAL_DIRECTION_0: vec2<f32> = vec2(0.94, 0.34);
const NORMAL_DIRECTION_1: vec2<f32> = vec2(-0.85, -0.53);
const CAPILLARY_DIRECTION: vec2<f32> = vec2(0.4226, 0.9063);
const CAPILLARY_ROTATION: mat2x2<f32> = mat2x2(
    vec2(0.819152, 0.573576),
    vec2(-0.573576, 0.819152),
);

// Mean square of the decoded WaveNormals.png XY slope. Resolved fade energy
// moves into unresolved roughness instead of disappearing with distance.
const WAVE_NORMALS_SLOPE_VARIANCE: f32 = 0.05602466;
const NORMAL_SCROLL_MULTIPLIER: f32 = 1.875;
const NORMAL_SCROLL_POWER: f32 = 1.4;
// Deterministic mean-square slope per owned wavelength octave. FFT values are
// phase means 2 * sum(k^2 * |h0(k)|^2) for the shipped H0 realization; Gerstner values are
// sums of 0.5 * (amplitude * wave_number)^2 for the shipped components.
const FFT_JONSWAP_SLOPE_VARIANCE: array<f32, 5> = array(
    0.06393624, 0.06369724, 0.06306524, 0.06170338, 0.06048288,
);
const GERSTNER_SLOPE_VARIANCE: array<f32, 5> = array(
    0.0194041, 0.027484603, 0.015417861, 0.0066486476, 0.010458008,
);

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

fn sample_fft_surface(uv: vec2<f32>, lod: u32) -> vec4<f32> {
    return textureSampleLevel(
        lod_data,
        lod_sampler,
        uv,
        i32(lod) + i32(lod_count()),
        0.0,
    );
}

fn direct_fft_normal_cross(world_xz: vec2<f32>, lod: u32) -> vec3<f32> {
    let sampled_xz = advected_world(world_xz);
    let cascade = cascade_layout.cascades[lod];
    return sample_fft_surface(world_to_uv(sampled_xz, cascade), lod).xyz;
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

fn outer_cascade_weight(world_xz: vec2<f32>) -> f32 {
    let outer_lod = lod_count() - 1u;
    return 1.0 - lod_alpha(world_xz, cascade_layout.cascades[outer_lod], cascade_layout);

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
        return sample_fft_surface(
            world_to_uv(advected_world(world_xz), cascade),
            lod,
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

// Crest `OceanNormalMapping.hlsl::SampleNormalMaps`: two opposed scroll
// directions and a second doubled scale blended by the cascade transition.
// z carries LEAN slope variance removed by mip filtering.
fn detail_normal_sample(world_xz: vec2<f32>, lod: u32, alpha: f32, ripple: f32) -> vec3<f32> {
    // Sub-grid ripples ride the current with the waves they decorate, and
    // stretch along it inside river bodies.
    let scrolled_xz = flow_frame(advected_world(world_xz));
    let cascade = cascade_layout.cascades[lod];
    let stretch = surface.detail.x * cascade.scale / 100.0;
    let speed_near = pow(
        log(1.0 + 2.0 * cascade.texel_width) * NORMAL_SCROLL_MULTIPLIER,
        NORMAL_SCROLL_POWER,
    );
    let speed_far = pow(
        log(1.0 + 4.0 * cascade.texel_width) * NORMAL_SCROLL_MULTIPLIER,
        NORMAL_SCROLL_POWER,
    );
    let texture_width = textureDimensions(detail_normal).x;
    let near_lod = screen_texture_lod(stretch, texture_width);
    let near_a = sample_detail_normal(
        (scrolled_xz + NORMAL_DIRECTION_0 * globals.time * speed_near) / stretch,
        near_lod,
    );
    let near_b = sample_detail_normal(
        (scrolled_xz + NORMAL_DIRECTION_1 * globals.time * speed_near) / stretch,
        near_lod,
    );
    let far_stretch = 2.0 * stretch;
    let far_lod = screen_texture_lod(far_stretch, texture_width);
    let far_a = sample_detail_normal(
        (scrolled_xz + NORMAL_DIRECTION_0 * globals.time * speed_far) / far_stretch,
        far_lod,
    );
    let far_b = sample_detail_normal(
        (scrolled_xz + NORMAL_DIRECTION_1 * globals.time * speed_far) / far_stretch,
        far_lod,
    );
    let near = near_a.xy + near_b.xy;
    let far = far_a.xy + far_b.xy;
    let near_weight = 1.0 - alpha;
    let variance = near_weight * near_weight * (near_a.z + near_b.z)
        + alpha * alpha * (far_a.z + far_b.z);
    let strength = ripple * surface.detail.y * surface.detail.z;
    return vec3(strength * mix(near, far, alpha), strength * strength * variance);
}

// GodotOceanWaves anchors this as a displacement-free fine normal cascade.
// DIVERGENCE: Aqua reuses Crest's trilinear WaveNormals at 16x frequency and
// its shipped 0.08 strength because minification makes it weaker than an FFT
// normal field; rotation avoids aligning its period with the authored layers.
fn capillary_normal_slope(world_xz: vec2<f32>, ripple: f32) -> vec2<f32> {
    let cascade = cascade_layout.cascades[0u];
    let stretch = surface.detail.x * cascade.scale
        / (100.0 * surface.capillary.x);
    let speed = pow(
        log(1.0 + 2.0 * cascade.texel_width) * NORMAL_SCROLL_MULTIPLIER,
        NORMAL_SCROLL_POWER,
    );
    let scrolled = advected_world(world_xz)
        + CAPILLARY_DIRECTION * globals.time * speed;
    let uv = CAPILLARY_ROTATION * scrolled / stretch;
    let lod = screen_texture_lod(stretch, textureDimensions(detail_normal).x);
    return ripple * surface.capillary.y * sample_detail_normal(uv, lod).xy;
}
