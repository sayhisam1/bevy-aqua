//! CPU limit checks and source contracts, not WGSL execution.
//! Runtime shader compilation and reflection-source captures remain required.

use crate::test_support::{compact_wgsl as compact, compact_wgsl_function as function};

const COMMON: &str = include_str!("../../bevy-aqua-core/src/cascade/common.wgsl");
const MATERIAL: &str = include_str!("../../bevy-aqua-core/src/cascade/material.wgsl");
const OPTICS: &str = include_str!("optics.wgsl");

fn assert_contains(source: &str, fragment: &str) {
    assert!(
        source.contains(&compact(fragment)),
        "WGSL contract changed: {fragment}"
    );
}

// Deliberate f32 mirror, tied to production by the complete helper contract.
fn fresnel(cosine: f32, f0: f32, exponent: f32) -> f32 {
    let grazing = (1.0 - cosine).clamp(0.0, 1.0).powf(exponent);
    grazing * (1.0 - f0) + f0
}

#[test]
fn dielectric_helper_has_no_reflection_source_or_body_state_dependency() {
    assert_eq!(
        function(COMMON, "godot_fresnel"),
        compact(
            "fn godot_fresnel(view_alignment: f32) -> f32 {
            let grazing = pow(clamp(1.0 - view_alignment, 0.0, 1.0), surface.fresnel.y);
            return mix(grazing, 1.0, surface.fresnel.x);
        }"
        ),
        "Fresnel must not depend on planar availability, body flags, or roughness",
    );
}

#[test]
fn schlick_reaches_unity_at_grazing_and_matches_fixed_reference_values() {
    // Independent hand-computed fifth-power references for F0 = 0.02.
    // cos=0.5 gives 0.02 + 0.98/32; cos=0.75 gives 0.02 + 0.98/1024.
    for (cosine, expected) in [
        (-0.2, 1.0),
        (0.0, 1.0),
        (0.5, 0.050625),
        (0.75, 0.02095703125),
        (1.0, 0.02),
        (1.2, 0.02),
    ] {
        assert!((fresnel(cosine, 0.02, 5.0) - expected).abs() < 1e-7);
    }
    for f0 in [0.0, 0.02, 0.04, 1.0] {
        for exponent in [1.0, 3.0, 5.0, 8.0] {
            assert_eq!(fresnel(0.0, f0, exponent), 1.0);
            assert_eq!(fresnel(1.0, f0, exponent), f0);
            let mut previous = 1.0;
            for step in 0..=100 {
                let value = fresnel(step as f32 / 100.0, f0, exponent);
                assert!(value >= f0 && value <= previous);
                previous = value;
            }
        }
    }
}

#[test]
fn near_far_and_reflection_fraction_share_the_dielectric_weight() {
    let near = function(MATERIAL, "shade_water_body");
    assert_contains(&near, "let fresnel = godot_fresnel(view_alignment);");
    let compose = function(MATERIAL, "compose_water");
    assert_contains(
        &compose,
        "let reflection_weight = clamp(body_lighting.fresnel * surface.fresnel.z, 0.0, 1.0);",
    );
    assert_contains(
        &compose,
        "if mode == DEBUG_MODE_REFLECTION_FRACTION { return vec4(vec3(reflection_weight), 1.0); }",
    );
    assert_contains(
        &compose,
        "var water = mix(local.body, reflected_radiance, reflection_weight);",
    );
    let far = function(OPTICS, "far_field_water");
    assert_contains(
        &far,
        "let reflection_weight = clamp(godot_fresnel(view_alignment) * surface.fresnel.z, 0.0, 1.0,);",
    );
    assert_contains(
        &far,
        "return mix(body, reflected_radiance, reflection_weight);",
    );
}

#[test]
fn body_roughness_requires_enabled_bounded_optics_and_nonnegative_override() {
    // Full selector contract protects both inactive flags, the negative inherit
    // sentinel, and the positive floor required by direct-light GGX/Smith.
    assert_eq!(
        function(COMMON, "invocation_sun_roughness"),
        compact(
            "fn invocation_sun_roughness() -> f32 {
            let body_active = invocation_bounded > 0.5 && invocation_optics_a.w > 0.5;
            return clamp(select(surface.sun.y, invocation_optics_b.y,
                body_active && invocation_optics_b.y >= 0.0,
            ), 0.001, 1.0);
        }"
        ),
    );
}

#[test]
fn near_far_and_both_local_light_loops_keep_authored_lobe_roughness() {
    let near = function(MATERIAL, "shade_water_body");
    assert_contains(&near, "var sun_roughness = invocation_sun_roughness();");
    assert_contains(
        &near,
        "let foam_surface_roughness = clamp(invocation_sun_roughness() + foam_roughness, invocation_sun_roughness(), 1.0,);",
    );
    assert_contains(
        &near,
        "sun_roughness = min(sqrt(foam_surface_roughness * foam_surface_roughness + perceptual_roughness * perceptual_roughness,), 1.0);",
    );
    assert_contains(
        &near,
        "fresnel, foam_roughness, environment_roughness, sun_roughness,",
    );
    let sun = function(MATERIAL, "shade_environment_and_sun");
    assert_contains(
        &sun,
        "let distribution = ggx_distribution(clamp(dot(near.lighting_normal, halfway), 0.0, 1.0), body_lighting.sun_roughness,);",
    );
    let far = function(OPTICS, "far_field_water");
    assert_contains(
        &far,
        "let sun_roughness = min(sqrt(invocation_sun_roughness() * invocation_sun_roughness() + perceptual_roughness * perceptual_roughness,), 1.0);",
    );
    assert_contains(
        &far,
        "let distribution = ggx_distribution(clamp(dot(lighting_normal, halfway), 0.0, 1.0), sun_roughness,);",
    );
    let local = function(MATERIAL, "shade_local_lights");
    let call = compact(
        "let contribution = local_light_contribution(sample, near.lighting_normal, to_view, in.sample_data.z, body_lighting.sun_roughness, local_grazing, local_crest_transmission, local_sss_enabled,);",
    );
    assert_eq!(
        local.matches(&call).count(),
        2,
        "Point and spot loops must both use body roughness"
    );
    // SSS and all nine Smith argument contracts live in the light crate tests.
    for source in [near, sun, far, local] {
        assert!(
            !source.contains("surface.sun.y"),
            "Do not bypass the body selector"
        );
    }
}
