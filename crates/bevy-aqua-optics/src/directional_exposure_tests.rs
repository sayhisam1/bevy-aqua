//! Source-level contracts for directional specular radiance routing.
//! These inspect production WGSL, not a CPU copy of its lighting arithmetic.
//! Shader execution and visual exposure sweeps remain separate validation.

use crate::test_support::{compact_wgsl as compact, compact_wgsl_function as function};

const MATERIAL: &str = include_str!("../../bevy-aqua-core/src/cascade/material.wgsl");
const INCIDENT: &str = include_str!("../../bevy-aqua-light/src/incident.wgsl");
const OPTICS: &str = include_str!("optics.wgsl");

fn assert_contains(source: &str, fragment: &str) {
    assert!(
        source.contains(&compact(fragment)),
        "Directional radiance contract changed: {fragment}"
    );
}

#[test]
fn near_sun_consumes_the_shared_exposed_shadowed_primary_radiance() {
    let resolve = function(INCIDENT, "resolve_primary_light");
    assert_contains(
        &resolve,
        "primary_light_color = filtered_primary_light_color(in.world_position, light.direction_to_light, light.sun_disk_angular_size, light.color.rgb,);",
    );
    assert_contains(
        &resolve,
        "let primary_light_radiance = primary_light_color * view.exposure * primary_light_shadow;",
    );
    assert_contains(
        &resolve,
        "return PrimaryLightState(view_z, ranges.first_point_light_index_offset, ranges.first_spot_light_index_offset, ranges.first_reflection_probe_index_offset, primary_light_shadow, primary_light_color, primary_light_radiance,);",
    );

    let shade = function(MATERIAL, "shade_environment_and_sun");
    assert_contains(
        &shade,
        "reflected_radiance += sun_specular * surface.sun.x * primary.radiance;",
    );
    // The shared radiance already includes both factors. Applying either here
    // again would square exposure or shadows; raw color would bypass them.
    for forbidden in [
        "primary.color",
        "primary.shadow",
        "view.exposure",
        "surface.reflection.z",
    ] {
        assert!(
            !shade.contains(forbidden),
            "Unexpected near factor: {forbidden}"
        );
    }
}

#[test]
fn far_sun_consumes_filtered_exposed_radiance_without_lux_normalization() {
    let far = function(OPTICS, "far_field_water");
    assert_contains(
        &far,
        "let filtered_light_color = filtered_primary_light_color(in.world_position, light.direction_to_light, light.sun_disk_angular_size, light.color.rgb,);",
    );
    assert_contains(
        &far,
        "let light_radiance = filtered_light_color * view.exposure;",
    );
    assert_contains(
        &far,
        "reflected_radiance += sun_specular * surface.sun.x * light_radiance;",
    );
    // Far shading intentionally omits near shadow sampling, but must retain
    // atmospheric filtering and exactly one exposure conversion.
    assert_eq!(far.matches("view.exposure").count(), 1);
    for forbidden in [
        "surface.reflection.z",
        "light_luminance",
        "light_strength",
        "fetch_directional_shadow",
    ] {
        assert!(
            !far.contains(forbidden),
            "Unexpected far factor: {forbidden}"
        );
    }
}
