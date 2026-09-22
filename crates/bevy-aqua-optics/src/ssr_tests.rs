const OPTICS: &str = include_str!("optics.wgsl");
const MATERIAL: &str = include_str!("../../bevy-aqua-core/src/cascade/material.wgsl");

fn function_body<'a>(source: &'a str, name: &str) -> &'a str {
    let marker = format!("fn {name}(");
    source
        .split(&marker)
        .nth(1)
        .unwrap_or_else(|| panic!("{name} missing"))
}

#[test]
fn march_is_gated_and_fades_before_it_samples() {
    let march = function_body(OPTICS, "screen_space_reflection");
    assert!(march.contains("surface.far_tier.z < 0.5"));
    assert!(march.contains("smoothstep(fade_start, fade_end, roughness)"));
    let sample_at = march.find("opaque_background(").expect("hit color");
    let gate_at = march
        .find("surface.far_tier.z < 0.5")
        .expect("setting gate");
    assert!(gate_at < sample_at);
    assert!(OPTICS.contains("const SSR_THICKNESS: f32 = 0.4;"));
    assert!(OPTICS.contains("const SSR_BISECTION_STEPS: u32 = 5u;"));
    assert!(march.contains("hit.gap < SSR_THICKNESS"));
    assert!(march.contains("smoothstep(0.0, SSR_EDGE_FADE, inset)"));
}

#[test]
fn topside_replaces_the_environment_lobe_before_the_sun() {
    for (source, name) in [
        (OPTICS, "far_field_water"),
        (MATERIAL, "shade_environment_and_sun"),
    ] {
        let body = function_body(source, name);
        let planar = body.find("planar.weight").expect("planar mix");
        let ssr = body.find("screen_space_reflection(").expect("ssr");
        let sun = body
            .find("reflected_radiance += sun_specular")
            .expect("sun lobe");
        assert!(
            planar < ssr && ssr < sun,
            "{name} should march between planar and sun"
        );
        assert!(body.contains("vec3(0.0),"));
    }
}

#[test]
fn underside_attenuates_a_hit_along_the_bounce_only() {
    let underside = function_body(OPTICS, "shade_underside");
    assert!(underside.contains("smoothstep(0.15, 0.35, fresnel)"));
    assert!(underside.contains("facet_up,"));
    assert!(underside.contains("ssr.distance"));
    assert!(underside.contains("mix(reflected, hit_reflected, ssr.weight * fresnel_gate)"));
    assert!(!underside.contains("attenuate_underwater_scene"));
    let open = underside
        .find("let reflected = medium_radiance_oriented(")
        .expect("open medium");
    let hit = underside
        .find("let hit_reflected = medium_radiance_oriented(")
        .expect("hit medium");
    assert!(open < hit);
}
