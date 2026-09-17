//! CPU regression checks plus explicit WGSL parity contracts. These tests do not
//! execute WGSL; a composed-shader GPU smoke test is still required.

const INCIDENT: &str = include_str!("incident.wgsl");
const MATERIAL: &str = include_str!("../../bevy-aqua-core/src/cascade/material.wgsl");
const OPTICS: &str = include_str!("../../bevy-aqua-optics/src/optics.wgsl");

fn compact(source: &str) -> String {
    source.chars().filter(|c| !c.is_whitespace()).collect()
}

// Deliberate f32 mirror of the shader, pinned by shader_lambda_matches_cpu_model.
fn lambda(cos_theta: f32, alpha: f32) -> f32 {
    let cosine = cos_theta.clamp(1e-5, 1.0);
    let tangent_squared = (1.0 - cosine * cosine).max(0.0) / (cosine * cosine);
    0.5 * ((1.0 + alpha * alpha * tangent_squared).sqrt() - 1.0)
}

#[test]
fn shader_lambda_matches_cpu_model() {
    let signature = "fn smith_masking_shadowing(";
    let start = INCIDENT.find(signature).expect("Smith helper must exist");
    let end = start + INCIDENT[start..].find('}').expect("helper body must end") + 1;
    assert_eq!(
        compact(&INCIDENT[start..end]),
        compact(
            "fn smith_masking_shadowing(cos_theta: f32, alpha: f32) -> f32 {
                let cosine = clamp(cos_theta, 1e-5, 1.0);
                let tangent_squared = max(1.0 - cosine * cosine, 0.0) / (cosine * cosine);
                return 0.5 * (sqrt(1.0 + alpha * alpha * tangent_squared) - 1.0);
            }"
        ),
        "Update the CPU model and its independent checks when changing the WGSL",
    );
}

#[test]
fn lambda_matches_independent_ggx_projected_area_reference() {
    // GGX G1 from the projected-area form, evaluated in f64, without tan(theta)
    // or subtraction of nearly equal numbers. Compare G1 rather than small Lambda.
    for cosine in [1e-5_f32, 0.001, 0.05, 0.25, 0.5, 0.9, 1.0] {
        for alpha in [0.0_f32, 0.001, 0.02, 0.1, 0.5, 1.0] {
            let c = f64::from(cosine);
            let a = f64::from(alpha);
            let projected_area = (a * a + (1.0 - a * a) * c * c).sqrt();
            let reference_g1 = 2.0 * c / (c + projected_area);
            let actual_g1 = f64::from(1.0 / (1.0 + lambda(cosine, alpha)));
            assert!(
                (actual_g1 - reference_g1).abs() < 2e-6,
                "cosine={cosine} alpha={alpha}: {actual_g1} != {reference_g1}",
            );
        }
    }
    // Fixed nontrivial reference: tan²(theta)=3, alpha=1 gives Lambda=0.5.
    assert!((lambda(0.5, 1.0) - 0.5).abs() < 1e-6);
}

#[test]
fn lambda_limits_and_monotonicity() {
    for alpha in [0.0, 0.02, 0.1, 0.5, 1.0] {
        assert_eq!(lambda(1.0, alpha), 0.0);
        let mut previous = f32::INFINITY;
        for cosine in [0.0, 1e-5, 0.001, 0.1, 0.5, 1.0] {
            let value = lambda(cosine, alpha);
            assert!(value.is_finite() && value >= 0.0 && value <= previous);
            previous = value;
        }
    }
    for cosine in [0.0, 1e-5, 0.1, 0.5, 1.0] {
        assert_eq!(lambda(cosine, 0.0), 0.0);
        let mut previous = 0.0;
        for alpha in [0.0, 0.02, 0.1, 0.5, 1.0] {
            let value = lambda(cosine, alpha);
            assert!(value >= previous);
            previous = value;
        }
    }
    assert_eq!(lambda(0.0, 0.5), lambda(1e-5, 0.5));
    assert_eq!(lambda(1.1, 0.5), 0.0);
}

#[test]
fn all_nine_shader_calls_pass_cosine_before_roughness() {
    for (source, expected) in [
        (
            INCIDENT,
            [
                "let light_mask = smith_masking_shadowing(dot_nv, invocation_sun_roughness());",
                "let light_mask = smith_masking_shadowing(dot_nv, sun_roughness);",
                "let view_mask = smith_masking_shadowing(max(dot_nl, 2e-5), sun_roughness);",
            ],
        ),
        (
            MATERIAL,
            [
                "let sss_light_mask = smith_masking_shadowing(dot_nv, invocation_sun_roughness());",
                "let light_mask = smith_masking_shadowing(dot_nv, body_lighting.sun_roughness);",
                "let view_mask = smith_masking_shadowing(dot_nl, body_lighting.sun_roughness);",
            ],
        ),
        (
            OPTICS,
            [
                "let sss_light_mask = smith_masking_shadowing(dot_nv, invocation_sun_roughness());",
                "let light_mask = smith_masking_shadowing(dot_nv_sun, sun_roughness);",
                "let view_mask_sun = smith_masking_shadowing(dot_nl, sun_roughness);",
            ],
        ),
    ] {
        let actual: Vec<_> = source
            .lines()
            .filter(|line| {
                line.contains("smith_masking_shadowing(") && !line.trim_start().starts_with("fn ")
            })
            .map(compact)
            .collect();
        let expected: Vec<_> = expected.into_iter().map(compact).collect();
        assert_eq!(actual, expected);
    }
}
