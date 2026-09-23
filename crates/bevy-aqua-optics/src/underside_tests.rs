#[test]
fn underside_uses_current_near_surface_variance_contract() {
    let source = include_str!("optics.wgsl");
    let underside = source.split("fn shade_underside(").nth(1).unwrap();
    assert!(underside.contains("near.near_detail_weight"));
    assert!(underside.contains("near.filtered_detail_variance"));
    assert!(!source.contains("lighting_normal_strength"));
    assert!(underside.contains("camera_depth_path(in)"));
    assert!(underside.contains("camera_depth_debug_from_path("));
    assert!(underside.contains("viewport_origin"));
}

#[test]
fn underside_tir_uses_bounded_water_side_medium_not_air_probe() {
    let source = include_str!("optics.wgsl");
    // `underside_reflection` precedes and is called by `shade_underside`.
    let underside = source.split("fn underside_reflection(").nth(1).unwrap();
    assert!(!underside.contains("bounce_path"));
    assert!(underside.contains("let open = medium_radiance_oriented("));
    assert!(underside.contains("PATH_LENGTH_MAX"));
    assert!(underside.contains("-water_normal"));
    assert!(underside.contains("invocation_scatter_scale()"));
    assert!(underside.contains("invocation_scatter_tint()"));
    assert!(underside.contains("invocation_scattering_asymmetry()"));
    let before_tir = underside
        .split("if dot(transmitted_direction")
        .next()
        .unwrap();
    assert!(!before_tir.contains("sample_environment(\n        reflected_direction"));
}

fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a.into_iter().zip(b).map(|(x, y)| x * y).sum()
}
fn scale3(v: [f64; 3], s: f64) -> [f64; 3] {
    [v[0] * s, v[1] * s, v[2] * s]
}
fn add3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
fn sub3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    add3(a, scale3(b, -1.0))
}
fn norm3(v: [f64; 3]) -> [f64; 3] {
    scale3(v, 1.0 / dot3(v, v).sqrt())
}
#[test]
fn reflected_ray_stays_in_local_water_halfspace() {
    for i in 0..128 {
        let a = (i as f64 + 0.5) / 128.0 * std::f64::consts::TAU;
        let up = norm3([0.35 * a.cos(), 1.0, 0.35 * a.sin()]);
        let base = [a.cos(), 0.0, a.sin()];
        let tangent = norm3(sub3(base, scale3(up, dot3(base, up))));
        let cos_i = 0.05 + 0.9 * (0.5 + 0.5 * a.sin());
        let incident = add3(
            scale3(up, cos_i),
            scale3(tangent, (1.0 - cos_i * cos_i).sqrt()),
        );
        let water_normal = scale3(up, -1.0);
        let reflected = sub3(
            incident,
            scale3(water_normal, 2.0 * dot3(incident, water_normal)),
        );
        assert!(dot3(reflected, up) <= 1e-12);
    }
}

fn visible_facet_up(incident: [f64; 3], candidate: [f64; 3]) -> [f64; 3] {
    let incident = norm3(incident);
    let candidate = norm3(candidate);
    let c = dot3(incident, candidate);
    norm3(add3(candidate, scale3(incident, (1e-3 - c).max(0.0))))
}

#[test]
fn shading_normal_visibility_clamp_crosses_continuously() {
    let incident = norm3([0.0, -0.2, 1.0]);
    let tangent = norm3([1.0, 0.0, 0.0]);
    let mut previous: Option<[f64; 3]> = None;
    for c in [-1.0, -0.5, -1e-3, 0.0, 0.999e-3, 1.001e-3, 0.5] {
        let candidate = add3(scale3(incident, c), scale3(tangent, (1.0 - c * c).sqrt()));
        let up = visible_facet_up(incident, candidate);
        assert!(dot3(incident, up) > 0.0);
        let water_normal = scale3(up, -1.0);
        let reflected = sub3(
            incident,
            scale3(water_normal, 2.0 * dot3(incident, water_normal)),
        );
        assert!(dot3(reflected, up) < 0.0);
        assert!(up.into_iter().all(f64::is_finite));
        if (c - 1e-3).abs() < 2e-6 {
            if let Some(prior) = previous {
                assert!(dot3(sub3(up, prior), sub3(up, prior)).sqrt() < 1e-3);
            }
        }
        previous = Some(up);
    }
}

#[test]
fn random_grazing_microdetail_is_bounded_to_water_halfspace() {
    let incident = norm3([0.0, -0.01, 1.0]);
    for i in 0..256 {
        let a = (i as f64 + 0.5) * std::f64::consts::TAU / 256.0;
        let candidate = norm3([a.cos(), 0.15 * a.sin(), -0.8 + 1.6 * (i as f64 / 255.0)]);
        let up = visible_facet_up(incident, candidate);
        let n = scale3(up, -1.0);
        let reflected = sub3(incident, scale3(n, 2.0 * dot3(incident, n)));
        assert!(dot3(incident, up) > 0.0);
        assert!(dot3(reflected, up) < 0.0);
        assert!(reflected.into_iter().all(f64::is_finite));
    }
}

#[test]
fn underside_shader_applies_visibility_guard_before_interface_math() {
    let source = include_str!("optics.wgsl");
    let underside = source.split("fn underside_reflection(").nth(1).unwrap();
    assert!(underside.contains("let corrected_up = candidate_up"));
    assert!(underside.contains("max(1e-3 - visibility, 0.0)"));
    assert!(underside.contains("let facet_up = safe_normalize(corrected_up, incident)"));
    assert!(underside.contains("let water_normal = -facet_up"));
    assert!(underside.contains("invocation_scattering_asymmetry(),\n        facet_up,"));
}

#[test]
fn visibility_clamp_preserves_shared_geometric_hemisphere() {
    let geometric_up = norm3([0.2, 1.0, -0.1]);
    let incident = norm3([0.1, 0.4, 1.0]);
    assert!(dot3(incident, geometric_up) > 0.0);
    for i in 0..128 {
        let a = (i as f64 + 0.5) * std::f64::consts::TAU / 128.0;
        let raw = norm3([a.cos(), 0.05 + a.sin().abs(), a.sin()]);
        let candidate = if dot3(raw, geometric_up) > 0.0 {
            raw
        } else {
            scale3(raw, -1.0)
        };
        let corrected = visible_facet_up(incident, candidate);
        assert!(dot3(candidate, geometric_up) > 0.0);
        assert!(dot3(corrected, geometric_up) > 0.0);
        assert!(dot3(corrected, incident) > 0.0);
    }
}
