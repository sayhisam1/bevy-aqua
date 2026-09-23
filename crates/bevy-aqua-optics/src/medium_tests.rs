use std::f64::consts::PI;

fn fresnel(n1: f64, n2: f64, cos_i: f64) -> f64 {
    let eta = n1 / n2;
    let sin2_t = eta * eta * (1.0 - cos_i * cos_i);
    if sin2_t >= 1.0 {
        return 1.0;
    }
    let cos_t = (1.0 - sin2_t).sqrt();
    let rs = (n1 * cos_i - n2 * cos_t) / (n1 * cos_i + n2 * cos_t);
    let rp = (n2 * cos_i - n1 * cos_t) / (n2 * cos_i + n1 * cos_t);
    0.5 * (rs * rs + rp * rp)
}

#[test]
fn exact_fresnel_normal_critical_and_grazing() {
    let n = 1.333_f64;
    let f0 = ((n - 1.0) / (n + 1.0)).powi(2);
    assert!((fresnel(n, 1.0, 1.0) - f0).abs() < 1e-12);
    let critical = (1.0 / n).asin();
    assert!((critical * 180.0 / PI - 48.62).abs() < 0.05);
    assert_eq!(fresnel(n, 1.0, (critical + 1e-4).cos()), 1.0);
    assert!(fresnel(n, 1.0, 1e-6) > 0.999_98);
}

fn analytic(sigma: f64, t: f64, rd_y: f64, l_y: f64, d0: f64) -> f64 {
    let ly = l_y.max(0.02);
    let i0 = (-sigma * d0 / ly).exp();
    let k = sigma * (1.0 - rd_y / ly);
    if k.abs() <= 1e-5 {
        i0 * t
    } else {
        i0 * (1.0 - (-k * t).exp()) / k
    }
}

fn midpoint(sigma: f64, t: f64, rd_y: f64, l_y: f64, d0: f64) -> f64 {
    let n = 200_000;
    let ds = t / n as f64;
    (0..n)
        .map(|i| {
            let s = (i as f64 + 0.5) * ds;
            (-sigma * ((d0 - s * rd_y) / l_y.max(0.02) + s)).exp() * ds
        })
        .sum()
}

#[test]
fn closed_form_matches_quadrature_and_series_limit() {
    for (sigma, t, rd_y, l_y, d0) in [
        (0.05, 20.0, -0.7, 0.8, 3.0),
        (0.3, 4.0, 0.2, 0.6, 2.0),
        (0.1, 8.0, 0.500_001, 0.5, 5.0),
    ] {
        let a = analytic(sigma, t, rd_y, l_y, d0);
        let q = midpoint(sigma, t, rd_y, l_y, d0);
        assert!((a - q).abs() <= 2e-5 * q.abs().max(1.0), "{a} != {q}");
    }
}

#[test]
fn shader_keeps_safety_and_energy_contracts() {
    let source = include_str!("medium.wgsl");
    assert!(source.contains("clamp(g, -0.99, 0.99)"));
    assert!(source.contains("let sigma_s = min(safe_sigma_t, sigma_p + RAYLEIGH);"));
    assert!(source.contains("if t_end < 1e-4"));
    assert!(source.contains("min(max(t_end, 0.0), PATH_LENGTH_MAX)"));
}

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
fn refract_oriented(l: [f64; 3], up: [f64; 3]) -> [f64; 3] {
    let l = norm3(l);
    let up = norm3(up);
    let c = dot3(l, up).clamp(0.0, 1.0);
    let tangent = sub3(l, scale3(up, c));
    let length = dot3(tangent, tangent).sqrt();
    let sin_t = length / 1.333;
    let tangent_dir = if length > 1e-12 {
        scale3(tangent, 1.0 / length)
    } else {
        [0.0; 3]
    };
    add3(
        scale3(tangent_dir, sin_t),
        scale3(up, (1.0 - sin_t * sin_t).sqrt()),
    )
}
fn rotate_x(v: [f64; 3], angle: f64) -> [f64; 3] {
    let (s, c) = angle.sin_cos();
    [v[0], c * v[1] - s * v[2], s * v[1] + c * v[2]]
}

#[test]
fn oriented_snell_y_wrapper_and_rotation_covariance() {
    let grazing = refract_oriented([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
    assert!((grazing[1] - 0.661_225_108_791).abs() < 1e-12);
    let normal = refract_oriented([0.0, 1.0, 0.0], [0.0, 1.0, 0.0]);
    assert!(dot3(sub3(normal, [0.0, 1.0, 0.0]), sub3(normal, [0.0, 1.0, 0.0])) < 1e-24);
    let l = norm3([0.4, 0.8, -0.2]);
    let angle = 0.73;
    let rotated = refract_oriented(rotate_x(l, angle), rotate_x([0.0, 1.0, 0.0], angle));
    let expected = rotate_x(refract_oriented(l, [0.0, 1.0, 0.0]), angle);
    assert!(dot3(sub3(rotated, expected), sub3(rotated, expected)) < 1e-24);
}

#[test]
fn tilted_local_medium_is_continuous_across_world_y_zero() {
    let up = norm3([0.0, 1.0, 1.0]);
    let tangent = norm3([0.0, 1.0, -1.0]);
    let values: Vec<f64> = [-1e-6, 0.0, 1e-6]
        .into_iter()
        .map(|world_y_delta| {
            // World Y crosses zero while the local-medium projection changes smoothly.
            let rd = norm3(add3(scale3(tangent, 0.8), scale3(up, -0.8 + world_y_delta)));
            analytic(0.2, 32.0, dot3(rd, up), 0.7, 0.0)
        })
        .collect();
    assert!(values.iter().all(|v| v.is_finite() && *v >= 0.0));
    assert!((values[2] - values[0]).abs() < 1e-4);
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

#[test]
fn oriented_shader_keeps_world_y_wrapper_and_local_sun_horizon() {
    let source = include_str!("medium.wgsl");
    assert!(source.contains("return medium_radiance_oriented("));
    assert!(source.contains("vec3(0.0, 1.0, 0.0),"));
    assert!(source.contains("let cos_air = clamp(dot(l_air, up), 0.0, 1.0);"));
    assert!(source.contains("if cos_air <= 0.0"));
    assert!(source.contains("dot(l_water, up),"));
    assert!(source.contains("let rd_up = dot(rd, up);"));
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
