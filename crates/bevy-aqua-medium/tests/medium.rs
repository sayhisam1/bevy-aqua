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
    let sigma = sigma.max(0.0);
    let ly = l_y.max(0.02);
    let i0 = (-sigma * d0 / ly).exp();
    let k = sigma * (1.0 - rd_y / ly);
    let x = (k * t).clamp(-80.0, 80.0);
    if x.abs() <= 1e-5 {
        i0 * t * (1.0 - x * 0.5 + x * x / 6.0)
    } else {
        i0 * (1.0 - (-x).exp()) / k
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
    let source = include_str!("../src/medium.wgsl");
    assert!(source.contains("clamp(g, -0.99, 0.99)"));
    assert!(source.contains("let sigma_s = min(safe_sigma_t, sigma_p + RAYLEIGH);"));
    assert!(source.contains("if t_end < 1e-4"));
    assert!(source.contains("min(max(t_end, 0.0), PATH_LENGTH_MAX)"));
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
fn oriented_shader_keeps_world_y_wrapper_and_local_sun_horizon() {
    let source = include_str!("../src/medium.wgsl");
    assert!(source.contains("return medium_radiance_oriented("));
    assert!(source.contains("vec3(0.0, 1.0, 0.0),"));
    assert!(source.contains("let cos_air = clamp(dot(l_air, up), 0.0, 1.0);"));
    assert!(source.contains("if cos_air <= 0.0"));
    assert!(source.contains("dot(l_water, up),"));
    assert!(source.contains("let rd_up = dot(rd, up);"));
}

#[test]
fn medium_segment_composition_and_interface_conversion() {
    for sigma in [0.0_f64, 0.001, 0.1, 0.9] {
        for rd_y in [-0.8_f64, 0.0, 0.2] {
            let d0 = 8.0;
            let t1 = 3.0;
            let t2 = 5.0;
            let coefficient = sigma.min(0.02193);
            let medium = |scene: f64, t: f64, depth: f64| {
                scene * (-sigma * t).exp() + coefficient * analytic(sigma, t, rd_y, 0.7, depth)
            };
            let terminal = 0.4;
            let combined = medium(terminal, t1 + t2, d0);
            let split = medium(medium(terminal, t2, d0 - t1 * rd_y), t1, d0);
            assert!((combined - split).abs() < 1e-10);
            assert_eq!(medium(terminal, 0.0, d0), terminal);
            let n = 1.333_f64;
            let f0 = fresnel(1.0, n, 1.0);
            let air = (1.0 - f0) * combined / n.powi(2);
            assert!((air * n.powi(2) / (1.0 - f0) - combined).abs() < 1e-12);
        }
    }
}

#[test]
fn particle_zero_retains_rayleigh_and_scattering_is_bounded_by_extinction() {
    for scale in [0.0_f32, 1.0, 10.0] {
        for extinction in [0.0_f32, 0.001, 0.3] {
            let particle = 0.02 * scale;
            let scattering = extinction.min(particle + 0.00193);
            assert!(scattering <= extinction);
            if extinction == 0.0 {
                assert_eq!(scattering, 0.0);
            }
            if scale == 0.0 && extinction == 0.3 {
                assert_eq!(scattering, 0.00193);
            }
        }
    }
    let medium = include_str!("../src/medium.wgsl");
    assert!(medium.contains("min(safe_sigma_t, sigma_p + RAYLEIGH)"));
}
