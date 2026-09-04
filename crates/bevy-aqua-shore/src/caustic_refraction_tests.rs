//! Scalar reference tests; these do not execute WGSL or validate bed projection.
const ETA: f32 = 1.0 / 1.333;

fn path(depth: f32, cos_air: f32) -> f32 {
    let c = cos_air.clamp(0.0, 1.0);
    depth / (1.0 - ETA * ETA * (1.0 - c * c)).sqrt()
}

#[test]
fn shader_keeps_reference_equation_and_air_incidence_separate() {
    let s = include_str!("water.wgsl");
    assert!(s.contains("const CAUSTIC_AIR_TO_WATER_ETA: f32 = 1.0 / 1.333;"));
    assert!(s.contains("let cos_incident = clamp(bed_incidence, 0.0, 1.0);"));
    assert!(s.contains("let cos_transmitted = sqrt(1.0 - CAUSTIC_AIR_TO_WATER_ETA\n        * CAUSTIC_AIR_TO_WATER_ETA * (1.0 - cos_incident * cos_incident));"));
    assert!(s.contains("let incoming_path = water_depth / cos_transmitted;"));
    assert!(s.contains("* bed_incidence * sun_shadow;"));
    assert!(
        s.contains("water_depth >= maximum_depth || sun_direction.y <= 0.0 || sun_shadow <= 0.0")
    );
}

#[test]
fn transmitted_path_matches_independent_angle_reference() {
    for c in [0.0_f32, 0.001, 0.02, 0.1, 0.5, 0.9, 1.0] {
        let theta_air = (c as f64).acos();
        let theta_water = (theta_air.sin() / 1.333_f64).asin();
        let reference = 2.0 / theta_water.cos();
        assert!((path(2.0, c) as f64 - reference).abs() < 1e-6);
    }
}

#[test]
fn normal_incidence_and_zero_depth_are_unchanged() {
    assert_eq!(path(2.0, 1.0), 2.0);
    for c in [0.0, 0.02, 0.5, 1.0] {
        assert_eq!(path(0.0, c), 0.0);
    }
}

#[test]
fn grazing_path_is_finite_monotonic_and_bounded() {
    let limit = 2.0 / (1.0 - ETA * ETA).sqrt();
    let mut previous = limit;
    for i in 0..=1000 {
        let d = path(2.0, i as f32 / 1000.0);
        assert!(d.is_finite() && d >= 2.0 && d <= limit);
        assert!(d <= previous);
        previous = d;
    }
    assert!(limit > 3.02 && limit < 3.03);
}

#[test]
fn two_meter_grazing_bed_loses_spurious_air_path_extinction() {
    let c = 0.02_f32;
    let old_path = 2.0 / c.max(0.02);
    let new_path = path(2.0, c);
    assert_eq!(old_path, 100.0);
    assert!(new_path > 3.02 && new_path < 3.03);
    for sigma in [0.0_f32, 0.3, 0.7, 1.0] {
        let old_transmission = (-sigma * old_path).exp();
        let new_transmission = (-sigma * new_path).exp();
        assert!(new_transmission >= old_transmission && new_transmission <= 1.0);
        // Incidence remains the AIR projected-area factor, not cos(theta_water).
        let weighted = c * new_transmission;
        assert!(weighted >= 0.0 && weighted <= c);
    }
}
