//! CPU reference + source-contract guards; these do not execute WGSL on a GPU.
const OPTICS: &str = include_str!("optics.wgsl");
const MATERIAL: &str = include_str!("../../bevy-aqua-core/src/cascade/material.wgsl");

fn function<'a>(source: &'a str, name: &str) -> &'a str {
    let start = source.find(&format!("fn {name}(")).unwrap();
    let tail = &source[start..];
    &tail[..tail.find("\n}\n").unwrap() + 3]
}
fn ordered(source: &str, parts: &[&str]) {
    let mut tail = source;
    for part in parts {
        let index = tail
            .find(part)
            .unwrap_or_else(|| panic!("missing ordered clause: {part}"));
        tail = &tail[index + part.len()..];
    }
}
fn threshold() -> f32 {
    OPTICS
        .split("const TRANSMISSION_OPAQUE_OPTICAL_DEPTH: f32 = ")
        .nth(1)
        .unwrap()
        .split(';')
        .next()
        .unwrap()
        .parse()
        .unwrap()
}
fn minimum(extinction: [f32; 3]) -> f32 {
    extinction.into_iter().fold(f32::INFINITY, f32::min)
}
fn opaque(has_background: bool, raw: f32, refracted: f32, valid: bool, extinction: f32) -> bool {
    // Source guards below bind this reference's decisions to the actual shader.
    !has_background
        || raw <= 1e-6
        || extinction * if valid { refracted } else { raw } >= threshold()
}
#[test]
fn clear_water_thresholds_and_least_attenuated_channel() {
    assert!((threshold() - 10.0_f32 * 2.0_f32.ln()).abs() < 1e-6);
    let ext = minimum([0.5, 0.1, 0.3]);
    for (path, expected) in [(7.0, false), (24.0, false), (69.0, false), (70.0, true)] {
        assert_eq!(
            opaque(true, path, path, false, ext),
            expected,
            "path={path}"
        );
    }
    assert!(!opaque(true, 23.0, 0.0, false, 0.3));
    assert!(opaque(true, 24.0, 0.0, false, 0.3));
    assert!(opaque(true, threshold(), 0.0, false, 1.0));
    assert!((-ext * 70.0).exp() <= 1.0 / 1024.0);
}
#[test]
fn opacity_gate_uses_authored_extinction() {
    let gate = function(OPTICS, "far_path_opaque");
    let transmission = function(OPTICS, "beauty_transmission");
    assert!(!OPTICS.contains("fn beauty_extinction("));
    assert!(!OPTICS.contains("0.52, 0.42, 0.62"));
    assert!(!OPTICS.contains("fn depth_aware_body_albedo("));
    for source in [gate, transmission] {
        assert!(source.contains("let extinction = invocation_extinction();"));
        assert!(source.contains("min(extinction.r, min(extinction.g, extinction.b))"));
        assert!(!source.contains("deep_water_weight"));
    }
    assert!(
        function(OPTICS, "deep_water_weight")
            .contains("smoothstep(SHALLOW_WATER_DEPTH, DEEP_WATER_DEPTH, water_depth)")
    );
    assert!(MATERIAL.contains("far_tier *= deep_water_weight(far_water_depth);"));
}
#[test]
fn accepted_refraction_not_raw_depth_controls_gate() {
    assert!(!opaque(true, 100.0, 7.0, true, 0.1));
    assert!(opaque(true, 7.0, 100.0, true, 0.1));
    assert!(opaque(true, 100.0, 7.0, false, 0.1));
    assert!(!opaque(true, 7.0, 100.0, false, 0.1));
    ordered(
        function(OPTICS, "far_path_opaque"),
        &[
            "camera_depth_debug_from_path(in, normal, path)",
            "accepted.path_length,",
            "accepted.refracted_path_length,",
            "accepted.refracted_sample_valid,",
            "minimum_extinction * water_path >= TRANSMISSION_OPAQUE_OPTICAL_DEPTH",
        ],
    );
    let beauty = function(OPTICS, "beauty_transmission");
    ordered(
        beauty,
        &[
            "camera_depth_debug_from_path(in, normal, depth_path)",
            "let use_refraction = depth_debug.refracted_sample_valid;",
            "depth_debug.path_length,",
            "depth_debug.refracted_path_length,",
            "use_refraction,",
            "minimum_extinction * water_path < TRANSMISSION_OPAQUE_OPTICAL_DEPTH",
            "opaque_background(background_uv)",
        ],
    );
    assert!(
        function(OPTICS, "camera_depth_debug_from_path")
            .contains("refracted_raw_depth > 0.0\n        && refracted_raw_depth < in.position.z")
    );
}
#[test]
fn no_background_and_zero_path_preserve_existing_contract() {
    assert!(opaque(false, 7.0, 100.0, true, 0.1));
    assert!(opaque(true, 0.0, 100.0, true, 0.1));
    ordered(
        function(OPTICS, "far_path_opaque"),
        &[
            "if !path.has_background || path.path_length <= LUMINANCE_EPSILON {",
            "return true;",
            "camera_depth_debug_from_path(",
        ],
    );
    let beauty = function(OPTICS, "beauty_transmission");
    ordered(
        beauty,
        &[
            "if !(depth_path.has_background && depth_path.path_length > LUMINANCE_EPSILON) {",
            "return surface_medium_radiance(vec3(0.0), to_view, PATH_LENGTH_MAX);",
            "camera_depth_debug_from_path(",
            "opaque_background(",
        ],
    );
    assert!(function(OPTICS, "empty_camera_depth_path").contains("result.has_background = false;"));
}
#[test]
fn raw_depth_cache_survives_foam_and_rejection_reconstructs_near_normal() {
    let fragment = function(MATERIAL, "fragment");
    ordered(
        fragment,
        &[
            "var near = resolve_near_surface(in, surface_lod, shading_normal, far_tier, mode);",
            "shared_depth_path = camera_depth_path(in);",
            "has_shared_depth_path = true;",
            "if !far_path_opaque(in, near.normal, shared_depth_path) {",
            "far_tier = 0.0;",
            "near = resolve_near_surface(in, surface_lod, shading_normal, far_tier, mode);",
            "far_water = far_field_water(",
            "if far_tier >= 1.0 {",
            "return vec4(far_water, 1.0);",
            "let primary = resolve_primary_light(in, near.normal);",
            "let foam = prepare_surface_foam(",
            "shared_depth_path,",
            "has_shared_depth_path,",
            "resolve_transmission(in, near.normal,",
        ],
    );
    ordered(
        function(MATERIAL, "prepare_surface_foam"),
        &[
            "var shared_depth_path = cached_depth_path;",
            "var has_shared_depth_path = has_cached_depth_path;",
            "if !has_shared_depth_path {",
            "shared_depth_path = camera_depth_path(in);",
            "return FoamState(",
            "shared_depth_path,",
            "has_shared_depth_path,",
        ],
    );
    ordered(
        function(OPTICS, "resolve_transmission"),
        &[
            "var shared_depth_path = foam.depth_path;",
            "if !foam.has_depth_path {",
            "shared_depth_path = camera_depth_path(in);",
            "if mode == DEBUG_MODE_BEAUTY {",
            "beauty_transmission(",
            "in, normal, to_view, primary, shared_depth_path, source_slot,",
        ],
    );
    // Only raw depth is shared. Accepted refracted depth must be recomputed with the restored normal.
    assert_eq!(
        fragment
            .matches("shared_depth_path = camera_depth_path(in);")
            .count(),
        1
    );
}
#[test]
fn zero_near_weight_still_reconstructs_normal_including_negative_y() {
    let near = function(OPTICS, "resolve_near_surface");
    ordered(
        near,
        &[
            "if mode >= DEBUG_MODE_BEAUTY {",
            "let near_weight = 1.0 - far_tier;",
            "var resolved_slope = normal.xz / max(normal.y, MIN_NORMAL_Y);",
            "normal = safe_normalize(",
            "vec3(resolved_slope.x, 1.0, resolved_slope.y)",
        ],
    );
    // An outer near-weight guard around all reconstruction is NOT equivalent.
    assert!(!near.contains("if mode >= DEBUG_MODE_BEAUTY &&"));
    if near.contains("if near_weight != 0.0 {") {
        ordered(
            near,
            &[
                "if near_weight != 0.0 {",
                "detail_normal_sample(",
                "capillary_normal_slope(",
                "\n        }",
                "normal = safe_normalize(",
            ],
        );
    }
    let reconstruct = |n: [f32; 3]| {
        let y = n[1].max(0.01);
        let v = [n[0] / y, 1.0, n[2] / y];
        let len = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        v.map(|x| x / len)
    };
    for n in [
        [0.0, -1.0, 0.0],
        [0.6, -0.8, 0.0],
        [0.6, 0.8, 0.0],
        [1.0, 0.0, 0.0],
    ] {
        // Finite sampled detail/capillary values contribute exactly zero.
        let mut with_zero_detail = n;
        with_zero_detail[0] += 0.0 * 0.75;
        with_zero_detail[2] += 0.0 * -0.35;
        let reference = reconstruct(with_zero_detail);
        let skipped_detail = reconstruct(n);
        assert_eq!(reference, skipped_detail);
        assert!(reference[1] > 0.0);
    }
    assert_ne!(reconstruct([0.0, -1.0, 0.0]), [0.0, -1.0, 0.0]);
}
