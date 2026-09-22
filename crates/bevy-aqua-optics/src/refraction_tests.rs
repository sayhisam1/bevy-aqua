//! CPU geometry regressions and WGSL parity contracts for camera refraction.
//! These do not execute WGSL. Runtime shader compilation and visual checks
//! remain necessary.

use crate::test_support::{compact_wgsl as compact, compact_wgsl_function};
use bevy::math::{Mat4, Quat, Vec2, Vec3};

const OPTICS: &str = include_str!("optics.wgsl");

fn assert_shader_contains(fragment: &str) {
    let sources = compact(OPTICS) + &compact(include_str!("screen.wgsl"));
    assert!(
        sources.contains(&compact(fragment)),
        "WGSL contract changed: {fragment}"
    );
}

// Mirrors only the camera projection math, not the texture lookup.
fn projected_perturbation(clip_from_world: Mat4, normal: Vec3) -> Vec2 {
    let clip = clip_from_world * Vec3::new(normal.x, 0.0, normal.z).extend(0.0);
    Vec2::new(clip.x, -clip.y) * 0.5
}

fn reconstruct(view_from_clip: Mat4, uv: Vec2, depth: f32) -> Vec3 {
    let ndc = Vec3::new(2.0 * uv.x - 1.0, 1.0 - 2.0 * uv.y, depth);
    let position = view_from_clip * ndc.extend(1.0);
    position.truncate() / position.w
}

fn project(clip_from_view: Mat4, point: Vec3) -> (Vec2, f32) {
    let clip = clip_from_view * point.extend(1.0);
    let ndc = clip.truncate() / clip.w;
    (Vec2::new(0.5 * ndc.x + 0.5, 0.5 - 0.5 * ndc.y), ndc.z)
}

fn accepted_path(original: f32, water: Vec3, candidate: Option<Vec3>) -> f32 {
    candidate.map_or(original, |background| background.distance(water))
}

fn valid_refracted_depth(candidate: f32, water: f32) -> bool {
    candidate > 0.0 && candidate < water
}

#[test]
fn projection_follows_camera_yaw_roll_and_ignores_translation() {
    let projection = Mat4::perspective_infinite_reverse_rh(std::f32::consts::FRAC_PI_2, 2.0, 0.1);
    let yaw = Mat4::from_rotation_y(std::f32::consts::FRAC_PI_2);
    let roll = Mat4::from_rotation_z(std::f32::consts::FRAC_PI_2);
    // A 90-degree horizontal field scale of 1/2 yields 1/4 in UV units.
    assert!(projected_perturbation(projection, Vec3::X).abs_diff_eq(Vec2::new(0.25, 0.0), 1e-6));
    assert!(
        projected_perturbation(projection * yaw, Vec3::Z).abs_diff_eq(Vec2::new(0.25, 0.0), 1e-6)
    );
    assert!(
        projected_perturbation(projection * roll, Vec3::X).abs_diff_eq(Vec2::new(0.0, -0.5), 1e-6)
    );
    for rotation in [Mat4::IDENTITY, yaw, roll, roll * yaw] {
        let camera = projection * rotation;
        assert_eq!(projected_perturbation(camera, Vec3::Y), Vec2::ZERO);
        let translated = camera * Mat4::from_translation(Vec3::new(100.0, -25.0, 7.0));
        assert!(
            projected_perturbation(camera, Vec3::new(0.3, 0.8, -0.2)).abs_diff_eq(
                projected_perturbation(translated, Vec3::new(0.3, 0.8, -0.2)),
                1e-6
            )
        );
    }
}

#[test]
fn projection_preserves_fov_and_aspect_scale() {
    let wide = Mat4::perspective_infinite_reverse_rh(std::f32::consts::FRAC_PI_2, 2.0, 0.1);
    let square = Mat4::perspective_infinite_reverse_rh(std::f32::consts::FRAC_PI_2, 1.0, 0.1);
    let narrow = Mat4::perspective_infinite_reverse_rh(std::f32::consts::FRAC_PI_3, 1.0, 0.1);
    assert!(
        (projected_perturbation(square, Vec3::X).x / projected_perturbation(wide, Vec3::X).x - 2.0)
            .abs()
            < 1e-6
    );
    assert!(
        (projected_perturbation(narrow, Vec3::X).x / projected_perturbation(square, Vec3::X).x
            - 3.0_f32.sqrt())
        .abs()
            < 1e-6
    );
}

#[test]
fn reconstructed_water_path_is_metric_off_axis_and_orthographic() {
    let perspective = Mat4::perspective_infinite_reverse_rh(2.0, 1.6, 0.1);
    let orthographic = Mat4::orthographic_rh(-20.0, 20.0, -20.0, 20.0, 100.0, 0.1);
    for (projection, water, bed, expected) in [
        // Same perspective ray: 3-4-5 triangle, not axial separation 4.
        (
            perspective,
            Vec3::new(3.0, 0.0, -4.0),
            Vec3::new(6.0, 0.0, -8.0),
            5.0,
        ),
        // Parallel orthographic ray: X/Y must not scale with depth.
        (
            orthographic,
            Vec3::new(7.0, 2.0, -4.0),
            Vec3::new(7.0, 2.0, -9.0),
            5.0,
        ),
    ] {
        let (uv, raw_depth) = project(projection, bed);
        let reconstructed = reconstruct(projection.inverse(), uv, raw_depth);
        assert!(reconstructed.abs_diff_eq(bed, 2e-5));
        assert!((accepted_path(0.0, water, Some(reconstructed)) - expected).abs() < 2e-5);
    }
}

#[test]
fn accepted_path_controls_absorption_and_opacity_gate() {
    let water = Vec3::new(1.0, 2.0, -5.0);
    let bed = water + Vec3::new(3.0, 0.0, -4.0);
    let path = accepted_path(40.0, water, Some(bed));
    assert!((path - 5.0).abs() < 1e-6);
    // A shallow refracted bed must remain visible even if the original path
    // is already opaque at the least-attenuated channel's extinction.
    let extinction = 0.3;
    let cutoff = 10.0 * 2.0_f32.ln();
    assert!(extinction * path < cutoff);
    assert!(extinction * 40.0 > cutoff);
    assert!((-extinction * path).exp() > 0.2);
    assert!((-extinction * 40.0).exp() < 0.001);
    // Distances survive a rigid camera transform, including translation.
    let camera = Mat4::from_rotation_translation(Quat::from_rotation_y(0.7), Vec3::splat(8.0));
    assert!(
        (accepted_path(
            40.0,
            camera.transform_point3(water),
            Some(camera.transform_point3(bed))
        ) - path)
            .abs()
            < 2e-6
    );
}

#[test]
fn clear_equal_and_foreground_depth_keep_the_original_path() {
    let projection = Mat4::perspective_infinite_reverse_rh(1.2, 1.6, 0.1);
    let water = Vec3::new(0.0, 0.0, -5.0);
    let (_, water_depth) = project(projection, water);
    for candidate in [None, Some(water), Some(Vec3::new(0.0, 0.0, -2.0))] {
        let raw_depth = candidate.map_or(0.0, |point| project(projection, point).1);
        let valid = valid_refracted_depth(raw_depth, water_depth);
        assert!(!valid);
        // Lazy reconstruction is important: clear reverse-Z is at infinity.
        let background =
            valid.then(|| reconstruct(projection.inverse(), Vec2::splat(0.5), raw_depth));
        assert_eq!(accepted_path(7.0, water, background), 7.0);
    }
    let bed = Vec3::new(0.0, 0.0, -10.0);
    assert!(valid_refracted_depth(
        project(projection, bed).1,
        water_depth
    ));
}

#[test]
fn wgsl_projection_and_metric_reconstruction_match_cpu_contracts() {
    assert_shader_contains(
        "let ndc = vec3(uv * vec2(2.0, -2.0) + vec2(-1.0, 1.0), raw_depth); let position = view.view_from_clip * vec4(ndc, 1.0); return position.xyz / max(position.w, LUMINANCE_EPSILON);",
    );
    assert_shader_contains(
        "let clip_perturbation = view.clip_from_world * vec4(normal.x, 0.0, normal.z, 0.0); let screen_perturbation = 0.5 * vec2(clip_perturbation.x, -clip_perturbation.y);",
    );
    assert_shader_contains(
        "if background.z < water.z { result.path_length = length(background - water); }",
    );
    assert_shader_contains(
        "result.has_background = scene_raw_depth > 0.0; if result.has_background {",
    );
    assert_shader_contains("result.refracted_path_length = path.path_length;");
    assert_shader_contains(
        "result.refracted_sample_valid = refracted_raw_depth > 0.0 && refracted_raw_depth < in.position.z; if result.refracted_sample_valid { let background = camera_view_position(result.refracted_uv, refracted_raw_depth); let water = (view.view_from_world * in.world_position).xyz; result.refracted_path_length = length(background - water); result.refracted_receiver_world = (view.world_from_view * vec4(background, 1.0)).xyz; }",
    );
}

// Function-local order checks keep sampling guards from matching another path.
fn assert_in_order(source: &str, fragments: &[&str]) {
    let mut remaining = source;
    for fragment in fragments {
        let fragment = compact(fragment);
        let (_, tail) = remaining
            .split_once(&fragment)
            .unwrap_or_else(|| panic!("WGSL ordered contract changed: {fragment}"));
        remaining = tail;
    }
}

#[test]
fn wgsl_beauty_gates_and_attenuates_the_accepted_path() {
    let beauty = compact_wgsl_function(OPTICS, "beauty_transmission");
    // Keep negated positive comparisons: <= / >= would admit NaN paths.
    assert_in_order(
        &beauty,
        &[
            "if !(depth_path.has_background && depth_path.path_length > LUMINANCE_EPSILON) { return open_transmission(to_view); }",
            "let depth_debug = camera_depth_debug_from_path(in, normal, depth_path);",
            "let use_refraction = depth_debug.refracted_sample_valid; let water_path = select(depth_debug.path_length, depth_debug.refracted_path_length, use_refraction,);",
            "let extinction = invocation_extinction();",
            "let minimum_extinction = min(extinction.r, min(extinction.g, extinction.b));",
            "let opacity = column_opacity(water_path);",
            "if !(minimum_extinction * water_path < TRANSMISSION_OPAQUE_OPTICAL_DEPTH) { let body = surface_medium_radiance(vec3(0.0), to_view, water_path); return TransmissionState(body, vec4(0.0), false, opacity); }",
            "let background_uv = select(depth_debug.screen_uv, depth_debug.refracted_uv, use_refraction,);",
            "let scene_colour = opaque_background(background_uv);",
            "let lit_scene = illuminate_bed(scene_colour, in, primary, depth_debug, use_refraction, background_uv, source_slot,);",
            "let body = surface_medium_radiance(lit_scene, to_view, water_path);",
            "return TransmissionState(body, vec4(0.0), false, opacity);",
        ],
    );
    assert_eq!(beauty.matches("camera_depth_debug_from_path(").count(), 1);
    assert_eq!(beauty.matches("opaque_background(").count(), 1);
    assert!(!beauty.contains("camera_depth_path("));
}

#[test]
fn wgsl_transmission_routes_modes_before_sampling_and_reuses_foam_depth() {
    let resolve = compact_wgsl_function(OPTICS, "resolve_transmission");
    assert_in_order(
        &resolve,
        &[
            "let is_diagnostic = mode >= DEBUG_MODE_WATER_PATH && mode <= DEBUG_MODE_SEA_FLOOR;",
            "if mode != DEBUG_MODE_BEAUTY && !is_diagnostic { return open_transmission(to_view); }",
            "var shared_depth_path = foam.depth_path; if !foam.has_depth_path { shared_depth_path = camera_depth_path(in); }",
            "if mode == DEBUG_MODE_BEAUTY { return beauty_transmission(in, normal, to_view, primary, shared_depth_path, source_slot,); }",
            "let depth_debug = camera_depth_debug_from_path(in, normal, shared_depth_path);",
            "if mode == DEBUG_MODE_WATER_PATH {",
            "return TransmissionState(body, vec4(vec3(path), 1.0), true, vec3(1.0));",
            "if mode == DEBUG_MODE_REFRACTION_VALIDITY {",
            "return TransmissionState(body, output, true, vec3(1.0));",
            "let refraction_enabled = mode == DEBUG_MODE_TRANSMISSION || mode == DEBUG_MODE_BEER_LAMBERT || mode == DEBUG_MODE_SEA_FLOOR;",
            "let use_refraction = refraction_enabled && depth_debug.refracted_sample_valid;",
            "let background_uv = select(depth_debug.screen_uv, depth_debug.refracted_uv, use_refraction,);",
            "let scene_colour = opaque_background(background_uv);",
            "if mode == DEBUG_MODE_TRANSMISSION || mode == DEBUG_MODE_UNREFRACTED { return TransmissionState(body, vec4(scene_colour, 1.0), true, vec3(1.0)); }",
            "let lit_scene = illuminate_bed(scene_colour, in, primary, depth_debug, use_refraction, background_uv, source_slot,);",
            "let water_path = select(depth_debug.path_length, depth_debug.refracted_path_length, use_refraction,);",
            "body = surface_medium_radiance(lit_scene, to_view, water_path);",
            "if mode == DEBUG_MODE_BEER_LAMBERT { return TransmissionState(body, vec4(body, 1.0), true, vec3(1.0)); }",
            "return TransmissionState(body, vec4(0.0), false, column_opacity(water_path));",
        ],
    );
    assert_eq!(resolve.matches("camera_depth_path(").count(), 1);
    assert_eq!(resolve.matches("camera_depth_debug_from_path(").count(), 1);
    assert_eq!(resolve.matches("beauty_transmission(").count(), 1);
    assert_eq!(resolve.matches("opaque_background(").count(), 1);
    // Diagnostics must not inherit beauty's opacity or missing-background gates.
    assert!(!resolve.contains("TRANSMISSION_OPAQUE_OPTICAL_DEPTH"));
    assert!(!resolve.contains("has_background"));
    assert!(!resolve.contains("shallow_extinction_scale"));
}

#[test]
fn wgsl_braces_are_balanced() {
    let mut depth = 0i32;
    for (line_index, line) in OPTICS.lines().enumerate() {
        let code = line.split("//").next().unwrap_or("");
        for character in code.chars() {
            match character {
                '{' => depth += 1,
                '}' => {
                    depth -= 1;
                    assert!(
                        depth >= 0,
                        "WGSL has an unmatched closing brace on line {}",
                        line_index + 1
                    );
                }
                _ => {}
            }
        }
    }
    assert_eq!(depth, 0, "WGSL has unmatched opening braces");
}
