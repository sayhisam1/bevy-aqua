//! Source contracts and CPU checks only; native rendering is a separate gate.
const OPTICS: &str = include_str!("optics.wgsl");
const MATERIAL: &str = include_str!("../../bevy-aqua-core/src/cascade/material.wgsl");

fn function<'a>(source: &'a str, name: &str) -> &'a str {
    let start = source.find(&format!("fn {name}(")).unwrap();
    let tail = &source[start..];
    &tail[..tail.find("\n}\n").unwrap() + 3]
}

#[test]
fn resolved_normal_is_shared_without_extra_detail_sampling() {
    let near = function(OPTICS, "resolve_near_surface");
    assert!(near.contains("let lighting_normal_strength = 1.0;"));
    assert!(near.contains(
        "let lighting_distance = length(in.world_position.xz - view.world_position.xz);"
    ));
    assert!(near.contains("normal.xz / max(normal.y, MIN_NORMAL_Y)"));
    let detail_start = near.find("if near_weight != 0.0 {").unwrap();
    let detail_end = near[detail_start..].find("\n        }").unwrap() + detail_start;
    let detail_branch = &near[detail_start..detail_end];
    for call in ["detail_normal_sample(", "capillary_normal_slope("] {
        assert_eq!(near.matches(call).count(), 1);
        assert!(detail_branch.contains(call));
    }
    assert!(detail_branch.contains("near_weight * near_weight * detail.z"));
    assert!(detail_branch.contains("capillary_resolved_weight(in.undisplaced_xz)"));
    let far = function(OPTICS, "far_field_water");
    assert!(far.contains("let lighting_normal = near.lighting_normal;"));
    assert!(far.contains("sample_diffuse_environment(lighting_normal)"));
    for forbidden in [
        "GODOT_NORMAL_",
        "geometric_normal",
        "detail_normal_sample(",
        "capillary_normal_slope(",
    ] {
        assert!(!far.contains(forbidden));
    }
    assert!(far.contains("dot(light_direction, lighting_normal)")); // broad SSS
    assert!(far.contains("godot_fresnel(view_alignment) * surface.fresnel.z"));
    assert!(far.contains("return mix(body, reflected_radiance, reflection_weight);"));
}

#[test]
fn both_endpoints_use_actual_roughness_inputs_and_one_far_call() {
    let far = function(OPTICS, "far_field_water");
    let body = function(MATERIAL, "shade_water_body");
    let call = "unresolved_wave_roughness(\n        in.undisplaced_xz,\n        to_view,\n        in.sample_data.y,\n        near.lighting_normal_strength,\n        near.filtered_detail_variance,\n    )";
    assert!(far.contains(call));
    assert!(body.contains(call));
    assert!(!far.contains("max(surface.reflection.w, 0.05)"));
    assert!(far.contains("+ perceptual_roughness * perceptual_roughness"));
    assert!(far.contains("lighting_normal, perceptual_roughness);")); // planar
    assert!(far.contains("reflection,\n        lighting_normal,\n        perceptual_roughness,"));
    let fragment = function(MATERIAL, "fragment");
    assert_eq!(fragment.matches("far_field_water(").count(), 1);
    assert!(fragment.contains("far_field_water(\n            in,\n            surface_level,\n            near,\n            to_view,\n            far_water_depth,\n        )"));
    let gate = fragment
        .find("if !far_path_opaque(in, near.normal, shared_depth_path)")
        .unwrap();
    let restore = fragment[gate..]
        .find("near = resolve_near_surface")
        .unwrap()
        + gate;
    let call = fragment.find("far_water = far_field_water(").unwrap();
    let early = fragment.find("if far_tier >= 1.0").unwrap();
    let body = fragment
        .find("let primary = resolve_primary_light")
        .unwrap();
    assert!(gate < restore && restore < call && call < early && early < body);
    assert!(fragment[gate..restore].contains("far_tier = 0.0;"));
    assert!(fragment[..gate].contains("has_shared_depth_path = true;"));
    assert!(fragment[body..].contains("shared_depth_path,\n        has_shared_depth_path,"));
    assert!(fragment[body..].contains("far_water,\n        far_tier,"));
}

#[test]
fn unit_strength_removes_only_resolved_transfer_and_keeps_cap() {
    let roughness = function(OPTICS, "unresolved_wave_roughness");
    for clause in [
        "lighting_normal_strength * lighting_normal_strength",
        "+ removed_fraction * resolved_variance",
        "unresolved_variance += filtered_variance;",
        "(1.0 - capillary_resolved * capillary_resolved)",
        "surface.reflection.y * grazing_boost",
        "return min(sqrt(max(slope_variance, 0.0)), surface.reflection.w);",
    ] {
        assert!(roughness.contains(clause));
    }
    let wave_alpha = |u: f32, r: f32, s: f32| (u + (1.0 - s * s) * r).max(0.0).sqrt().min(0.28);
    for resolved in [0.0, 0.01, 0.06, 0.3] {
        assert_eq!(wave_alpha(0.0, resolved, 1.0), 0.0);
        assert!((wave_alpha(0.01, resolved, 1.0) - 0.1).abs() < 1e-6);
        assert_eq!(wave_alpha(0.2, resolved, 1.0), 0.28);
    }
    assert!(wave_alpha(0.0, 0.06, 0.015) > 0.2);
}
