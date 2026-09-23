//! Prototype regression contracts, not approval of geometry, optics, or GPU cost.
//! CPU references do not execute WGSL. The frozen-offset 0.05 m / 1% view-Z
//! heuristic can reject smooth slopes and accept small inter-surface jumps.
const OPTICS: &str = include_str!("optics.wgsl");
const SCREEN: &str = include_str!("screen.wgsl");

fn function(name: &str) -> &str {
    let marker = format!("fn {name}(");
    let source = if SCREEN.contains(&marker) { SCREEN } else { OPTICS };
    let start = source.find(&marker).unwrap();
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

#[test]
fn centers_initialize_reconstruct_once_and_reuse_without_depth_reload() {
    let types = include_str!("../../bevy-aqua-core/src/cascade/types.wgsl");
    for clause in [
        "receiver_world: vec3<f32>,",
        "raw_receiver_world: vec3<f32>,",
        "refracted_receiver_world: vec3<f32>,",
    ] {
        assert!(types.contains(clause));
    }
    let empty = function("empty_camera_depth_path");
    for clause in [
        "result.path_length = 0.0;",
        "result.receiver_world = vec3(0.0);",
        "result.has_background = false;",
    ] {
        assert!(empty.contains(clause));
    }
    let raw = function("camera_depth_path");
    ordered(
        raw,
        &[
            "var result = empty_camera_depth_path();",
            "#ifdef DEPTH_PREPASS",
            "let scene_raw_depth = prepass_utils::prepass_depth(in.position, 0u);",
            "result.has_background = scene_raw_depth > 0.0;",
            "if result.has_background {",
            "let background = camera_view_position(result.screen_uv, scene_raw_depth);",
            "result.receiver_world = (view.world_from_view * vec4(background, 1.0)).xyz;",
            "if background.z < water.z {",
            "result.path_length = length(background - water);",
        ],
    );
    assert_eq!(raw.matches("prepass_utils::prepass_depth(").count(), 1);
    let refracted = function("camera_depth_debug_from_path");
    ordered(
        refracted,
        &[
            "result.refracted_path_length = path.path_length;",
            "result.refracted_uv = path.screen_uv;",
            "result.raw_receiver_world = path.receiver_world;",
            "result.refracted_receiver_world = path.receiver_world;",
            "result.refracted_sample_valid = false;",
            "result.has_background = path.has_background;",
            "#ifdef DEPTH_PREPASS",
            "let refracted_raw_depth = prepass_utils::prepass_depth(refracted_position, 0u);",
            "result.refracted_sample_valid = refracted_raw_depth > 0.0\n        && refracted_raw_depth < in.position.z;",
            "if result.refracted_sample_valid {",
            "result.refracted_path_length = length(background - water);",
            "result.refracted_receiver_world = (view.world_from_view * vec4(background, 1.0)).xyz;",
        ],
    );
    assert_eq!(
        refracted.matches("prepass_utils::prepass_depth(").count(),
        1
    );
    for name in ["illuminate_bed", "resolve_transmission"] {
        assert!(!function(name).contains("prepass_utils::prepass_depth("));
    }
}

#[test]
fn selected_flag_keeps_uv_receiver_and_path_on_same_raw_or_refracted_lane() {
    let resolve = function("resolve_transmission");
    ordered(
        resolve,
        &[
            "let refraction_enabled = mode == DEBUG_MODE_TRANSMISSION",
            "|| mode == DEBUG_MODE_BEER_LAMBERT",
            "|| mode == DEBUG_MODE_SEA_FLOOR;",
            "let use_refraction = refraction_enabled\n        && depth_debug.refracted_sample_valid;",
        ],
    );
    let beauty = function("beauty_transmission");
    assert!(beauty.contains("let use_refraction = depth_debug.refracted_sample_valid;"));
    // Whitespace-normalized exact expressions, not merely presence of field names.
    for lane in [resolve, beauty] {
        let compact: String = lane.split_whitespace().collect();
        for clause in [
            "select(depth_debug.screen_uv,depth_debug.refracted_uv,use_refraction,)",
            "select(depth_debug.path_length,depth_debug.refracted_path_length,use_refraction,)",
            "illuminate_bed(scene_colour,in,primary,depth_debug,use_refraction,background_uv,source_slot,)",
        ] {
            assert!(compact.contains(clause), "{clause}");
        }
    }
    let bed = function("illuminate_bed");
    assert!(bed.contains("select(depth.path_length, depth.refracted_path_length, use_refraction)"));
    assert!(bed.contains(
        "select(depth.raw_receiver_world, depth.refracted_receiver_world, use_refraction)"
    ));
    assert!(!bed.contains("in.undisplaced_xz"));
}

#[test]
fn receiver_admission_precedes_neighbors_and_caustic_texture_sampling() {
    ordered(
        function("illuminate_bed"),
        &[
            "if surface.caustics.x * surface.sea_floor.w <= 0.0 || !depth.has_background {",
            "return scene_colour;",
            "if !(selected_path > LUMINANCE_EPSILON) { return scene_colour; }",
            "if !(all(abs(receiver) < vec3(1e20))) { return scene_colour; }",
            "var receiver_slot = 0u;",
            "var level = cascade_layout.bed_range.z;",
            "if field_params.info.x > 0.5 {",
            "let field = sample_field_level(receiver.xz);",
            "receiver_slot = u32(field.y + 0.5);",
            "if receiver_slot != source_slot { return scene_colour; }",
            "if receiver_slot != 0u { level = field.x; }",
            "} else if source_slot != 0u {",
            "return scene_colour;",
            "let water_depth = level - receiver.y;",
            "if !(water_depth > LUMINANCE_EPSILON && water_depth < surface.caustics.w) {",
            "return scene_colour;",
            "let incident = strongest_incident_directional_light(",
            "if !incident.valid || incident.direction.y <= 0.0 || incident.shadow <= 0.0 {",
            "return scene_colour;",
            "let footprint = receiver_caustic_footprint(in, background_uv, receiver, use_refraction);",
            "if footprint < 0.0 { return scene_colour; }",
            "return caustic_bed_radiance(\n        scene_colour,\n        receiver.xz,\n        water_depth,\n        footprint,",
        ],
    );
}

#[test]
fn footprint_is_four_conditional_neighbors_with_max_span_and_invalid_sentinel() {
    let fp = function("receiver_caustic_footprint");
    ordered(
        fp,
        &[
            "#ifdef DEPTH_PREPASS",
            "let span = view.viewport.zw;",
            "if any(span <= vec2(1.0)) { return -1.0; }",
            "let step_uv = vec2(1.0) / span;",
            "if any(background_uv <= step_uv) || any(background_uv >= vec2(1.0) - step_uv) {",
            "return -1.0;",
            "let center_view = (view.view_from_world * vec4(receiver_world, 1.0)).xyz;",
            "if !(all(abs(center_view) < vec3(1e20))) { return -1.0; }",
            "let maximum_z_jump = max(0.05, abs(center_view.z) * 0.01);",
            "for (var i = 0u; i < 4u; i++) {",
            "var offset = vec2(1.0, 0.0);",
            "if i == 1u { offset = vec2(-1.0, 0.0); }",
            "if i == 2u { offset = vec2(0.0, 1.0); }",
            "if i == 3u { offset = vec2(0.0, -1.0); }",
            "let uv = background_uv + offset * step_uv;",
            "let pixel = select(",
            "uv * span,",
            "min(uv * span, span - vec2(1.0)),",
            "use_refraction,",
            ") + view.viewport.xy;",
            "let raw_depth = prepass_utils::prepass_depth(vec4(pixel, in.position.zw), 0u);",
            "if !(raw_depth > 0.0 && raw_depth < in.position.z) { return -1.0; }",
            "let neighbor_view = camera_view_position(uv, raw_depth);",
            "if !(all(abs(neighbor_view) < vec3(1e20))) { return -1.0; }",
            "if abs(neighbor_view.z - center_view.z) > maximum_z_jump { return -1.0; }",
            "let neighbor_world = (view.world_from_view * vec4(neighbor_view, 1.0)).xyz;",
            "if !(all(abs(neighbor_world) < vec3(1e20))) { return -1.0; }",
            "footprint = max(footprint, length(neighbor_world.xz - receiver_world.xz));",
            "if !(footprint > 0.0 && footprint < 1e20) { return -1.0; }",
            "return footprint;",
            "#else",
            "return -1.0;",
            "#endif",
        ],
    );
    assert_eq!(fp.matches("prepass_utils::prepass_depth(").count(), 1);
    for forbidden in [
        "return 0.0;",
        "dpdx(",
        "dpdy(",
        "textureSample",
        "clamp(",
        "camera_depth_debug_from_path(",
    ] {
        assert!(!fp.contains(forbidden));
    }
    // No neighbor refraction replay: a limitation, not physical correctness.
    assert!(OPTICS.contains("frozen selected refraction offset, NOT neighbor acceptance replay"));
    assert!(
        fp.contains("Smooth steep slopes may be rejected too. Small discontinuities can pass;")
    );
    assert!(fp.contains("this is not a surface-ID test"));
}

fn receiver_depth(
    ocean: f32,
    field: Option<(f32, u32)>,
    source: u32,
    bed: f32,
    limit: f32,
) -> Option<f32> {
    let level = match field {
        Some((_, slot)) if slot != source => return None,
        Some((level, slot)) if slot != 0 => level,
        None if source != 0 => return None,
        _ => ocean,
    };
    let depth = level - bed;
    (depth > 1e-6 && depth < limit).then_some(depth)
}
#[test]
fn cpu_receiver_level_uses_ocean_or_same_body_not_surface_fragment_height() {
    assert_eq!(receiver_depth(10.0, None, 0, 8.0, 20.0), Some(2.0));
    // Slot zero uses ocean elevation even if the sampled field level differs.
    assert_eq!(
        receiver_depth(10.0, Some((-99.0, 0)), 0, 8.0, 20.0),
        Some(2.0)
    );
    for (level, bed) in [(10.0, 8.0), (-10.0, -12.0)] {
        assert_eq!(
            receiver_depth(0.0, Some((level, 3)), 3, bed, 20.0),
            Some(2.0)
        );
        assert_eq!(receiver_depth(0.0, Some((level, 3)), 2, bed, 20.0), None);
        assert_eq!(receiver_depth(level, None, 3, bed, 20.0), None);
        assert_eq!(receiver_depth(level, None, 0, level, 20.0), None);
        assert_eq!(receiver_depth(level, None, 0, level + 1.0, 20.0), None);
        assert_eq!(receiver_depth(level, None, 0, level - 20.0, 20.0), None);
    }
    assert_eq!(receiver_depth(0.0, Some((10.0, 3)), 0, 8.0, 20.0), None);
    assert_eq!(receiver_depth(10.0, Some((10.0, 0)), 3, 8.0, 20.0), None);
    for bed in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert_eq!(receiver_depth(10.0, None, 0, bed, 20.0), None);
    }
}

#[test]
fn cpu_selection_falls_back_exactly_and_edge_threshold_is_inclusive_monotone() {
    let raw = ([1.0_f32, 8.0, -3.0], [0.2, 0.3], 4.0);
    let refracted = ([9.0_f32, -2.0, 6.0], [0.7, 0.8], 12.0);
    for (enabled, valid, expected) in [
        (false, false, raw),
        (false, true, raw),
        (true, false, raw),
        (true, true, refracted),
    ] {
        let selected = if enabled && valid { refracted } else { raw };
        assert_eq!(selected, expected);
    }
    let threshold = |z: f32| 0.05_f32.max(z.abs() * 0.01);
    let accepted = |z: f32, jump: f32| jump.abs() <= threshold(z);
    let mut previous = 0.0;
    for depth in [0.0, 1.0, 4.99, 5.0, 5.01, 10.0, 100.0, 1000.0] {
        let t = threshold(-depth);
        assert!(t >= previous);
        assert_eq!(t, threshold(depth));
        assert!(accepted(-depth, t));
        assert!(accepted(-depth, -t));
        assert!(!accepted(-depth, t + 1e-4));
        previous = t;
    }
    assert_eq!(threshold(-5.0), 0.05);
    assert!(accepted(-100.0, 0.5)); // unrelated surfaces can pass at distance
    assert!(!accepted(-1.0, 0.06)); // a smooth steep slope can fail nearby
}

// CPU references use f32 like WGSL; source guards below bind their arithmetic
// to the shader. They do not execute GPU loads or certify caustic acceptance.
fn refracted_depth_texel(uv: f32, size: f32, origin: f32) -> i32 {
    ((uv * size).min(size - 1.0) + origin) as i32
}

#[test]
fn zero_distortion_depth_texels_match_raw_centers_for_even_and_odd_viewports() {
    // Each axis is independent; include native dimensions, odd extents, and
    // one-texel viewports. Integer nonzero origins model viewport offsets.
    for extent in [1, 2, 3, 7, 799, 800, 1024, 1025, 1439, 1440, 2559, 2560] {
        let size = extent as f32;
        for origin in [0.0_f32, 23.0, 47.0] {
            for index in 0..extent {
                let fragment = origin + index as f32 + 0.5;
                let uv = (fragment - origin) / size;
                assert_eq!(
                    refracted_depth_texel(uv, size, origin),
                    fragment as i32,
                    "extent={extent}, origin={origin}, index={index}",
                );
            }
        }
    }
}

#[test]
fn refracted_depth_uv_endpoints_stay_in_first_and_last_viewport_texels() {
    for extent in [1, 2, 3, 799, 800, 1024, 1025, 1440, 2559, 2560] {
        let size = extent as f32;
        for origin in [0.0_f32, 23.0, 47.0] {
            assert_eq!(refracted_depth_texel(0.0, size, origin), origin as i32);
            assert_eq!(
                refracted_depth_texel(1.0, size, origin),
                origin as i32 + extent - 1,
            );
            // The bound must not shrink interior UVs into the previous texel.
            let final_center = (size - 0.5) / size;
            assert_eq!(
                refracted_depth_texel(final_center, size, origin),
                origin as i32 + extent - 1,
            );
        }
    }
}

#[test]
fn footprint_neighbors_step_one_actual_texel_with_matching_raw_and_refracted_centers() {
    for extent in [3, 4, 7, 799, 800, 1024, 1025, 1439, 1440, 2559, 2560] {
        let size = extent as f32;
        let step_uv = 1.0 / size;
        for origin in [0.0_f32, 23.0, 47.0] {
            for index in 1..extent - 1 {
                let uv = (index as f32 + 0.5) / size;
                assert!(uv > step_uv && uv < 1.0 - step_uv);
                let center = refracted_depth_texel(uv, size, origin);
                assert_eq!(center, (uv * size + origin) as i32);
                for sign in [-1, 1] {
                    let neighbor_uv = uv + sign as f32 * step_uv;
                    let neighbor = refracted_depth_texel(neighbor_uv, size, origin);
                    assert_eq!(neighbor, center + sign);
                    assert_eq!(neighbor, (neighbor_uv * size + origin) as i32);
                }
            }
            // Bounds rejection remains conservative, inclusive, and precedes
            // all neighbor loads. Do not clamp a rejected footprint inward.
            for uv in [0.0, 0.5 / size, step_uv, 1.0 - step_uv, 1.0] {
                assert!(uv <= step_uv || uv >= 1.0 - step_uv);
            }
        }
    }
}

#[test]
fn refracted_depth_uses_viewport_size_and_endpoint_bound_without_snapping_the_ray() {
    let refracted = function("camera_depth_debug_from_path");
    ordered(
        refracted,
        &[
            "result.refracted_uv = clamp(path.screen_uv + refract_offset, vec2(0.0), vec2(1.0));",
            "let refracted_pixel = min(",
            "result.refracted_uv * view.viewport.zw,",
            "view.viewport.zw - vec2(1.0),",
            ") + view.viewport.xy;",
            "let refracted_position = vec4(refracted_pixel, in.position.zw);",
            "let refracted_raw_depth = prepass_utils::prepass_depth(refracted_position, 0u);",
            "let background = camera_view_position(result.refracted_uv, refracted_raw_depth);",
        ],
    );
    assert!(!refracted.contains("result.refracted_uv * (view.viewport.zw - vec2(1.0))"));
    assert!(
        function("camera_depth_path")
            .contains("let scene_raw_depth = prepass_utils::prepass_depth(in.position, 0u);")
    );
    assert!(
        function("camera_view_position")
            .contains("let ndc = vec3(uv * vec2(2.0, -2.0) + vec2(-1.0, 1.0), raw_depth);")
    );
    ordered(
        function("opaque_background"),
        &[
            "subview_uv * view.viewport.zw + view.viewport.xy",
            "let full_uv = color_pixel / dimensions;",
            "return textureSampleLevel(",
            "view_bindings::view_transmission_texture,",
            "view_bindings::view_transmission_sampler,",
            "full_uv,",
            "0.0,",
        ],
    );
}

#[test]
fn cpu_footprint_rejects_clamping_foreground_and_degenerate_not_sharp_zero() {
    // Mirrors the shader's admission and max-of-four XZ distances. The depth
    // values are reverse-Z; center water depth is 0.5 in this fixture.
    let footprint = |uv: f32, span: f32, raw: [f32; 4], distances: [f32; 4]| {
        if span <= 1.0 || uv <= 1.0 / span || uv >= 1.0 - 1.0 / span {
            return None;
        }
        let mut maximum = 0.0_f32;
        for (depth, distance) in raw.into_iter().zip(distances) {
            if !(depth > 0.0 && depth < 0.5) || !distance.is_finite() {
                return None;
            }
            maximum = maximum.max(distance);
        }
        (maximum > 0.0 && maximum < 1e20).then_some(maximum)
    };
    assert_eq!(
        footprint(0.5, 100.0, [0.2; 4], [0.1, 0.4, 0.3, 0.2]),
        Some(0.4)
    );
    for index in 0..4 {
        for invalid in [0.0, -0.1, 0.5, 0.6, f32::NAN] {
            let mut depths = [0.2; 4];
            depths[index] = invalid;
            assert_eq!(footprint(0.5, 100.0, depths, [0.1; 4]), None);
        }
    }
    for uv in [0.0, 0.01, 0.99, 1.0] {
        assert_eq!(footprint(uv, 100.0, [0.2; 4], [0.1; 4]), None);
    }
    assert_eq!(footprint(0.5, 1.0, [0.2; 4], [0.1; 4]), None);
    for distances in [[0.0; 4], [f32::NAN; 4], [f32::INFINITY; 4], [1e20; 4]] {
        assert_eq!(footprint(0.5, 100.0, [0.2; 4], distances), None);
    }
}

#[test]
fn transmission_color_clamps_physical_pixel_to_viewport_centers_before_normalizing() {
    ordered(
        function("opaque_background"),
        &[
            "let dimensions = vec2<f32>(textureDimensions(view_bindings::view_transmission_texture));",
            "let color_pixel = clamp(",
            "subview_uv * view.viewport.zw + view.viewport.xy,",
            "view.viewport.xy + vec2(0.5),",
            "view.viewport.xy + view.viewport.zw - vec2(0.5),",
            ");",
            "let full_uv = color_pixel / dimensions;",
            "return textureSampleLevel(",
            "view_bindings::view_transmission_texture,",
            "view_bindings::view_transmission_sampler,",
            "full_uv,",
            "0.0,",
        ],
    );
    let color = function("opaque_background");
    assert_eq!(color.matches("clamp(").count(), 1);
    assert_eq!(color.matches("textureSampleLevel(").count(), 1);
    assert!(!color.contains("textureLoad("));
    assert!(!color.contains("prepass_depth("));
}

// Independent scalar-channel reference for mip-zero bilinear ClampToEdge.
// Integer viewport origin/size, nonempty viewport, full-resolution backing.
// These CPU contracts do not execute WGSL or establish native visual acceptance.
fn transmission_color_pixel(
    uv: [f32; 2],
    origin: [u32; 2],
    size: [u32; 2],
    inset: bool,
) -> [f32; 2] {
    std::array::from_fn(|axis| {
        let lo = origin[axis] as f32;
        let span = size[axis] as f32;
        let pixel = uv[axis] * span + lo;
        if inset {
            pixel.clamp(lo + 0.5, lo + span - 0.5)
        } else {
            pixel
        }
    })
}

fn linear_transmission_sample(
    pixel: [f32; 2],
    target: [u32; 2],
    texel: impl Fn(u32, u32) -> f32,
) -> f32 {
    // Include physical-dimension normalization and the inverse sampler mapping.
    let p: [f32; 2] = std::array::from_fn(|a| {
        let uv = pixel[a] / target[a] as f32;
        uv * target[a] as f32 - 0.5
    });
    let base = [p[0].floor(), p[1].floor()];
    let fraction = [p[0] - base[0], p[1] - base[1]];
    let mut value = 0.0;
    for y in 0..2 {
        for x in 0..2 {
            let ix = (base[0] + x as f32).clamp(0.0, (target[0] - 1) as f32) as u32;
            let iy = (base[1] + y as f32).clamp(0.0, (target[1] - 1) as f32) as u32;
            let wx = if x == 0 {
                1.0 - fraction[0]
            } else {
                fraction[0]
            };
            let wy = if y == 0 {
                1.0 - fraction[1]
            } else {
                fraction[1]
            };
            value += wx * wy * texel(ix, iy);
        }
    }
    value
}

#[test]
fn inset_linear_filter_isolates_exterior_sentinels_at_edges_and_corners() {
    let origin = [64, 48];
    let size = [512, 384];
    let sample = |uv, inset, sentinel| {
        linear_transmission_sample(
            transmission_color_pixel(uv, origin, size, inset),
            [640, 480],
            |x, y| {
                if (64..576).contains(&x) && (48..432).contains(&y) {
                    0.0
                } else {
                    sentinel
                }
            },
        )
    };
    for uv in [[0.0, 0.5], [1.0, 0.5], [0.5, 0.0], [0.5, 1.0]] {
        assert!((sample(uv, false, 1.0) - 0.5).abs() < 1e-5);
    }
    for u in [0.0, 0.25 / 512.0, 0.5, 1.0 - 0.25 / 512.0, 1.0] {
        for v in [0.0, 0.25 / 384.0, 0.5, 1.0 - 0.25 / 384.0, 1.0] {
            let uv = [u, v];
            assert_eq!(sample(uv, true, 1.0), sample(uv, true, 0.0));
        }
    }
    for uv in [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]] {
        assert!((sample(uv, false, 1.0) - 0.75).abs() < 1e-5);
        assert_eq!(sample(uv, true, 1.0), 0.0);
    }
}

#[test]
fn full_target_color_clamp_matches_physical_clamp_to_edge() {
    for size in [[512, 384], [7, 3], [1, 1], [1, 7], [7, 1]] {
        for u in [0.0, 0.001, 0.25, 0.5, 0.999, 1.0] {
            for v in [0.0, 0.001, 0.25, 0.5, 0.999, 1.0] {
                let sample = |inset| {
                    linear_transmission_sample(
                        transmission_color_pixel([u, v], [0, 0], size, inset),
                        size,
                        |x, y| ((x * 7 + y * 13) % 31) as f32 / 31.0,
                    )
                };
                assert!((sample(false) - sample(true)).abs() < 1e-5);
            }
        }
    }
}

#[test]
fn interior_color_coordinates_and_linear_interpolation_remain_unchanged() {
    for (origin, size) in [([64, 48], [512, 384]), ([3, 5], [7, 9]), ([0, 0], [3, 5])] {
        for y in 0..size[1] {
            for x in 0..size[0] {
                let uv = [
                    (x as f32 + 0.5) / size[0] as f32,
                    (y as f32 + 0.5) / size[1] as f32,
                ];
                assert_eq!(
                    transmission_color_pixel(uv, origin, size, false),
                    transmission_color_pixel(uv, origin, size, true)
                );
            }
        }
        // Non-center interior coordinate: retain interpolation, do not snap.
        let uv = [1.25 / size[0] as f32, 1.75 / size[1] as f32];
        let old = transmission_color_pixel(uv, origin, size, false);
        let new = transmission_color_pixel(uv, origin, size, true);
        assert_eq!(old, new);
        let value = linear_transmission_sample(new, [640, 480], |x, y| (x + y) as f32);
        assert!((value - (origin[0] + origin[1]) as f32 - 2.0).abs() < 1e-4);
    }
}

#[test]
fn singleton_axes_and_nonzero_origins_clamp_to_their_own_texel_centers() {
    for origin in [[0, 0], [23, 47], [64, 48]] {
        for size in [[1, 1], [1, 7], [9, 1], [7, 9]] {
            for uv in [[0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [1.0, 0.0], [0.5, 0.5]] {
                let pixel = transmission_color_pixel(uv, origin, size, true);
                for a in 0..2 {
                    assert!(pixel[a] >= origin[a] as f32 + 0.5);
                    assert!(pixel[a] <= (origin[a] + size[a]) as f32 - 0.5);
                    if size[a] == 1 {
                        assert_eq!(pixel[a], origin[a] as f32 + 0.5);
                    }
                }
                let sentinel = linear_transmission_sample(pixel, [640, 480], |x, y| {
                    if (origin[0]..origin[0] + size[0]).contains(&x)
                        && (origin[1]..origin[1] + size[1]).contains(&y)
                    {
                        0.0
                    } else {
                        1.0
                    }
                });
                assert!(sentinel.abs() < 1e-5);
            }
        }
    }
}
