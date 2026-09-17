use super::*;

#[test]
fn fixed_cadence_is_independent_of_render_rate() {
    for fps in [30_u32, 60, 120] {
        let elapsed = (0..fps).fold(0.0, |time, _| time + 1.0 / f64::from(fps));
        assert_eq!(target_tick(elapsed), 30, "{fps} Hz");
    }
    assert_eq!(target_tick(0.0), 0);
    assert_eq!(target_tick(STEP_SECONDS as f64), 1);
}

#[test]
fn state_texture_matches_the_generated_shader_contract() {
    let image = make_state_texture();
    let size = image.texture_descriptor.size;
    assert_eq!(size.width, RESOLUTION);
    assert_eq!(size.height, RESOLUTION);
    assert_eq!(size.depth_or_array_layers, FOAM_LOD_COUNT);
    assert_eq!(image.texture_descriptor.dimension, TextureDimension::D2);
    assert_eq!(image.texture_descriptor.format, TextureFormat::Rgba16Float);
    assert!(
        image
            .texture_descriptor
            .usage
            .contains(TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING)
    );
}

// These are CPU reference math and source contracts, not WGSL execution.
// Ignore whitespace and line comments so shader layout changes are harmless.
fn shader_function(source: &str, name: &str) -> String {
    let source: String = source
        .lines()
        .map(|line| line.split("//").next().unwrap())
        .collect::<Vec<_>>()
        .join("\n")
        .chars()
        .filter(|c| !c.is_whitespace())
        .collect();
    let source = source.replace(",)", ")");
    let start = source.find(&format!("fn{name}(")).unwrap();
    let body_start = start + source[start..].find('{').unwrap();
    let mut depth = 0;
    for (offset, byte) in source[body_start..].bytes().enumerate() {
        match byte {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return source[body_start + 1..body_start + offset].to_owned();
                }
            }
            _ => {}
        }
    }
    panic!("unclosed shader function {name}");
}

fn smooth_step(edge: f32, value: f32) -> f32 {
    let t = (value / edge).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn bank_strength(margin: f32, half_width: f32, speed: f32) -> f32 {
    let width = (0.2 * half_width).clamp(0.25, 1.5);
    let bank = (1.0 - smooth_step(width, margin)) * smooth_step(0.1, margin);
    let strength = bank * (speed / 1.6 - 0.2).clamp(0.0, 1.0);
    if strength <= 0.001 { 0.0 } else { strength }
}

#[test]
fn bank_band_tapers_at_both_edges_and_leaves_channel_clear() {
    for half_width in [0.5, 2.0, 20.0] {
        let width = (0.2_f32 * half_width).clamp(0.25, 1.5);
        for margin in [-0.1, 0.0, width, half_width] {
            assert_eq!(bank_strength(margin, half_width, 2.0), 0.0);
        }
        assert!(bank_strength(0.05, half_width, 2.0) > 0.0);
        assert!(bank_strength(0.001, half_width, 2.0) < bank_strength(0.05, half_width, 2.0));
        assert!(
            bank_strength(width - 0.01, half_width, 2.0)
                < bank_strength(width * 0.5, half_width, 2.0)
        );
        for speed in [0.0, 0.16, 0.32] {
            assert_eq!(bank_strength(0.1, half_width, speed), 0.0);
        }
        assert_eq!(
            bank_strength(0.1, half_width, 2.0),
            bank_strength(0.1, half_width, 20.0)
        );
    }
}

#[test]
fn bank_coverage_union_is_bounded_and_independent_of_persistent_foam() {
    let union = |persistent: f32, bank: f32| 1.0 - (1.0 - persistent) * (1.0 - bank);
    for i in 0..=100 {
        let persistent = i as f32 / 100.0;
        for j in 0..=100 {
            let bank = j as f32 / 100.0;
            let combined = union(persistent, bank);
            assert!((0.0..=1.0).contains(&combined));
            assert!(combined + 1e-6 >= persistent.max(bank));
        }
        assert!((union(0.0, persistent) - persistent).abs() < 1e-6);
        assert_eq!(union(1.0, persistent), 1.0);
    }
    let patterned_bank = bank_strength(0.1, 2.0, 2.0) * 0.4;
    assert!(patterned_bank > 0.0);
    assert!((union(0.0, patterned_bank) - patterned_bank).abs() < 1e-6);
}

#[test]
fn river_shader_patterns_once_after_world_space_advection() {
    let river = shader_function(include_str!("shade.wgsl"), "river_streak_coverage");
    for contract in [
        "if!state.enabled{return0.0;}",
        "clamp(0.2*state.sample.w,0.25,1.5)",
        "(1.0-smoothstep(0.0,bank_width,state.sample.z))*smoothstep(0.0,0.1,state.sample.z)",
        "bank*clamp(speed/1.6-0.2,0.0,1.0)",
        "ifstrength<=0.001{return0.0;}",
        "letadvected=advected_world(world_xz);",
        "dot(dir,advected)/(1.0+0.9*min(speed,4.0))",
        "dot(vec2(-dir.y,dir.x),advected)",
        "surface_foam_mask(vec2(along,across*1.6),lod,alpha,0.7,vec2(0.0))",
        "returnstrength*pattern;",
    ] {
        assert!(
            river.contains(contract),
            "missing river contract: {contract}"
        );
    }
    assert_eq!(river.matches("surface_foam_mask(").count(), 1);
    assert_eq!(river.matches("advected_world(").count(), 1);
}

#[test]
fn material_uses_bank_coverage_without_persistent_density_gate_or_second_pattern() {
    let material = include_str!("../../bevy-aqua-core/src/cascade/material.wgsl");
    let prepare = shader_function(material, "prepare_surface_foam");
    let streak_start = prepare.find("letstreak=river_streak_coverage(").unwrap();
    // The bank call must be outside the persistent-density conditional.
    let prefix = &prepare[..streak_start];
    assert_eq!(prefix.matches('{').count(), prefix.matches('}').count());
    let bank_path = &prepare[streak_start..];
    assert!(!bank_path.contains("surface_foam_mask("));
    assert!(bank_path.contains("white_foam_density+=streak;"));
    assert!(bank_path.contains("white_foam=1.0-(1.0-white_foam)*(1.0-streak);"));
    assert!(bank_path.contains("clamp(white_foam,0.0,1.0)"));
    let local = shader_function(material, "shade_local_lights");
    assert!(local.contains("letfoam_active=foam.white_density>0.0;"));
    let compose = shader_function(material, "compose_water");
    assert!(compose.contains("iffoam.white_mask>0.0{"));
    assert!(compose.contains("clamp(CREST_FOAM_WHITE_COLOR.a*foam.white_mask,0.0,1.0)"));
}

#[test]
fn bank_roughness_keeps_the_unfaded_persistent_density() {
    // The roughness lane adds raw persistent density and bank coverage,
    // independently of the distance/shoreline attenuation in the white lane.
    let factor =
        |persistent: f32, bank: f32, fade: f32| smooth_step(1.0, (persistent + bank) * 0.75) * fade;
    for distance in [0.0_f32, 100.0, 500.0] {
        let fade = (-distance * 0.0075).exp();
        assert!((factor(1.0, 0.0, fade) - 0.84375 * fade).abs() < 1e-6);
        assert!((factor(0.0, 0.4, fade) - 0.216 * fade).abs() < 1e-6);
        assert!((factor(0.6, 0.4, fade) - factor(1.0, 0.0, fade)).abs() < 1e-6);
    }
    let material = include_str!("../../bevy-aqua-core/src/cascade/material.wgsl");
    let prepare = shader_function(material, "prepare_surface_foam");
    assert!(prepare.contains("returnFoamState(foam_density+streak,visible_foam_density,"));
    let body = shader_function(material, "shade_water_body");
    assert!(body.contains(
        "letfoam_factor=smoothstep(0.0,1.0,foam.roughness_density*0.75)*foam_distance_fade;"
    ));
}
