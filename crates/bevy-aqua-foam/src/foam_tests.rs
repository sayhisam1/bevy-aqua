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

#[test]
fn foam_current_uses_distinct_source_and_history_coordinates() {
    let shader = include_str!("foam.wgsl");
    assert!(shader.contains("world_xz - foam.advection.xy * foam.advection.z"));
    assert!(shader.contains("world_xz - foam.advection.xy * dt"));
    assert_eq!(
        shader
            .matches("world_to_uv(wave_sample_xz(world_xz), cascade)")
            .count(),
        2
    );
    assert!(shader.contains("reproject(history_xz, slice, foam.source_layout)"));
    assert!(shader.contains("reproject(history_xz, slice, foam.target_layout)"));
    // Density lookup is world anchored: do not add a second material advection.
    let shade = include_str!("shade.wgsl");
    let density = shade
        .split("fn sample_foam_density(")
        .nth(1)
        .unwrap()
        .split("fn surface_foam_mask(")
        .next()
        .unwrap();
    assert!(!density.contains("advected_world"));
    let world = Vec2::new(11.0, -4.0);
    let flow = Vec2::new(2.0, -1.0);
    let wave_time = 3.0;
    assert_eq!(world - flow * wave_time, Vec2::new(5.0, -1.0));
    assert_eq!(world - flow * 0.0, world); // layout-only dispatch
    let dt = STEP_SECONDS;
    assert!(((world - flow * dt) + flow * dt - world).length() < 1e-5);
}

#[test]
fn foam_advection_uniform_follows_the_two_layouts_and_source_parameters() {
    let layout = lod::GpuLayout::new(&lod::layout(Vec2::ZERO), Vec2::ZERO, 0.0);
    let mut uniform = Uniform::new(layout);
    uniform.advection = Vec4::new(2.0, 0.6, 6.0, 0.0);
    let mut bytes = Vec::new();
    bevy::render::render_resource::encase::UniformBuffer::new(&mut bytes)
        .write(&uniform)
        .expect("foam uniform write");
    let expected_offset = 2 * lod::GpuLayout::min_size().get() as usize + 3 * 16;
    assert_eq!(bytes.len(), expected_offset + 16);
    let values: [f32; 4] = std::array::from_fn(|i| {
        let start = expected_offset + i * 4;
        f32::from_le_bytes(bytes[start..start + 4].try_into().unwrap())
    });
    assert_eq!(values, [2.0, 0.6, 6.0, 0.0]);
}

// These contracts inspect Rust source. They do not execute a GPU dispatch or
// establish pipeline readiness, command submission, or GPU completion.
#[test]
fn foam_history_commits_only_after_readiness_and_surface_copy_encoding() {
    let write = shader_function(include_str!("render.rs"), "write_foam");
    let encode = write.find("pass::run_spans(").unwrap();
    let before_encode = &write[..encode];
    for guard in [
        "ifview.into_inner().is_none(){return;}",
        "if!waves_status.written{return;}",
        "let(Some(state_a),Some(state_b),Some(surface))=(images.get(&frame.state_a),images.get(&frame.state_b),images.get(&frame.surface))else{return;};",
        "let(Some(group_a_to_b),Some(group_b_to_a))=(prepared.groups.get(\"a_to_b\"),prepared.groups.get(\"b_to_a\"))else{return;};",
        "letSome(ready)=prepared.passes.ready_all(&cache,&[(UPDATE,PREVIOUS),(UPDATE,PREVIOUS_ZERO),(UPDATE,CURRENT),])else{return;};",
    ] {
        assert!(
            before_encode.contains(guard),
            "missing readiness guard: {guard}"
        );
    }
    for field in ["state_is_a", "completed_tick", "state_layout"] {
        let assignment = format!("prepared.{field}=");
        assert!(
            !before_encode.contains(&assignment),
            "early history commit: {field}"
        );
        assert_eq!(
            write.matches(&assignment).count(),
            1,
            "history commit: {field}"
        );
    }
    assert!(before_encode.contains(
        "letsource=ifstate_is_a{state_a}else{state_b};steps.push(pass::Step::CopyTexture{source:&source.texture,target:&surface.texture,extent:Extent3d{width:RESOLUTION,height:RESOLUTION,depth_or_array_layers:LOD_COUNTasu32,},});"
    ));
    assert_eq!(
        &write[encode..],
        "pass::run_spans(&mutcontext,&[pass::Span::new(\"aqua_foam_compute\",steps)]);prepared.state_is_a=state_is_a;prepared.completed_tick=prepared.completed_tick.saturating_add(pending_steps.min(MAX_CATCH_UP_STEPS));prepared.state_layout=Some(frame.uniform.target_layout.clone());"
    );
}

#[test]
fn foam_capped_catchup_and_zero_step_reprojection_keep_history_in_sync() {
    let write = shader_function(include_str!("render.rs"), "write_foam");
    for contract in [
        "letpending_steps=frame.uniform.step.x.saturating_sub(prepared.completed_tick);",
        "letdispatch_count=pending_steps.clamp(1,MAX_CATCH_UP_STEPS);",
        "letstate_is_a=prepared.state_is_a;",
        "letmutstate_is_a=state_is_a;",
        "forstepin0..dispatch_count{letgroup=ifstate_is_a{group_a_to_b}else{group_b_to_a};letupdate=ifpending_steps==0{ready.get(UPDATE,PREVIOUS_ZERO)}elseifstep==0{ready.get(UPDATE,PREVIOUS)}else{ready.get(UPDATE,CURRENT)};steps.push(pass::Step::Dispatch{pipeline:update,group,workgroups,});state_is_a=!state_is_a;}",
        "prepared.completed_tick=prepared.completed_tick.saturating_add(pending_steps.min(MAX_CATCH_UP_STEPS));",
        "prepared.state_is_a=state_is_a;",
    ] {
        assert!(
            write.contains(contract),
            "missing dispatch contract: {contract}"
        );
    }
    // Even with no simulation tick pending, one reprojection dispatch swaps
    // history. A capped catch-up advances only the ticks actually dispatched.
    for pending in [0, 1, MAX_CATCH_UP_STEPS, MAX_CATCH_UP_STEPS + 1] {
        let dispatches = pending.clamp(1, MAX_CATCH_UP_STEPS);
        let advanced = pending.min(MAX_CATCH_UP_STEPS);
        assert_eq!(dispatches, advanced.max(1));
        for initial in [false, true] {
            let final_state = (0..dispatches).fold(initial, |state, _| !state);
            assert_eq!(final_state, initial ^ (dispatches % 2 != 0));
            if pending == 0 {
                assert_eq!(advanced, 0);
                assert_eq!(final_state, !initial);
            }
        }
    }
}
