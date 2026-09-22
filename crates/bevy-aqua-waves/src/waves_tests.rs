use super::*;
use bevy_aqua_core::cascade as lod;

#[test]
fn generated_spectrum_is_stable_and_finite() {
    let waves = generate_components(1.0, 0.0);
    let wave = waves.iter().find(|wave| wave.wavelength >= 0.75).unwrap();
    assert!((wave.wavelength - 0.776_168).abs() < 0.000_001);
    assert!((wave.amplitude - 0.008_521).abs() < 0.000_001);
    assert!((wave.direction.length() - 1.0).abs() < 0.000_001);
    assert!(waves.iter().all(|wave| {
        wave.wavelength.is_finite()
            && wave.direction.is_finite()
            && wave.amplitude.is_finite()
            && wave.phase.is_finite()
    }));
}

#[test]
fn displacement_bounds_follow_active_wave_source() {
    let layout = lod::GpuLayout::new(&lod::layout(Vec2::ZERO), Vec2::ZERO, 0.0);
    let mut settings = OceanWaves::default();
    let gerstner = displacement_bounds(&settings, &layout, 1.0);
    assert!((gerstner[0].vertical - 1.980_311_6).abs() < 0.000_001);
    for bound in gerstner {
        assert_eq!(bound.horizontal, ANALYTIC_CHOP * bound.vertical);
    }
    assert!(
        gerstner
            .windows(2)
            .all(|pair| pair[0].vertical > pair[1].vertical)
    );

    settings.model = WaveModel::Spectral;
    let fft = displacement_bounds(&settings, &layout, 1.0);
    let unpadded = fft::cumulative_height_bounds(&layout, 1.0, &fft::SpectrumAuthoring::default());
    for (bound, source) in fft.into_iter().zip(unpadded) {
        assert!(
            bound.vertical >= 1.25 * source,
            "FFT bounds require 25% arithmetic/storage margin",
        );
        assert!(bound.vertical.is_finite() && bound.horizontal.is_finite());
        assert_eq!(bound.horizontal, FFT_CHOP * bound.vertical);
    }

    let moderate_startup_bounds = displacement_bounds(&settings, &layout, 1.0);
    settings.sea_state = bevy_aqua_core::SeaState::Calm;
    assert_eq!(
        displacement_bounds(&settings, &layout, 1.0),
        moderate_startup_bounds,
        "FFT bounds must follow the startup amplitude, not live sea state",
    );
}

#[test]
fn update_keeps_fft_uniform_and_dispatch_bins_in_lockstep() {
    let layout = lod::GpuLayout::new(&lod::layout(Vec2::ZERO), Vec2::ZERO, 0.0);
    let uniform = make_uniform(layout.clone(), 1.0, 0.0);
    let settings = OceanWaves {
        model: WaveModel::Spectral,
        ..default()
    };
    let frame = Frame {
        output: default(),
        surface: default(),
        raw: default(),
        scratch_a: default(),
        scratch_b: default(),
        h0: [default()],
        fft_state: [default(), default()],
        fft_scratch: [default(), default()],
        uniform,
        fft_uniform: fft::Uniform {
            layout: layout.clone(),
            params: Vec4::ZERO,
            mode: Vec4::new(fft::ATTENUATION_BINS as f32, 0.0, 0.0, 0.0),
        },
        model: WaveModel::Spectral,
        analytic_variance: [0.0; LOD_COUNT],
        spectral_variance: [0.0; 8],
        fft_bins: fft::ATTENUATION_BINS,
    };
    let mut app = App::new();
    app.insert_resource(Time::<()>::default())
        .insert_resource(lod::Data::new(default(), default(), default(), layout))
        .insert_resource(settings)
        .insert_resource(frame)
        .add_systems(Update, update);

    let assert_bins = |app: &App, expected: u32| {
        let frame = app.world().resource::<Frame>();
        assert_eq!(frame.fft_bins, expected);
        assert_eq!(frame.fft_uniform.mode.x, expected as f32);
    };

    // Startup without terrain must collapse both consumers in the same update.
    app.update();
    assert_bins(&app, 1);

    // Late bed insertion enables all attenuation bins in that update.
    app.world_mut().insert_resource(BedHeightMap {
        image: default(),
        origin: Vec2::ZERO,
        size: Vec2::ONE,
        height_range: [0.0, 1.0],
    });
    app.update();
    assert_bins(&app, fft::ATTENUATION_BINS);

    // Runtime attenuation changes exercise both transition directions.
    app.world_mut()
        .resource_mut::<OceanWaves>()
        .shallow_water_attenuation = 0.0;
    app.update();
    assert_bins(&app, 1);
    app.world_mut()
        .resource_mut::<OceanWaves>()
        .shallow_water_attenuation = 1.0;
    app.update();
    assert_bins(&app, fft::ATTENUATION_BINS);

    // Removing the bed returns both consumers to the one-bin path immediately.
    app.world_mut().remove_resource::<BedHeightMap>();
    app.update();
    assert_bins(&app, 1);
}

#[test]
fn gerstner_positive_direction_is_travel_direction() {
    let mut wave = generate_components(1.0, 0.0)[0];
    wave.direction = Vec2::X;
    wave.amplitude = 2.0;
    wave.chop_amplitude = -3.2;
    wave.phase = 0.0;
    wave.wave_number = 1.0;
    wave.angular_frequency = 1.0;

    assert_eq!(displacement(wave, Vec2::ZERO, 0.0), Vec3::Y * 2.0);
    let quarter_phase = displacement(wave, Vec2::ZERO, TAU / 4.0);
    assert!((quarter_phase - Vec3::new(3.2, 0.0, 0.0)).length() < 0.000_001);
}

#[test]
fn cascade_ranges_partition_bands_and_combine_downward() {
    let layout = lod::GpuLayout::new(&lod::layout(Vec2::ZERO), Vec2::ZERO, 0.0);
    let uniform = make_uniform(layout, 1.0, 0.0);
    assert_eq!(uniform.ranges.map(|range| range.x), [0, 8, 16, 24, 32]);
    assert_eq!(uniform.ranges.map(|range| range.y), [8, 16, 24, 32, 40]);
    for pair in uniform.ranges.windows(2) {
        assert_eq!(pair[0].y, pair[1].x);
    }
}

#[test]
fn wind_direction_rotates_gerstner_components() {
    let base = generate_components(1.0, 0.0);
    let rotated = generate_components(1.0, core::f32::consts::FRAC_PI_2);
    for (plain, turned) in base.iter().zip(rotated) {
        assert_eq!(plain.wavelength, turned.wavelength);
        assert!((plain.amplitude - turned.amplitude).abs() < 1e-7);
        let expected = Vec2::from_angle(
            core::f32::consts::FRAC_PI_2 + plain.direction.y.atan2(plain.direction.x),
        );
        assert!((turned.direction - expected).length() < 1e-6);
    }
}

#[test]
fn fft_spectrum_density_is_rotation_invariant() {
    use bevy_aqua_fft::{BinSpec, SpectrumAuthoring, spectral_bin};
    let layout = lod::GpuLayout::new(&lod::layout(Vec2::ZERO), Vec2::ZERO, 0.0);
    let cascade = layout.cascades[0];
    let spec = BinSpec {
        texel_width: cascade.texel_width,
        texture_res: cascade.texture_res,
        min_wavelength: 0.5 * cascade.max_wavelength,
        max_wavelength: cascade.max_wavelength,
    };
    const RESOLUTION: u32 = lod::RESOLUTION;
    // Wave vector (60, 80) * delta_k sits inside cascade zero's band
    // (wavelength 0.96 m). A minus-quarter-turn wind must evaluate that
    // travelling bin exactly like (-80, 60) * delta_k with no wind.
    // This sign makes authored wind headings actual travel headings.
    let turned = spectral_bin(
        RESOLUTION,
        spec,
        60 + 80 * RESOLUTION,
        &SpectrumAuthoring {
            wind_radians: -core::f32::consts::FRAC_PI_2,
            ..SpectrumAuthoring::default()
        },
    )
    .expect("bin active");
    let plain = spectral_bin(
        RESOLUTION,
        spec,
        (RESOLUTION - 80) + 60 * RESOLUTION,
        &SpectrumAuthoring::default(),
    )
    .expect("bin active");
    assert!((turned.k_length - plain.k_length).abs() < 1e-5);
    let scale = plain.raw_variance.abs().max(f32::MIN_POSITIVE);
    assert!((turned.raw_variance - plain.raw_variance).abs() / scale < 1e-4);
}

#[test]
fn uniform_flow_abi_matches_wgsl_declaration() {
    // One vec4 of advection after layout, waves, ranges, and time: xy is
    // the world-space current. The WGSL declarations must stay vec4 (not
    // vec2): naga and encase must agree on the member layout, and a
    // trailing vec2 made the query pass read misaligned data.
    use bevy::render::render_resource::encase::ShaderType;
    assert_eq!(<Uniform as ShaderType>::min_size().get(), 1_632);
    let cascades = bevy_aqua_core::cascade::layout(Vec2::ZERO);
    let uniform = Uniform {
        layout: bevy_aqua_core::GpuLayout::new(&cascades, Vec2::ZERO, 0.0),
        waves: [GpuWave::default(); GPU_WAVE_COUNT],
        ranges: [bevy::math::UVec4::ZERO; LOD_COUNT],
        time: Vec4::new(1.0, 2.0, 3.0, 4.0),
        flow: Vec4::new(7.0, 8.0, 9.0, 10.0),
    };
    let mut bytes = Vec::new();
    bevy::render::render_resource::encase::UniformBuffer::new(&mut bytes)
        .write(&uniform)
        .expect("uniform write");
    assert_eq!(bytes.len(), 1_632);
    let tail: [f32; 8] = std::array::from_fn(|index| {
        f32::from_le_bytes(
            bytes[1600 + index * 4..1604 + index * 4]
                .try_into()
                .unwrap(),
        )
    });
    assert_eq!(tail, [1.0, 2.0, 3.0, 4.0, 7.0, 8.0, 9.0, 10.0]);
}

fn displacement(wave: Component, position: Vec2, time: f32) -> Vec3 {
    let angle = wave.wave_number * wave.direction.dot(position) + wave.phase
        - wave.angular_frequency * time;
    let horizontal = wave.chop_amplitude * angle.sin();
    Vec3::new(
        horizontal * wave.direction.x,
        wave.amplitude * angle.cos(),
        horizontal * wave.direction.y,
    )
}

#[test]
fn sea_state_scales_both_backends_without_changing_bands() {
    let calm = generate_components(bevy_aqua_core::SeaState::Calm.amplitude_multiplier(), 0.0);
    let moderate = generate_components(
        bevy_aqua_core::SeaState::Moderate.amplitude_multiplier(),
        0.0,
    );
    let rough = generate_components(bevy_aqua_core::SeaState::Rough.amplitude_multiplier(), 0.0);
    for ((calm, moderate), rough) in calm.iter().zip(moderate).zip(rough) {
        assert_eq!(calm.wavelength, moderate.wavelength);
        assert_eq!(moderate.wavelength, rough.wavelength);
        assert!((calm.amplitude - 0.5 * moderate.amplitude).abs() < 1e-7);
        assert!((rough.amplitude - 1.5 * moderate.amplitude).abs() < 1e-7);
    }

    let layout = lod::GpuLayout::new(&lod::layout(Vec2::ZERO), Vec2::ZERO, 0.0);
    let calm = fft::cumulative_height_bounds(
        &layout,
        bevy_aqua_core::SeaState::Calm.amplitude_multiplier(),
        &fft::SpectrumAuthoring::default(),
    );
    let moderate = fft::cumulative_height_bounds(
        &layout,
        bevy_aqua_core::SeaState::Moderate.amplitude_multiplier(),
        &fft::SpectrumAuthoring::default(),
    );
    let rough = fft::cumulative_height_bounds(
        &layout,
        bevy_aqua_core::SeaState::Rough.amplitude_multiplier(),
        &fft::SpectrumAuthoring::default(),
    );
    for ((calm, moderate), rough) in calm.into_iter().zip(moderate).zip(rough) {
        assert!((calm / moderate - 0.5).abs() < 1e-5);
        assert!((rough / moderate - 1.5).abs() < 1e-5);
    }
}

#[test]
fn decorative_surface_detail_is_coherent_and_decorrelated() {
    let shader = include_str!("displace.wgsl");
    assert!(shader.contains(
        "flow_frame(advected_xz - heading_frame(DETAIL_TRAVEL_0) * globals.time * speed)"
    ));
    assert!(shader.contains(
        "flow_frame(advected_xz - heading_frame(DETAIL_TRAVEL_1) * globals.time * speed)"
    ));
    assert!(shader.contains("DETAIL_B_ROTATION"));
    assert!(shader.contains("DETAIL_B_SCALE: f32 = 1.41421356"));
    assert!(shader.contains("CAPILLARY_RESOLVED_STRENGTH: f32 = 0.45"));
    assert!(shader.contains("CAPILLARY_RESOLVED_ENERGY: f32"));
    assert!(shader.contains("0.70710678 * (sample_a.xy + sample_b.xy)"));
    assert!(shader.contains("0.5 * (sample_a.z + sample_b.z)"));
    assert!(!shader.contains("CAPILLARY_A_SCALE * base_stretch,\n    ).xy"));
    assert!(shader.contains("flow_frame(\n        advected_world(world_xz) - direction"));
    assert!(!shader.contains("NORMAL_DIRECTION_1: vec2<f32> = vec2(-0.85, -0.53)"));
}

#[test]
fn river_frame_transforms_world_heading_before_detail_sampling() {
    fn river_frame(point: Vec2, flow: Vec2) -> Vec2 {
        let speed = flow.length();
        let direction = flow / speed;
        Vec2::new(
            direction.dot(point) / (1.0 + 0.55 * speed.min(4.5)),
            Vec2::new(-direction.y, direction.x).dot(point) * 1.35,
        )
    }

    let point = Vec2::new(13.0, -7.0);
    let flow = Vec2::new(-0.8, 1.6);
    let heading = Vec2::new(0.91, 0.41).normalize();
    let world_step = 0.37 * heading;
    let actual = river_frame(point - world_step, flow) - river_frame(point, flow);
    let direction = flow.normalize();
    let expected = Vec2::new(
        -direction.dot(world_step) / (1.0 + 0.55 * flow.length().min(4.5)),
        -Vec2::new(-direction.y, direction.x).dot(world_step) * 1.35,
    );
    assert!((actual - expected).length() < 1e-6);

    let shader = include_str!("displace.wgsl");
    let motion = "flow_frame(advected_xz - heading_frame(DETAIL_TRAVEL_0) * globals.time * speed)";
    assert!(shader.contains(motion));
    assert!(!shader.contains("flow_frame(advected_xz) - direction_a"));
}

#[test]
fn exclusive_filter_is_bounded_and_leaves_geometry_unchanged() {
    let image = lod::make_fft_surface_texture();
    assert_eq!(image.texture_descriptor.mip_level_count, 9);
    assert_eq!(image.texture_descriptor.size.depth_or_array_layers, 25);
    assert!(image.data.is_none());
    let shader = include_str!("displace.wgsl");
    assert!(shader.contains("let layer = i32(lod_count() + 2u * field)"));
    assert!(shader.contains("let uv = fract(world_to_uv(advected_world(world_xz), cascade))"));
    assert!(shader.contains("textureNumLevels(fft_surface) - 1u"));
    let material = include_str!("../../bevy-aqua-core/src/cascade/material.wgsl");
    assert!(material.contains("if !bounded && surface.reflection.x > 0.5"));
    assert!(material.contains("length(dpdx(in.undisplaced_xz))"));
    assert!(material.contains("length(dpdy(in.undisplaced_xz))"));
    assert!(
        material.find("return shade_underside").unwrap()
            < material
                .find("filtered_fft_normal(in.undisplaced_xz,")
                .unwrap()
    );
    let deformation = include_str!("../../bevy-aqua-core/src/cascade/deform.wgsl");
    assert!(!deformation.contains("filtered_fft_normal"));
    let render = include_str!("render.rs");
    assert!(render.contains("LOD_COUNT as u32 + 2 * (4 + frame.fft_bins)"));
    assert!(render.contains("SURFACE_MIPS.iter().enumerate()"));
    let bytes: u32 = (0..=8).map(|m| (256u32 >> m).pow(2) * 25 * 8).sum();
    assert_eq!(bytes, 17_476_200);
    assert_eq!(5 + 2 * (4 + 1), 15);
    assert_eq!(5 + 2 * (4 + 4) + 4, 25);
}

#[test]
fn dense_mip_filter_preserves_constants_and_removes_nyquist() {
    let mean = |v: [f32; 4]| v.into_iter().sum::<f32>() * 0.25;
    assert_eq!(mean([0.3; 4]), 0.3);
    assert_eq!(mean([1.0, -1.0, -1.0, 1.0]), 0.0);
    let mip_texels: u32 = (1..=8).map(|mip| (lod::RESOLUTION >> mip).pow(2)).sum();
    assert_eq!(mip_texels, 21_845);
    assert!(mip_texels < lod::RESOLUTION.pow(2) / 3 + 1);
    let shader = include_str!("fft_surface_filter.wgsl");
    assert_eq!(shader.matches("textureLoad(source,").count(), 4);
    assert!(shader.contains("0.25 *"));
}

#[test]
fn decorative_frequency_and_energy_do_not_follow_mesh_lod() {
    let shader = include_str!("displace.wgsl");
    let start = shader.find("fn detail_normal_sample(").unwrap();
    let body = &shader[start..shader[start..].find("// GodotOceanWaves").unwrap() + start];
    assert!(body.contains("cascade_layout.cascades[0u]"));
    assert_eq!(body.matches("transformed_detail_normal(").count(), 2);
    assert!(!body.contains("mix(near, far"));
    let optics = include_str!("../../bevy-aqua-optics/src/optics.wgsl");
    assert!(optics.contains("let lod_blend_energy = 2.0;"));
}

#[test]
fn exclusive_derivatives_never_repeat_shoaling() {
    let shader = include_str!("fft_surface.wgsl");
    let begin = shader.find("fn spectral_displacement(").unwrap();
    let end = shader[begin..].find("@compute").unwrap() + begin;
    let body = &shader[begin..end];
    assert!(body.contains("textureLoad(height_x"));
    assert!(body.contains("textureLoad(z_field"));
    assert!(!body.contains("bed_range"));
    assert!(shader.contains("let field = id.z - LOD_COUNT"));
    assert!(shader.contains("% vec2(i32(FFT_RESOLUTION))"));
    assert!(shader.contains("vec4(dx, dx.y * dx.y)"));
    assert!(shader.contains("vec4(dz, dz.y * dz.y)"));
    let shade = include_str!("displace.wgsl");
    assert!(shade.contains("spectral_depth_weights(water_depth, sqrt(minimum * maximum))"));
    assert!(shade.contains("let shallow = (1.0 - smoothstep(4.0, 6.0, water_depth))"));
    assert!(shade.contains("small.z * local_outer.z"));
    assert!(shade.contains("i32(21u + field)"));
    assert!(shade.contains("2.0 * exp2(level) / cascade.texture_res"));
    assert!(!shade.contains("exp2(ceil(level))"));
}

#[test]
fn moment_filter_transfers_only_removed_slope_energy() {
    for samples in [[-0.4_f32, 0.1, 0.2, 0.5], [0.3; 4], [-1.0, 1.0, -1.0, 1.0]] {
        let mean = samples.into_iter().sum::<f32>() * 0.25;
        let second = samples.into_iter().map(|x| x * x).sum::<f32>() * 0.25;
        for retained in [0.0_f32, 0.2, 0.5, 1.0] {
            let resolved = retained * retained * mean * mean;
            let missing = (second - resolved).max(0.0);
            assert!((resolved + missing - second).abs() < 1e-7);
        }
    }
    let optics = include_str!("../../bevy-aqua-optics/src/optics.wgsl");
    assert!(optics.contains("unresolved_variance = get_spectral_filtered_variance()"));
    assert!(optics.contains("band_count = 0u"));
    assert!(!optics.contains("spectral_band_resolved_weight"));
    let shader = include_str!("displace.wgsl");
    let base = &shader[shader.find("fn filtered_fft_normal(").unwrap()
        ..shader
            .find("// Preserve the producer's accurate local short-wave")
            .unwrap()];
    let band_loop = &base[base.find("for (var field = 0u;").unwrap()..];
    assert!(!band_loop.contains("lod_alpha("));
    assert!(!base.contains("array<vec3<f32>"));
    assert!(base.contains("derivative_x += resolved_dx"));
    assert!(base.contains("max(dx.w + dz.w - resolved_energy, 0.0)"));
}

#[test]
fn sum_derivatives_before_forming_the_surface_normal() {
    let dx = [Vec3::new(0.1, 0.2, -0.1), Vec3::new(-0.05, -0.1, 0.02)];
    let dz = [Vec3::new(-0.1, 0.05, 0.1), Vec3::new(0.02, 0.1, -0.03)];
    let sum_dx: Vec3 = dx.into_iter().sum();
    let sum_dz: Vec3 = dz.into_iter().sum();
    let n = (Vec3::Z + sum_dz).cross(Vec3::X + sum_dx);
    assert!((n - Vec3::new(-0.119, 1.1171, -0.1655)).length() < 1e-5);
}

#[test]
fn packed_real_ifft_preserves_displacement_and_analytic_derivatives() {
    // Independent DFT reference: two Hermitian spectra share one complex IFFT.
    let n = 32usize;
    let mut height = vec![Vec2::ZERO; n];
    height[3] = Vec2::new(0.3, -0.2);
    height[n - 3] = Vec2::new(0.3, 0.2);
    let mut derivative = vec![Vec2::ZERO; n];
    for k in 0..n {
        let signed = if k < n / 2 {
            k as f32
        } else {
            k as f32 - n as f32
        };
        let wave_number = signed * std::f32::consts::TAU / n as f32;
        derivative[k] = wave_number * Vec2::new(-height[k].y, height[k].x);
    }
    let inverse_at = |field: &[Vec2], x: usize| -> Vec2 {
        field
            .iter()
            .enumerate()
            .map(|(k, h)| {
                let phase = std::f32::consts::TAU * (k * x) as f32 / n as f32;
                Vec2::new(
                    h.x * phase.cos() - h.y * phase.sin(),
                    h.x * phase.sin() + h.y * phase.cos(),
                )
            })
            .sum::<Vec2>()
            / n as f32
    };
    let packed: Vec<_> = height
        .iter()
        .zip(&derivative)
        .map(|(a, b)| *a + Vec2::new(-b.y, b.x))
        .collect();
    for x in 0..n {
        let both = inverse_at(&packed, x);
        assert!((both.x - inverse_at(&height, x).x).abs() < 1e-6);
        assert!((both.y - inverse_at(&derivative, x).x).abs() < 1e-6);
    }
    let resolve = include_str!("fft_resolve.wgsl");
    assert!(resolve.contains("vec3(packed.y, packed.x, packed.z) * normalization"));
    let surface = include_str!("fft_surface.wgsl");
    let overwrite = &surface[surface.find("fn resolve_analytic(").unwrap()
        ..surface.find("fn resolve_surface(").unwrap()];
    assert!(!overwrite.contains("textureStore(displacement"));
    assert!(surface.contains("LOD_COUNT + 2u * field"));
    assert!(surface.contains("vec3(b.y, a.w, b.z)"));
    assert!(surface.contains("vec3(b.z, b.x, b.w)"));
}

#[test]
fn shallow_residual_is_staged_before_analytic_overwrite_and_mips() {
    let render = include_str!("render.rs");
    let execution = &render[render
        .find("fn fft_spans(")
        .unwrap_or_else(|| render.find("fn fft_spans<").unwrap())..];
    let residual = execution
        .find("\"aqua_surface_shallow_prediction\"")
        .unwrap();
    let analytic = execution
        .find("\"aqua_surface_analytic_overwrite\"")
        .unwrap();
    let mips = execution.find("for (index, key) in SURFACE_MIPS").unwrap();
    assert!(residual < analytic && analytic < mips);
    assert!(
        execution[..residual]
            .rfind("if frame.fft_bins > 1 {")
            .is_some()
    );
    assert!(render.contains("array_layer_count: Some(21)"));
    assert!(render.contains("base_array_layer: 21"));
    assert!(render.contains("array_layer_count: Some(4)"));
    let shader = include_str!("fft_surface_residual.wgsl");
    assert!(shader.contains("vec4(increment, curvature)"));
    assert!(shader.contains("vec4(-60000.0), vec4(60000.0)"));
    assert!(shader.contains("bed_water_depth(world)"));
    assert!(!shader.contains("fract(uv)"));
}
