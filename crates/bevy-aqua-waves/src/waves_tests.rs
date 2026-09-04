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
fn gerstner_signs_match_crest() {
    let mut wave = generate_components(1.0, 0.0)[0];
    wave.direction = Vec2::X;
    wave.amplitude = 2.0;
    wave.chop_amplitude = -3.2;
    wave.phase = 0.0;
    wave.wave_number = 1.0;
    wave.angular_frequency = 1.0;

    assert_eq!(displacement(wave, Vec2::ZERO, 0.0), Vec3::Y * 2.0);
    let quarter_phase = displacement(wave, Vec2::ZERO, TAU / 4.0);
    assert!((quarter_phase - Vec3::new(-3.2, 0.0, 0.0)).length() < 0.000_001);
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
    // bin exactly like the pre-rotated vector (80, -60) * delta_k with
    // no wind; allow float slack for the rotation itself.
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
        80 + (RESOLUTION - 60) * RESOLUTION,
        &SpectrumAuthoring::default(),
    )
    .expect("bin active");
    assert_eq!(turned.k_length, plain.k_length);
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
    let angle = wave.wave_number * wave.direction.dot(position)
        + wave.phase
        + wave.angular_frequency * time;
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
