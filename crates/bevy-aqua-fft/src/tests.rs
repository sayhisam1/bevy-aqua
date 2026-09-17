use super::*;

#[derive(Clone, Copy, Debug, Default)]
struct LayerEnergy {
    expected: f64,
    realized: f64,
}

#[test]
fn inverse_radix_two_matches_direct_dft() {
    let input: Vec<glam::Vec2> = (0..8).map(|index| gaussian_pair(17 + index)).collect();
    let mut actual = input.clone();
    inverse_radix_two(&mut actual);
    for (sample, value) in actual.iter().enumerate() {
        let expected: glam::Vec2 = input
            .iter()
            .enumerate()
            .map(|(frequency, coefficient)| {
                let angle = TAU * (sample * frequency) as f32 / input.len() as f32;
                coefficient.rotate(glam::Vec2::from_angle(angle))
            })
            .fold(glam::Vec2::ZERO, |a, b| a + b);
        assert!((*value - expected).length() < 1e-5, "sample {sample}");
    }
}

fn default_cascades() -> Vec<BinSpec> {
    (0..5)
        .map(|lod| {
            let scale = 24.0 * 2.0_f32.powi(lod);
            let texel_width = 4.0 * scale / 256.0;
            BinSpec {
                texel_width,
                texture_res: 256.0,
                min_wavelength: 2.0 * texel_width,
                max_wavelength: 4.0 * texel_width,
            }
        })
        .collect()
}

fn layer_energy(
    field: &H0Field,
    cascades: &[BinSpec],
    amplitude_multiplier: f32,
    authoring: &SpectrumAuthoring,
) -> Vec<LayerEnergy> {
    let normalization = spectrum_normalization(field.resolution, cascades, authoring);
    let transform_scale = (field.resolution as f32).powi(2);
    let layer_texels = field.resolution as usize * field.resolution as usize;
    let mut energies = vec![LayerEnergy::default(); cascades.len()];

    for (texel, rgba) in field.bytes.as_chunks::<16>().0.iter().enumerate() {
        let slice = texel / layer_texels;
        let flat_index = (texel % layer_texels) as u32;
        let Some(bin) = spectral_bin(field.resolution, cascades[slice], flat_index, authoring)
        else {
            continue;
        };
        let x = f32::from_ne_bytes(rgba[0..4].try_into().unwrap());
        let y = f32::from_ne_bytes(rgba[4..8].try_into().unwrap());
        let variance = bin.raw_variance * normalization * amplitude_multiplier.powi(2);
        energies[slice].expected += f64::from(variance);
        energies[slice].realized += f64::from((x * x + y * y) / transform_scale.powi(2));
    }
    energies
}

#[test]
fn spectrum_authoring_reshapes_h0_deterministically() {
    let cascades = default_cascades();
    let short_fetch = SpectrumAuthoring {
        fetch: 50_000.0,
        ..SpectrumAuthoring::default()
    };
    let default_h0 = make_h0(256, &cascades, 1.0, &SpectrumAuthoring::default());
    let reshaped = make_h0(256, &cascades, 1.0, &short_fetch);
    let again = make_h0(256, &cascades, 1.0, &short_fetch);
    assert_ne!(default_h0.bytes, reshaped.bytes);
    assert_eq!(reshaped.bytes, again.bytes);
}

#[test]
fn fft_displacement_bounds_match_deterministic_h0() {
    let cascades = default_cascades();
    let field = make_h0(256, &cascades, 1.0, &SpectrumAuthoring::default());
    let mut expected = vec![0.0_f64; cascades.len()];
    for (index, rgba) in field.bytes.chunks_exact(16).enumerate() {
        let re = f32::from_ne_bytes(rgba[..4].try_into().unwrap()) as f64;
        let im = f32::from_ne_bytes(rgba[4..8].try_into().unwrap()) as f64;
        expected[index / (256 * 256)] += 2.0 * re.hypot(im) / (256 * 256) as f64;
    }
    for band in (0..cascades.len() - 1).rev() {
        expected[band] += expected[band + 1];
    }
    let actual = cumulative_height_bounds(256, &cascades, 1.0, &SpectrumAuthoring::default());
    for (actual, expected) in actual.into_iter().zip(expected) {
        assert!(actual.is_finite());
        assert!(actual as f64 >= expected, "{actual} understates {expected}");
        assert!(
            (actual as f64 - expected).abs() < 0.02,
            "{actual} != {expected}"
        );
    }
}

// Separate should_panic cases also work with the host's Cranelift test backend,
// where a panic may escape an in-test catch_unwind.
macro_rules! invalid_spec_test {
    ($name:ident, $field:ident, $value:expr, $message:literal) => {
        #[test]
        #[should_panic(expected = $message)]
        fn $name() {
            let spec = BinSpec {
                $field: $value,
                ..default_cascades()[0]
            };
            make_h0(8, &[spec], 1.0, &SpectrumAuthoring::default());
        }
    };
}

invalid_spec_test!(
    rejects_minimum_zero,
    min_wavelength,
    0.0,
    "min_wavelength must be finite and positive"
);
invalid_spec_test!(
    rejects_minimum_negative,
    min_wavelength,
    -1.0,
    "min_wavelength must be finite and positive"
);
invalid_spec_test!(
    rejects_minimum_nan,
    min_wavelength,
    f32::NAN,
    "min_wavelength must be finite and positive"
);
invalid_spec_test!(
    rejects_minimum_infinite,
    min_wavelength,
    f32::INFINITY,
    "min_wavelength must be finite and positive"
);
invalid_spec_test!(
    rejects_maximum_zero,
    max_wavelength,
    0.0,
    "max_wavelength must be finite and positive"
);
invalid_spec_test!(
    rejects_maximum_negative,
    max_wavelength,
    -1.0,
    "max_wavelength must be finite and positive"
);
invalid_spec_test!(
    rejects_maximum_nan,
    max_wavelength,
    f32::NAN,
    "max_wavelength must be finite and positive"
);
invalid_spec_test!(
    rejects_maximum_infinite,
    max_wavelength,
    f32::INFINITY,
    "max_wavelength must be finite and positive"
);
invalid_spec_test!(
    rejects_interval_equal,
    min_wavelength,
    1.5,
    "min_wavelength must be less than max_wavelength"
);
invalid_spec_test!(
    rejects_interval_reversed,
    min_wavelength,
    3.0,
    "min_wavelength must be less than max_wavelength"
);
invalid_spec_test!(
    rejects_texel_zero,
    texel_width,
    0.0,
    "FFT period must be finite and positive"
);
invalid_spec_test!(
    rejects_texel_negative,
    texel_width,
    -1.0,
    "FFT period must be finite and positive"
);
invalid_spec_test!(
    rejects_texel_nan,
    texel_width,
    f32::NAN,
    "FFT period must be finite and positive"
);
invalid_spec_test!(
    rejects_texel_infinite,
    texel_width,
    f32::INFINITY,
    "FFT period must be finite and positive"
);
invalid_spec_test!(
    rejects_texel_overflow,
    texel_width,
    f32::MAX,
    "FFT period must be finite and positive"
);
invalid_spec_test!(
    rejects_resolution_zero,
    texture_res,
    0.0,
    "FFT period must be finite and positive"
);
invalid_spec_test!(
    rejects_resolution_negative,
    texture_res,
    -1.0,
    "FFT period must be finite and positive"
);
invalid_spec_test!(
    rejects_resolution_nan,
    texture_res,
    f32::NAN,
    "FFT period must be finite and positive"
);
invalid_spec_test!(
    rejects_resolution_infinite,
    texture_res,
    f32::INFINITY,
    "FFT period must be finite and positive"
);

#[test]
#[should_panic(expected = "min_wavelength must be less than max_wavelength")]
fn normalization_rejects_invalid_interval() {
    let spec = BinSpec {
        min_wavelength: 3.0,
        ..default_cascades()[0]
    };
    spectrum_normalization(8, &[spec], &SpectrumAuthoring::default());
}

#[test]
#[should_panic(expected = "min_wavelength must be less than max_wavelength")]
fn bounds_reject_invalid_interval() {
    let spec = BinSpec {
        min_wavelength: 3.0,
        ..default_cascades()[0]
    };
    cumulative_height_bounds(8, &[spec], 1.0, &SpectrumAuthoring::default());
}

#[test]
fn generated_energy_tracks_analytic_curve() {
    let cascades = default_cascades();
    let authoring = SpectrumAuthoring::default();
    let field = make_h0(256, &cascades, 1.0, &authoring);
    let energies = layer_energy(&field, &cascades, 1.0, &authoring);
    assert!(energies.into_iter().all(|energy| {
        let ratio = energy.realized / energy.expected.max(f64::MIN_POSITIVE);
        (ratio - 1.0).abs() < 0.03
    }));
}
