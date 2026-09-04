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
    let expected = [252.608_06, 244.313_5, 227.737_76, 194.751_74, 129.361_68];
    let actual = cumulative_height_bounds(256, &cascades, 1.0, &SpectrumAuthoring::default());
    for (actual, expected) in actual.into_iter().zip(expected) {
        assert!((actual - expected).abs() < 0.02, "{actual} != {expected}");
    }
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
