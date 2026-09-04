//! Phase-mean vertical-height gradient energy of the live wave realization.
//! Bands are exclusive, fine-to-coarse; never upload cumulative values.
use bevy::prelude::*;
use bevy_aqua_core::{AnimWavesUniform, LOD_COUNT, cascade};

pub(crate) fn analytic(uniform: &AnimWavesUniform) -> [f32; LOD_COUNT] {
    std::array::from_fn(|band| {
        let range = uniform.ranges[band];
        uniform.waves[range.x as usize..range.y as usize]
            .iter()
            .map(|wave| {
                0.5 * (wave.amplitude * wave.wave_number).powi(2) * wave.direction.length_squared()
            })
            .sum()
    })
}

pub(crate) fn spectral(image: &Image, layout: &cascade::GpuLayout) -> [f32; LOD_COUNT] {
    let n = image.texture_descriptor.size.width;
    assert_eq!(image.texture_descriptor.size.height, n);
    assert_eq!(
        image.texture_descriptor.size.depth_or_array_layers,
        LOD_COUNT as u32
    );
    assert_eq!(
        image.texture_descriptor.format,
        bevy::render::render_resource::TextureFormat::Rgba32Float
    );
    let bytes = image
        .data
        .as_ref()
        .expect("H0 retains CPU bytes at generation");
    assert_eq!(bytes.len(), n as usize * n as usize * LOD_COUNT * 16);
    std::array::from_fn(|band| {
        let c = layout.cascades[band];
        spectral_layer(bytes, n, band, c.texel_width * c.texture_res) as f32
    })
}

fn spectral_layer(bytes: &[u8], n: u32, band: usize, period: f32) -> f64 {
    let signed = |v: u32| {
        if v <= n / 2 {
            v as f64
        } else {
            v as f64 - n as f64
        }
    };
    let dk = std::f64::consts::TAU / period as f64;
    let n4 = (n as f64).powi(4);
    (0..n * n)
        .map(|i| {
            let offset = (band * (n * n) as usize + i as usize) * 16;
            let re = f32::from_ne_bytes(bytes[offset..offset + 4].try_into().unwrap()) as f64;
            let im = f32::from_ne_bytes(bytes[offset + 4..offset + 8].try_into().unwrap()) as f64;
            let k2 = dk * dk * (signed(i % n).powi(2) + signed(i / n).powi(2));
            // Stockham is unnormalised; fft_resolve divides height by N².
            // Evolution uses h0(k)e^-iwt + conj(h0(-k))e^iwt.
            2.0 * k2 * (re * re + im * im) / n4
        })
        .sum()
}

// Runs after producer and material updates. Read-only comparison avoids AssetEvent::Modified
// on unchanged materials; scanning also covers assets added after startup.
pub(crate) fn upload(
    frame: Option<Res<crate::Frame>>,
    mut materials: ResMut<Assets<cascade::CascadeMaterial>>,
) {
    let packed = frame.as_ref().map_or([Vec4::ZERO; 2], |frame| {
        selected(
            frame.model,
            frame.analytic_variance,
            frame.spectral_variance,
        )
    });
    publish(&mut materials, packed);
}

fn selected(
    model: bevy_aqua_core::WaveModel,
    analytic: [f32; LOD_COUNT],
    spectral: [f32; LOD_COUNT],
) -> [Vec4; 2] {
    let bands = match model {
        bevy_aqua_core::WaveModel::Analytic => analytic,
        bevy_aqua_core::WaveModel::Spectral => spectral,
    };
    [
        Vec4::new(bands[0], bands[1], bands[2], bands[3]),
        Vec4::new(bands[4], 0.0, 0.0, 0.0),
    ]
}

fn publish(materials: &mut Assets<cascade::CascadeMaterial>, packed: [Vec4; 2]) -> usize {
    let changed: Vec<_> = materials
        .iter()
        .filter_map(|(id, material)| (material.surface.wave_slope_variance != packed).then_some(id))
        .collect();
    for &id in &changed {
        materials.get_mut(id).unwrap().surface.wave_slope_variance = packed;
    }
    changed.len()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn single_travelling_wave_exact_normalization() {
        // One stored coefficient H0=N²*A/2 creates A*cos(kx-wt).
        for n in [8u32, 16, 32] {
            let a = 0.75f32;
            let mut bytes = vec![0u8; (n * n) as usize * 16];
            bytes[16..20].copy_from_slice(&((n * n) as f32 * a / 2.0).to_ne_bytes());
            let actual = spectral_layer(&bytes, n, 0, std::f32::consts::TAU);
            assert!((actual - 0.5 * (a as f64).powi(2)).abs() < 1e-7);
        }
    }
    #[test]
    fn active_realization_and_amplitude_scaling() {
        let n = 32;
        let specs = [bevy_aqua_fft::BinSpec {
            texel_width: 1.0,
            texture_res: n as f32,
            max_wavelength: 16.0,
        }];
        let author = bevy_aqua_fft::SpectrumAuthoring::default();
        let field = bevy_aqua_fft::make_h0(n, &specs, 1.0, &author);
        let twice = bevy_aqua_fft::make_h0(n, &specs, 2.0, &author);
        let zero = bevy_aqua_fft::make_h0(n, &specs, 0.0, &author);
        let v = spectral_layer(&field.bytes, n, 0, 32.0);
        assert!(v > 0.0);
        assert!((spectral_layer(&twice.bytes, n, 0, 32.0) / v - 4.0).abs() < 1e-6);
        assert_eq!(spectral_layer(&zero.bytes, n, 0, 32.0), 0.0);
        let changed = bevy_aqua_fft::make_h0(
            n,
            &specs,
            1.0,
            &bevy_aqua_fft::SpectrumAuthoring {
                wind_speed: 4.0,
                fetch: 500.0,
                ..author
            },
        );
        assert!((spectral_layer(&changed.bytes, n, 0, 32.0) - v).abs() > 1e-6);
    }
    #[test]
    fn analytic_exclusive_ranges_and_scale() {
        let layout = cascade::GpuLayout::new(&cascade::layout(Vec2::ZERO), Vec2::ZERO, 0.0);
        // Use the production generator and actual selected wave-slot ranges.
        let u = crate::make_uniform(layout.clone(), 1.0, 0.0);
        let twice = crate::make_uniform(layout.clone(), 2.0, 0.0);
        assert_eq!(
            analytic(&crate::make_uniform(layout, 0.0, 0.0)),
            [0.0; LOD_COUNT]
        );
        let bands = analytic(&u);
        let all: f32 = u
            .waves
            .iter()
            .map(|w| 0.5 * (w.amplitude * w.wave_number).powi(2) * w.direction.length_squared())
            .sum();
        assert!((bands.iter().sum::<f32>() - all).abs() < 1e-6);
        for (a, b) in bands.into_iter().zip(analytic(&twice)) {
            assert!((b - 4.0 * a).abs() < 1e-6);
        }
    }

    #[test]
    fn model_switch_new_material_and_unchanged_upload() {
        let layout = cascade::GpuLayout::new(&cascade::layout(Vec2::ZERO), Vec2::ZERO, 0.0);
        let material = cascade::CascadeMaterial {
            texture: default(),
            layout,
            surface: default(),
            sea_floor: default(),
            detail_normal: default(),
            foam: default(),
            foam_pattern: default(),
            fft_surface: default(),
            fields: bevy_aqua_core::fields::FieldParams::none(),
            field_maps: default(),
            reflection_a: default(),
            reflection_b: default(),
            reflections: default(),
            caustics: default(),
        };
        assert_eq!(material.surface.wave_slope_variance, [Vec4::ZERO; 2]);
        let mut materials = Assets::default();
        let first = materials.add(material.clone());
        let a = selected(
            bevy_aqua_core::WaveModel::Analytic,
            [1.0; LOD_COUNT],
            [2.0; LOD_COUNT],
        );
        let s = selected(
            bevy_aqua_core::WaveModel::Spectral,
            [1.0; LOD_COUNT],
            [2.0; LOD_COUNT],
        );
        assert_eq!(a, [Vec4::ONE, Vec4::X]);
        assert_eq!(s, [Vec4::splat(2.0), Vec4::X * 2.0]);
        assert_eq!(publish(&mut materials, a), 1);
        assert_eq!(publish(&mut materials, a), 0);
        assert_eq!(publish(&mut materials, s), 1);
        let second = materials.add(material);
        assert_eq!(publish(&mut materials, s), 1);
        for handle in [&first, &second] {
            assert_eq!(
                materials.get(handle).unwrap().surface.wave_slope_variance,
                s
            );
        }
        assert_eq!(publish(&mut materials, [Vec4::ZERO; 2]), 2);
        assert_eq!(publish(&mut materials, [Vec4::ZERO; 2]), 0);
    }

    #[test]
    fn spectral_layers_are_exclusive_and_period_scales_squared() {
        let n = 8u32;
        let mut bytes = vec![0u8; (n * n) as usize * LOD_COUNT * 16];
        for band in 0..LOD_COUNT {
            let offset = (band * (n * n) as usize + 1) * 16;
            bytes[offset..offset + 4].copy_from_slice(&((band + 1) as f32).to_ne_bytes());
        }
        let base = spectral_layer(&bytes, n, 0, 8.0);
        for band in 0..LOD_COUNT {
            let v = spectral_layer(&bytes, n, band, 8.0);
            assert!((v / base - ((band + 1) as f64).powi(2)).abs() < 1e-12);
            assert!((spectral_layer(&bytes, n, band, 16.0) / v - 0.25).abs() < 1e-12);
        }
    }
}
