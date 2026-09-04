#![warn(unreachable_pub)]

//! Deterministic Tessendorf ocean-spectrum math and Stockham FFT shader source.
//!
//! The crate generates JONSWAP h0 fields, normalises their variance, computes
//! displacement bounds, and provides a CPU reference inverse transform. One
//! metre is one world unit.
//!
//! # Example
//!
//! ```
//! use bevy_aqua_fft::{BinSpec, SpectrumAuthoring, make_h0};
//!
//! let cascades = [BinSpec {
//!     texel_width: 1.0,
//!     texture_res: 256.0,
//!     min_wavelength: 2.0,
//!     max_wavelength: 4.0,
//! }];
//! let field = make_h0(256, &cascades, 1.0, &SpectrumAuthoring::default());
//! assert_eq!(field.layers, 1);
//! ```

use std::f32::consts::{PI, TAU};

/// Deep-water gravity acceleration in m/s².
pub const GRAVITY: f32 = 9.81;

/// Stockham transform shader with `horizontal` and `vertical` entry points.
pub const STOCKHAM_WGSL: &str = include_str!("../shaders/stockham.wgsl");

const TARGET_RMS_HEIGHT: f32 = 0.8;
const SEED: u32 = 0xA91C_5EED;

/// Turbulence fraction blended into the directional spread so cross-wind
/// components never die out completely.
pub const TURBULENCE: f32 = 0.145;

/// User-authorable JONSWAP spectrum parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpectrumAuthoring {
    /// Wind heading in radians, measured clockwise from world +X seen from
    /// above. The spectrum is authored along +X and rotated to match.
    pub wind_radians: f32,
    /// Wind speed in metres per second; drives JONSWAP's peak frequency.
    pub wind_speed: f32,
    /// Fetch length in metres (JONSWAP dimensionless-fetch input).
    pub fetch: f32,
}

impl Default for SpectrumAuthoring {
    fn default() -> Self {
        Self {
            wind_radians: 0.0,
            wind_speed: 20.0,
            fetch: 100_000.0,
        }
    }
}

/// FFT cascade dimensions and wavelength band.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BinSpec {
    /// World metres per texture texel.
    pub texel_width: f32,
    /// Texture side length in texels (the FFT period resolution).
    pub texture_res: f32,
    /// Inclusive shortest wavelength in metres. Independent of the upper bound
    /// so the coarsest cascade can retain its lower cutoff when extended.
    pub min_wavelength: f32,
    /// Exclusive longest wavelength in metres.
    pub max_wavelength: f32,
}

/// An active spectral bin before normalisation and amplitude scaling.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpectralBin {
    /// Wave number magnitude in rad/m.
    pub k_length: f32,
    /// Wavelength in metres.
    pub wavelength: f32,
    /// Variance density integrated over the bin area (m^2).
    pub raw_variance: f32,
}

/// Interleaved RGBA32-float h0 texels laid out layer by layer.
///
/// XY contain the complex coefficient and ZW are zero.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct H0Field {
    /// Raw texel bytes, `resolution * resolution * layers * 16`.
    pub bytes: Vec<u8>,
    /// Texture side length in texels.
    pub resolution: u32,
    /// Number of cascade slices.
    pub layers: u32,
}

/// Evaluates one flat texel index of a cascade slice as a JONSWAP bin.
///
/// Returns `None` for inactive bins: the DC/Nyquist rows and wavelengths
/// outside the cascade's `[min_wavelength, max_wavelength)` band.
pub fn spectral_bin(
    resolution: u32,
    cascade: BinSpec,
    flat_index: u32,
    authoring: &SpectrumAuthoring,
) -> Option<SpectralBin> {
    let index = glam::UVec2::new(flat_index % resolution, flat_index / resolution);
    let period = cascade.texel_width * cascade.texture_res;
    let delta_k = TAU / period;
    // The spectrum is authored along +X; rotating the wave vector evaluates
    // the same directional distribution around the requested wind. Length and
    // wavelength are rotation-invariant, so normalization is unaffected.
    let k = glam::Vec2::from_angle(authoring.wind_radians)
        .rotate(delta_k * signed_frequency(index, resolution).as_vec2());
    let k_length = k.length();
    let wavelength = TAU / k_length.max(f32::MIN_POSITIVE);
    let minimum = cascade.min_wavelength;
    let active = k_length > 0.0
        && index.x != resolution / 2
        && index.y != resolution / 2
        && wavelength >= minimum
        && wavelength < cascade.max_wavelength;
    active.then(|| SpectralBin {
        k_length,
        wavelength,
        raw_variance: jonswap_density(k, k_length, authoring) * delta_k * delta_k,
    })
}

/// Returns the scale that gives the spectrum its target RMS surface height.
pub fn spectrum_normalization(
    resolution: u32,
    cascades: &[BinSpec],
    authoring: &SpectrumAuthoring,
) -> f32 {
    let variance: f32 = cascades
        .iter()
        .copied()
        .flat_map(|cascade| {
            (0..resolution * resolution).filter_map(move |flat_index| {
                spectral_bin(resolution, cascade, flat_index, authoring)
            })
        })
        .map(|bin| bin.raw_variance)
        .sum();
    TARGET_RMS_HEIGHT.powi(2) / (2.0 * variance.max(f32::MIN_POSITIVE))
}

/// Generates one deterministic h0 coefficient slice per cascade.
pub fn make_h0(
    resolution: u32,
    cascades: &[BinSpec],
    amplitude_multiplier: f32,
    authoring: &SpectrumAuthoring,
) -> H0Field {
    let normalization = spectrum_normalization(resolution, cascades, authoring);
    let mut bytes = Vec::with_capacity(
        resolution as usize * resolution as usize * cascades.len() * 4 * size_of::<f32>(),
    );
    let transform_scale = (resolution as f32).powi(2);

    for (slice, cascade) in cascades.iter().copied().enumerate() {
        for flat_index in 0..resolution * resolution {
            let bin = spectral_bin(resolution, cascade, flat_index, authoring);
            let variance = bin.map_or(0.0, |bin| {
                bin.raw_variance * normalization * amplitude_multiplier.powi(2)
            });
            let gaussian = gaussian_pair(hash(SEED ^ ((slice as u32) << 24) ^ flat_index));
            let amplitude = transform_scale * (0.5 * variance).max(0.0).sqrt();
            let h0 = amplitude * gaussian;
            for value in [h0.x, h0.y, 0.0, 0.0] {
                bytes.extend_from_slice(&value.to_ne_bytes());
            }
        }
    }

    H0Field {
        bytes,
        resolution,
        layers: cascades.len() as u32,
    }
}

/// Phase-independent Fourier L1 displacement envelope per cascade,
/// accumulated coarse-to-fine: slice `i` bounds every band it owns plus all
/// finer ones combined into its AnimWaves output.
pub fn cumulative_height_bounds(
    resolution: u32,
    cascades: &[BinSpec],
    amplitude_multiplier: f32,
    authoring: &SpectrumAuthoring,
) -> Vec<f32> {
    let normalization = spectrum_normalization(resolution, cascades, authoring);
    let transform_scale = (resolution as f32).powi(2);
    let mut bands = vec![0.0_f64; cascades.len()];
    for (slice, cascade) in cascades.iter().copied().enumerate() {
        for flat_index in 0..resolution * resolution {
            let Some(bin) = spectral_bin(resolution, cascade, flat_index, authoring) else {
                continue;
            };
            let variance = bin.raw_variance * normalization * amplitude_multiplier.powi(2);
            let gaussian = gaussian_pair(hash(SEED ^ ((slice as u32) << 24) ^ flat_index));
            let h0 = transform_scale * (0.5 * variance).max(0.0).sqrt() * gaussian;
            // Evolution contains h0(k) and mirrored h0(-k). Summing all k
            // therefore contributes twice every stored coefficient.
            // Accumulate the actual f32 H0 realization in f64. Long-wave
            // support increases the dynamic range of this positive sum;
            // f32 accumulation can lose small terms and understate the bound.
            bands[slice] +=
                2.0 * f64::from(h0.x).hypot(f64::from(h0.y)) / f64::from(transform_scale);
        }
    }
    let mut cumulative = vec![0.0; cascades.len()];
    let mut coarser = 0.0;
    for (slice, band) in bands.iter().enumerate().rev() {
        coarser += band;
        // Round outward when publishing the conservative f32 envelope.
        cumulative[slice] = if coarser == 0.0 {
            0.0
        } else {
            (coarser as f32).next_up()
        };
    }
    cumulative
}

/// Applies the CPU reference inverse FFT.
///
/// The input length must be a power of two.
pub fn inverse_radix_two(values: &mut [glam::Vec2]) {
    let count = values.len();
    debug_assert!(count.is_power_of_two());
    for index in 0..count {
        let reversed = index.reverse_bits() >> (usize::BITS - count.ilog2());
        if index < reversed {
            values.swap(index, reversed);
        }
    }
    let mut span = 2;
    while span <= count {
        for block in (0..count).step_by(span) {
            for j in 0..span / 2 {
                let angle = TAU * j as f32 / span as f32;
                let rotated = values[block + j + span / 2].rotate(glam::Vec2::from_angle(angle));
                let a = values[block + j];
                values[block + j] = a + rotated;
                values[block + j + span / 2] = a - rotated;
            }
        }
        span *= 2;
    }
}

fn jonswap_density(k: glam::Vec2, k_length: f32, authoring: &SpectrumAuthoring) -> f32 {
    let direction = directional_spread(k.x / k_length);
    let wind_speed = authoring.wind_speed;
    let fetch = authoring.fetch;
    let omega = (GRAVITY * k_length).sqrt();
    let dimensionless_fetch = GRAVITY * fetch / wind_speed.powi(2);
    let alpha = 0.076 * dimensionless_fetch.powf(-0.22);
    let omega_peak = 22.0 * (GRAVITY.powi(2) / (wind_speed * fetch)).powf(1.0 / 3.0);
    let sigma: f32 = if omega <= omega_peak { 0.07 } else { 0.09 };
    let peak_shape =
        (-(omega - omega_peak).powi(2) / (2.0 * sigma.powi(2) * omega_peak.powi(2))).exp();
    let spectrum = alpha * GRAVITY.powi(2) / omega.powi(5)
        * (-1.25 * (omega_peak / omega).powi(4)).exp()
        * 3.3_f32.powf(peak_shape);
    let domega_dk = 0.5 * (GRAVITY / k_length).sqrt();
    spectrum * domega_dk / k_length * direction
}

fn directional_spread(cosine: f32) -> f32 {
    let forward = (2.0 / PI) * cosine.max(0.0).powi(2);
    (1.0 - TURBULENCE) * forward + TURBULENCE / TAU
}

fn signed_frequency(index: glam::UVec2, resolution: u32) -> glam::IVec2 {
    let signed = |value: u32| {
        if value <= resolution / 2 {
            value as i32
        } else {
            value as i32 - resolution as i32
        }
    };
    glam::IVec2::new(signed(index.x), signed(index.y))
}

fn hash(mut value: u32) -> u32 {
    value ^= value >> 16;
    value = value.wrapping_mul(0x7FEB_352D);
    value ^= value >> 15;
    value = value.wrapping_mul(0x846C_A68B);
    value ^ (value >> 16)
}

fn gaussian_pair(seed: u32) -> glam::Vec2 {
    let first = hash(seed).max(1);
    let second = hash(seed ^ 0x68BC_21EB);
    let u1 = (first as f32 + 0.5) / (u32::MAX as f32 + 1.0);
    let u2 = (second as f32 + 0.5) / (u32::MAX as f32 + 1.0);
    let radius = (-2.0 * u1.ln()).sqrt();
    let angle = TAU * u2;
    radius * glam::Vec2::new(angle.cos(), angle.sin())
}

#[cfg(test)]
mod tests;
