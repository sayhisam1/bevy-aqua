//! Analytic Gerstner and spectral FFT wave production for Aqua's AnimWaves cascades.
//!
//! Settings come from [`bevy_aqua_core::OceanWaves`].
#![warn(unreachable_pub)]

mod fft;
mod render;

#[doc(hidden)]
pub use render::Prepared as RenderPrepared;

use std::f32::consts::TAU;

use bevy::{
    asset::embedded_asset,
    prelude::*,
    render::extract_resource::{ExtractResource, ExtractResourcePlugin},
};
use bevy_aqua_core::cascade as lod;
use bevy_aqua_core::{
    AnimWavesUniform as Uniform, CascadeDataReady, CascadeMaterialsUpdated, GpuWave, LOD_COUNT,
    WAVE_SLOTS,
};

use bevy_aqua_core::{BedHeightMap, OceanWaves, WaveModel};

#[derive(Resource)]
struct ShaderLibraries {
    _handles: Vec<Handle<Shader>>,
}

/// Adds Aqua's wave producers: analytic Gerstner bands and the JONSWAP/FFT
/// spectral path. Orders itself against the umbrella's cascade anchors
/// ([`bevy_aqua_core::CascadeDataReady`] and [`bevy_aqua_core::CascadeMaterialsUpdated`]).
#[derive(Debug, Default, Clone, Copy)]
pub struct AquaWavesPlugin;

impl Plugin for AquaWavesPlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "anim_waves.wgsl");
        embedded_asset!(app, "anim_waves_combine.wgsl");
        embedded_asset!(app, "anim_waves_gather.wgsl");
        embedded_asset!(app, "fft_evolve.wgsl");
        embedded_asset!(app, "fft_resolve.wgsl");
        embedded_asset!(app, "fft_surface.wgsl");
        // Surface module consumed by the composed cascade material.
        embedded_asset!(app, "displace.wgsl");
        let displace = app
            .world()
            .resource::<AssetServer>()
            .load("embedded://bevy_aqua_waves/displace.wgsl");
        app.insert_resource(ShaderLibraries {
            _handles: vec![displace],
        });
        let stockham = app
            .world_mut()
            .get_resource_mut::<Assets<Shader>>()
            .map(|mut shaders| {
                shaders.add(Shader::from_wgsl(
                    bevy_aqua_fft::STOCKHAM_WGSL,
                    "bevy-aqua-fft/stockham.wgsl",
                ))
            });
        if let Some(stockham) = stockham {
            // The render app needs the literal shader handle before
            // RenderStartup, so insert it there directly.
            app.add_plugins(render::WaveRenderPlugin(stockham));
        }
        app.add_plugins(ExtractResourcePlugin::<Frame>::default());
        app.add_systems(Startup, init.after(CascadeDataReady));
        app.add_systems(
            PostUpdate,
            update
                .after(CascadeMaterialsUpdated)
                .after(bevy_aqua_core::WaterBodiesResolved),
        );
    }
}

const SHADER_PATH: &str = "embedded://bevy_aqua_waves/anim_waves.wgsl";
const COMBINE_SHADER_PATH: &str = "embedded://bevy_aqua_waves/anim_waves_combine.wgsl";
const GATHER_SHADER_PATH: &str = "embedded://bevy_aqua_waves/anim_waves_gather.wgsl";
const FFT_EVOLVE_SHADER_PATH: &str = "embedded://bevy_aqua_waves/fft_evolve.wgsl";
const FFT_RESOLVE_SHADER_PATH: &str = "embedded://bevy_aqua_waves/fft_resolve.wgsl";
const FFT_SURFACE_SHADER_PATH: &str = "embedded://bevy_aqua_waves/fft_surface.wgsl";
const OCTAVE_COUNT: usize = 14;
const COMPONENTS_PER_OCTAVE: usize = 8;
const COMPONENT_COUNT: usize = OCTAVE_COUNT * COMPONENTS_PER_OCTAVE;
// The fixed uniform carries 40 wave slots (the generator must fill every
// band slot; see the M2/M3 partition in `Uniform::new`).
const GPU_WAVE_COUNT: usize = WAVE_SLOTS;
const SMALLEST_WAVELENGTH_POWER: i32 = -4;
const DIRECTION_VARIANCE_DEGREES: f32 = 90.0;
const WIND_SPEED_KPH: f32 = 150.0;
const KPH_PER_MPS: f32 = 3.6;
const GRAVITY: f32 = 9.81;
const ANALYTIC_CHOP: f32 = 1.6;
// Keep stochastic Tessendorf crests below the fold-prone analytic chop ratio.
const FFT_CHOP: f32 = 0.8;
// Eight Stockham stages, phase evolution, resolve, cascade combination, and
// repeated rgba16float storage sit above the analytic Fourier L1 envelope.
// Keep the accepted 25% numeric/storage guard until a directed error proof
// justifies a smaller value.
const FFT_DISPLACEMENT_BOUND_PADDING: f32 = 1.25;
const MIN_AMPLITUDE: f32 = 0.001;
const WORKGROUP_SIZE: u32 = 8;

// Crest OceanWaveSpectrum.cs defaults before its amplitude-neutral v1 calibration.
const POWER_LOG10: [f32; OCTAVE_COUNT] = [
    -5.71, -5.03, -4.54, -3.88, -3.28, -2.32, -1.78, -1.21, -0.54, 0.28, 0.54, 1.03, 1.44, -8.0,
];

/// Maximum world-space displacement used to expand render bounds.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct DisplacementBounds {
    /// Largest horizontal (choppy) offset in metres.
    pub horizontal: f32,
    /// Largest vertical offset in metres.
    pub vertical: f32,
}

/// The startup sea-state amplitude multiplier; render bounds depend on it
/// until settings change the model.
#[doc(hidden)]
#[derive(Resource, Clone, Copy, Debug)]
pub struct StartupAmplitude(pub f32);

/// Returns cumulative displacement bounds for each cascade.
///
/// Each bound covers its owned band and every coarser band combined into its
/// AnimWaves slice. Gerstner bounds use active amplitude sums. FFT bounds use
/// a padded, phase-independent Fourier L1 envelope.
pub fn displacement_bounds(
    settings: &OceanWaves,
    layout: &lod::GpuLayout,
    amplitude_multiplier: f32,
) -> [DisplacementBounds; LOD_COUNT] {
    match settings.model {
        WaveModel::Analytic => {
            let components = generate_components(
                amplitude_multiplier,
                settings.wind_direction_degrees.to_radians(),
            );
            std::array::from_fn(|lod| {
                let minimum = 0.5 * layout.cascades[lod].max_wavelength;
                let maximum = layout.cascades[LOD_COUNT - 1].max_wavelength;
                let vertical = components
                    .iter()
                    .filter(|wave| wave.wavelength >= minimum && wave.wavelength < maximum)
                    .map(|wave| wave.amplitude.abs())
                    .sum::<f32>();
                DisplacementBounds {
                    horizontal: ANALYTIC_CHOP * vertical,
                    vertical,
                }
            })
        }
        WaveModel::Spectral => fft::cumulative_height_bounds(
            layout,
            amplitude_multiplier,
            &fft::SpectrumAuthoring {
                wind_radians: settings.wind_direction_degrees.to_radians(),
                wind_speed: settings.wind_speed,
                fetch: settings.fetch,
            },
        )
        .map(|vertical| {
            let vertical = FFT_DISPLACEMENT_BOUND_PADDING * vertical;
            DisplacementBounds {
                horizontal: FFT_CHOP * vertical,
                vertical,
            }
        }),
    }
}

fn make_uniform(
    layout: bevy_aqua_core::GpuLayout,
    amplitude_multiplier: f32,
    wind_radians: f32,
) -> Uniform {
    let components = generate_components(amplitude_multiplier, wind_radians);
    let selected: [Component; GPU_WAVE_COUNT] = components
        .into_iter()
        .filter(|wave| {
            wave.wavelength >= layout.cascades[0].max_wavelength * 0.5
                && wave.wavelength < layout.cascades[LOD_COUNT - 1].max_wavelength
        })
        .collect::<Vec<_>>()
        .try_into()
        .expect("Crest spectrum must fill every M3 wave band");

    let waves = selected.map(GpuWave::from);
    // The fixed M2 layout has no altitude scale transition. Each raw slice
    // therefore owns one exclusive Crest wavelength band.
    let ranges = std::array::from_fn(|lod| {
        let maximum = layout.cascades[lod].max_wavelength;
        let minimum = 0.5 * maximum;
        let start = selected.partition_point(|wave| wave.wavelength < minimum);
        let end = selected.partition_point(|wave| wave.wavelength < maximum);
        bevy::math::UVec4::new(start as u32, end as u32, 0, 0)
    });
    Uniform {
        layout,
        waves,
        ranges,
        time: Vec4::ZERO,
        flow: Vec4::ZERO,
    }
}

/// Generates the h0 spectrum texture and the initial frame resources.
/// Runs once at startup, after the umbrella assembled the cascade [`lod::Data`].
pub fn init(
    mut commands: Commands,
    data: Res<lod::Data>,
    settings: Res<OceanWaves>,
    mut images: ResMut<Assets<Image>>,
) {
    let authoring = fft::SpectrumAuthoring {
        wind_radians: settings.wind_direction_degrees.to_radians(),
        wind_speed: settings.wind_speed,
        fetch: settings.fetch,
    };
    let h0_jonswap = fft::make_h0(
        data.layout(),
        settings.sea_state.amplitude_multiplier(),
        &authoring,
    );
    commands.insert_resource(StartupAmplitude(settings.sea_state.amplitude_multiplier()));
    commands.insert_resource(Frame {
        output: data.texture(),
        raw: images.add(lod::make_lod_scratch()),
        scratch_a: images.add(lod::make_lod_scratch()),
        scratch_b: images.add(lod::make_lod_scratch()),
        h0: [images.add(h0_jonswap)],
        fft_state: std::array::from_fn(|_| images.add(fft::make_field_texture())),
        fft_scratch: std::array::from_fn(|_| images.add(fft::make_field_texture())),
        uniform: make_uniform(
            data.layout().clone(),
            settings.sea_state.amplitude_multiplier(),
            settings.wind_direction_degrees.to_radians(),
        ),
        fft_uniform: fft::Uniform {
            layout: data.layout().clone(),
            params: Vec4::new(0.0, settings.shallow_water_attenuation, 1.0, 0.0),
            // Startup uses the conservative count: the game's bed map (when
            // any) is inserted during the first frames, and a persistent
            // seeded texture (foam) must never observe a transient one-bin
            // frame near terrain.
            mode: Vec4::new(fft::ATTENUATION_BINS as f32, 0.0, 0.0, 0.0),
        },
        model: settings.model,
        // Conservative until the first runtime gate evaluation (see `mode`).
        fft_bins: fft::ATTENUATION_BINS,
    });
}

/// Refreshes per-frame uniforms (clock, current, attenuation gate) from
/// settings and the live cascade layout.
#[allow(clippy::too_many_arguments)]
pub fn update(
    time: Res<Time>,
    data: Res<lod::Data>,
    settings: Res<OceanWaves>,
    bed: Option<Res<BedHeightMap>>,
    mut frame: ResMut<Frame>,
) {
    frame.uniform.layout = data.layout().clone();
    frame.uniform.time.x = time.elapsed_secs();
    frame.uniform.flow = Vec4::new(settings.flow.x, settings.flow.y, 0.0, 0.0);
    frame.uniform.time.y = settings.shallow_water_attenuation.clamp(0.0, 1.0);
    frame.uniform.time.z = 1.0;
    frame.fft_uniform.layout = data.layout().clone();
    frame.fft_uniform.params.x = time.elapsed_secs();
    frame.fft_uniform.params.y = settings.shallow_water_attenuation.clamp(0.0, 1.0);
    frame.fft_uniform.params.z = 1.0;
    frame.fft_uniform.mode.x = frame.fft_bins as f32;
    frame.model = settings.model;
    frame.fft_bins = fft::active_bin_count(settings.shallow_water_attenuation, bed.is_none());
}

/// Per-frame wave simulation resources: textures, uniforms, model gates.
#[doc(hidden)]
#[derive(Resource, Clone, ExtractResource, Debug)]
pub struct Frame {
    output: Handle<Image>,
    raw: Handle<Image>,
    scratch_a: Handle<Image>,
    scratch_b: Handle<Image>,
    h0: [Handle<Image>; 1],
    fft_state: [Handle<Image>; 2],
    fft_scratch: [Handle<Image>; 2],
    uniform: Uniform,
    fft_uniform: fft::Uniform,
    model: WaveModel,
    // Active FFT attenuation-bin count per cascade (1 or ATTENUATION_BINS).
    fft_bins: u32,
}

impl Frame {
    /// The live simulation uniform shared with the query pass.
    pub fn uniform(&self) -> &bevy_aqua_core::AnimWavesUniform {
        &self.uniform
    }

    pub fn output(&self) -> Handle<Image> {
        self.output.clone()
    }
}

#[derive(Clone, Copy, Debug)]
struct Component {
    wavelength: f32,
    direction: Vec2,
    amplitude: f32,
    wave_number: f32,
    angular_frequency: f32,
    phase: f32,
    chop_amplitude: f32,
}

impl From<Component> for GpuWave {
    fn from(wave: Component) -> Self {
        Self {
            direction: wave.direction,
            amplitude: wave.amplitude,
            wave_number: wave.wave_number,
            angular_frequency: wave.angular_frequency,
            phase: wave.phase,
            chop_amplitude: wave.chop_amplitude,
        }
    }
}

fn generate_components(
    amplitude_multiplier: f32,
    wind_radians: f32,
) -> [Component; COMPONENT_COUNT] {
    let mut random = Random::new(0);
    let mut wavelengths = [0.0; COMPONENT_COUNT];
    let mut angles = [0.0; COMPONENT_COUNT];

    for octave in 0..OCTAVE_COUNT {
        let base = 2.0_f32.powi(SMALLEST_WAVELENGTH_POWER + octave as i32);
        for component in 0..COMPONENTS_PER_OCTAVE {
            let index = octave * COMPONENTS_PER_OCTAVE + component;
            let fraction = component as f32 / COMPONENTS_PER_OCTAVE as f32;
            let minimum = base * (1.0 + fraction);
            let maximum = (minimum + base / COMPONENTS_PER_OCTAVE as f32).min(2.0 * base);
            wavelengths[index] = minimum.lerp(maximum, random.next());

            let direction_fraction =
                (component as f32 + random.next()) / COMPONENTS_PER_OCTAVE as f32;
            angles[index] = (2.0 * direction_fraction - 1.0) * DIRECTION_VARIANCE_DEGREES;
        }
    }

    let mut amplitudes = [0.0; COMPONENT_COUNT];
    for (amplitude, wavelength) in amplitudes.iter_mut().zip(wavelengths) {
        *amplitude = random.next() * spectrum_amplitude(wavelength);
        if *amplitude < MIN_AMPLITUDE {
            *amplitude = 0.0;
        }
        *amplitude *= amplitude_multiplier;
    }

    let mut phase_random = Random::new(0);
    std::array::from_fn(|index| {
        let wavelength = wavelengths[index];
        let direction = Vec2::from_angle(wind_radians + angles[index].to_radians());
        let amplitude = amplitudes[index];
        let wave_number = TAU / wavelength;
        let phase = TAU * ((index % COMPONENTS_PER_OCTAVE) as f32 + phase_random.next())
            / COMPONENTS_PER_OCTAVE as f32;
        Component {
            wavelength,
            direction,
            amplitude,
            wave_number,
            angular_frequency: (GRAVITY * wave_number).sqrt(),
            phase,
            chop_amplitude: -ANALYTIC_CHOP * amplitude,
        }
    })
}

fn spectrum_amplitude(wavelength: f32) -> f32 {
    let power = wavelength.log2().clamp(
        SMALLEST_WAVELENGTH_POWER as f32,
        (SMALLEST_WAVELENGTH_POWER + OCTAVE_COUNT as i32 - 1) as f32,
    );
    let lower_wavelength = 2.0_f32.powf(power.floor());
    let octave = (power - SMALLEST_WAVELENGTH_POWER as f32) as usize;
    let alpha = (wavelength - lower_wavelength) / lower_wavelength;
    let next = (octave + 1).min(OCTAVE_COUNT - 1);
    let log_power = POWER_LOG10[octave].lerp(POWER_LOG10[next], alpha);
    let mut spectral_power = 10.0_f32.powf(log_power);

    let omega_lower = (GRAVITY * TAU / lower_wavelength).sqrt();
    let omega_upper = (GRAVITY * TAU / (2.0 * lower_wavelength)).sqrt();
    let delta_omega = (omega_lower - omega_upper) / COMPONENTS_PER_OCTAVE as f32;
    let omega = (GRAVITY * TAU / wavelength).sqrt();
    let wind_frequency = 0.87 * GRAVITY / (WIND_SPEED_KPH / KPH_PER_MPS);
    spectral_power *= (-1.291 * (wind_frequency / omega).powi(4)).exp();

    (2.0 * spectral_power * delta_omega).sqrt()
}

// Deterministic local substitute for Unity Random. Exact Unity bit parity is not required.
#[derive(Clone, Copy)]
struct Random(u32);

impl Random {
    const fn new(seed: u32) -> Self {
        Self(seed)
    }

    fn next(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (self.0 >> 8) as f32 / 16_777_216.0
    }
}

#[cfg(test)]
#[path = "waves_tests.rs"]
mod tests;
