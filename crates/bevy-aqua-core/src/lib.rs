#![warn(unreachable_pub)]

//! Shared Aqua authoring types and cross-crate render contracts.
//!
//! Application code should normally use the documented `bevy_aqua` root
//! facade. The normally documented items here are its authored water and
//! configuration contract. Items marked `doc(hidden)` remain public only so
//! Aqua's sibling feature crates can share GPU ABI, resources, and schedule
//! ordering; they are not an additional application-facing API tier.

#[doc(hidden)]
pub mod amortized_bake;
#[doc(hidden)]
pub mod bed;
mod body;
#[doc(hidden)]
pub mod cascade;
mod caustics;
#[doc(hidden)]
pub mod fields;
#[doc(hidden)]
pub mod pass;
#[doc(hidden)]
pub use bevy_aqua_geom::rings;
#[doc(hidden)]
pub mod view;
#[doc(hidden)]
pub mod waves_abi;

#[doc(hidden)]
pub use amortized_bake::AmortizedBake;
#[doc(inline)]
pub use bed::BedHeightMap;
#[doc(hidden)]
pub use bed::GpuFallback;
pub use body::WaterBody;
#[doc(hidden)]
pub use body::{ResolvedWaterBodies, ResolvedWaterBody, WaterBodyTransformError};
#[doc(hidden)]
pub use cascade::{
    BodyOptics, BodyParams, CascadeMaterial, Data, GpuCascade, GpuLayout, LOD_COUNT,
    PlanarReflectionParams, PlanarReflectionView, RESOLUTION, SurfaceParams, TILE_RESOLUTION,
    UpdateInputs, layout, lod_scale, make_detail_normal_texture, make_lod_scratch, make_texture,
    update as update_cascade_material,
};
#[doc(hidden)]
pub use caustics::{PATTERN_CAUSTIC_CELLS, caustic_green_srgb, write_pattern_caustics};
#[doc(hidden)]
pub use view::projected_detail_lod;

#[doc(hidden)]
pub use cascade::BASE_SCALE;
/// Registers every embedded WGSL module before any Aqua pipeline loads.
#[doc(hidden)]
pub use cascade::add_shader;
#[doc(hidden)]
pub use fields::{FIELD_LAYER_COUNT, FIELD_TEXTURE_FORMAT, FieldParams, MAX_BODIES, WaterFields};
#[doc(hidden)]
pub use rings::{Patch, Tile, build_patch, tile_layout};
#[doc(hidden)]
pub use view::{OceanView, ViewDetail, ViewOrder, ViewPos, ViewSeaLevel};
#[doc(hidden)]
pub use waves_abi::WAVE_SLOTS;
#[doc(hidden)]
pub use waves_abi::{AnimWavesUniform, AnimWavesUniformSlot, GpuWave};

pub use bevy_aqua_sdf::{FlowSample, RiverPath, RiverPoint, RiverSample, WaterShape};

use bevy::ecs::schedule::SystemSet;
use bevy::prelude::*;

/// Startup anchor: the umbrella has assembled the shared cascade [`Data`].
/// Feature plugins order their startup preparation after this set.
#[doc(hidden)]
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CascadeDataReady;

/// PostUpdate anchor: the umbrella has refreshed view tracking and the
/// cascade material uniforms. Producer updates order after this set.
#[doc(hidden)]
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CascadeMaterialsUpdated;

/// PostUpdate anchor: entity-local body shapes and propagated transforms have
/// been resolved into the shared world-space snapshot.
#[doc(hidden)]
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WaterBodiesResolved;

/// Render anchor: the AnimWaves cascades were written for this frame.
/// Consumers of the displacement textures (foam sim, wave query) order
/// their compute nodes after this set in `Core3d`.
#[doc(hidden)]
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AnimWavesWritten;

/// Whether the AnimWaves compute wrote the cascade textures this frame.
/// Render-side only; foam checks it before simulating so a skipped wave
/// pass skips foam too.
#[doc(hidden)]
#[derive(Resource, Debug, Default, Clone, Copy)]
pub struct AnimWavesStatus {
    pub written: bool,
}

/// Authored water optics for one body: Beer-Lambert extinction per channel,
/// particle-scatter scale and chromaticity, Henyey-Greenstein `g`, and crest
/// SSS tint. Localized bodies shade from `extinction`, `scatter_scale`,
/// `scatter_tint`, `scattering_asymmetry`, and `sun_roughness`.
#[derive(Component, Debug, Clone, Copy, PartialEq)]
pub struct WaterOptics {
    /// Per-channel extinction in inverse metres. Clear mountain water:
    /// red dies a little faster than green/blue, giving brown-green pools.
    pub extinction: Vec3,
    /// Particle-scatter multiplier for the underwater volume and the surface body.
    pub scatter_scale: f32,
    /// Per-channel particle scatter chromaticity.
    pub scatter_tint: Vec3,
    /// Henyey-Greenstein `g` for the shared medium. Higher values brighten
    /// looking toward the sun and darken looking away.
    pub scattering_asymmetry: f32,
    /// Surface roughness driving the Fresnel response; negative inherits
    /// the ocean value. Calm fresh water wants ~0.1 so grazing angles
    /// reflect the sky at near-Schlick strength instead of the damped
    /// ocean ceiling (~16% at 0.4).
    pub sun_roughness: f32,
    /// Sunlit subsurface scattering tint through pinched crests.
    pub sss_tint: Vec3,
}

impl WaterOptics {
    /// Accepted dark blue ocean profile: red-heavy extinction removes red
    /// first while the blue-heavy scatter keeps an ocean-blue body.
    pub const DEEP_OCEAN: Self = Self {
        extinction: Vec3::new(0.3, 0.1, 0.05),
        scatter_scale: 0.8,
        scatter_tint: Vec3::new(0.85, 1.0, 1.22),
        scattering_asymmetry: 0.8,
        sun_roughness: -1.0,
        sss_tint: Vec3::new(0.088_506_84, 0.497, 0.456_150_74),
    };
    /// Blue-green coastal profile; green survives longer than blue for a
    /// restrained teal.
    pub const COASTAL: Self = Self {
        extinction: Vec3::new(0.86, 0.24, 0.39),
        scatter_scale: 0.8,
        scatter_tint: Vec3::new(0.85, 1.0, 1.22),
        scattering_asymmetry: 0.8,
        sun_roughness: -1.0,
        sss_tint: Vec3::new(0.06, 0.55, 0.45),
    };
    /// Green-leading tropical profile; blue decays much sooner than green
    /// for a stylized tropical teal.
    pub const TROPICAL: Self = Self {
        extinction: Vec3::new(0.78, 0.14, 0.52),
        scatter_scale: 0.8,
        scatter_tint: Vec3::new(0.85, 1.0, 1.22),
        scattering_asymmetry: 0.8,
        sun_roughness: -1.0,
        sss_tint: Vec3::new(0.025, 0.62, 0.38),
    };
    /// Clear flowing fresh water over visible beds: transmission dominated
    /// by bed colour and depth, a touch of green only in deep pools, and a
    /// sky reflection that actually arrives at grazing angles.
    pub const CLEAR_FRESH: Self = Self {
        extinction: Vec3::new(0.24, 0.13, 0.10),
        scatter_scale: 0.4,
        sun_roughness: 0.1,
        ..Self::DEEP_OCEAN
    };

    /// Named presets in runtime cycle order.
    pub const PRESETS: [(&'static str, Self); 3] = [
        ("deep-ocean", Self::DEEP_OCEAN),
        ("coastal", Self::COASTAL),
        ("tropical", Self::TROPICAL),
    ];
}

/// The authoritative unbounded ocean configuration.
///
/// Insert this resource to enable the camera-centred ocean. Remove it for
/// bounded-water-only worlds. [`AquaSettings`] and [`OceanWaves`] remain the
/// authoritative global appearance and simulation resources.
#[derive(Resource, Debug, Default, Clone, Copy, PartialEq)]
pub struct Ocean {
    /// Undisplaced ocean surface level in world metres.
    pub level: f32,
}

/// Selects the producer behind Aqua's stable AnimWaves cascade interface.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum WaveModel {
    /// Crest-style deterministic Gerstner component bands.
    #[default]
    Analytic,
    /// Tessendorf inverse-FFT wave synthesis over the fetch-limited
    /// JONSWAP spectrum with directional spreading.
    Spectral,
}

/// Selects a preset displacement-energy level.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SeaState {
    /// Half-height waves for sheltered or calm presentation.
    Calm,
    /// The accepted reference wave energy.
    #[default]
    Moderate,
    /// One-and-a-half-height waves for rough presentation.
    Rough,
}

impl SeaState {
    /// Presets in increasing energy order.
    pub const ALL: [Self; 3] = [Self::Calm, Self::Moderate, Self::Rough];

    /// Returns the canonical command-line spelling.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Calm => "calm",
            Self::Moderate => "moderate",
            Self::Rough => "rough",
        }
    }

    /// Returns the displacement-amplitude multiplier.
    ///
    /// These are Aqua presentation presets, not Beaufort sea-state numbers.
    pub const fn amplitude_multiplier(self) -> f32 {
        match self {
            Self::Calm => 0.5,
            Self::Moderate => 1.0,
            Self::Rough => 1.5,
        }
    }
}

impl std::fmt::Display for SeaState {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Configures ocean displacement and wave synthesis.
///
/// Insert this resource before adding the Aqua plugins to replace its
/// defaults. The `model` and `shallow_water_attenuation` fields can change at
/// runtime. `sea_state` is sampled during startup because it determines
/// generated spectrum assets.
#[derive(Resource, Debug, Clone, Copy)]
pub struct OceanWaves {
    pub model: WaveModel,
    /// Startup-only displacement-energy preset.
    pub sea_state: SeaState,
    /// Strength of depth-driven shallow-water shoaling, clamped to `0..=1`.
    pub shallow_water_attenuation: f32,
    /// Wind direction in degrees, measured clockwise from world +X when seen
    /// from above. Both models align their spectra along this axis and
    /// spread components across the accepted directional variance around it.
    /// Startup-only: changing it after startup restarts nothing, so set it
    /// before the plugins run (like `sea_state`).
    pub wind_direction_degrees: f32,
    /// Spectral-model wind speed in metres per second, reshaping JONSWAP's
    /// peak frequency. Startup-only; the analytic model scales Crest's
    /// accepted amplitude curve via `sea_state` instead. Default reproduces
    /// the shipped spectrum (20 m/s).
    pub wind_speed: f32,
    /// Spectral-model fetch in metres (JONSWAP dimensionless fetch). Startup
    /// only. Default reproduces the shipped spectrum (100 km).
    pub fetch: f32,
    /// World-space current in metres per second. Wave content advects at this
    /// velocity (a sampling-space `x - flow * t` shift, which is exact Doppler
    /// advection for both Gerstner and FFT spectra) so rivers and tidal
    /// drifts read as moving water. Zero keeps today's anchored ocean.
    pub flow: Vec2,
}

impl Default for OceanWaves {
    fn default() -> Self {
        Self {
            model: WaveModel::Analytic,
            sea_state: SeaState::Moderate,
            shallow_water_attenuation: 0.95,
            wind_direction_degrees: 0.0,
            wind_speed: 20.0,
            fetch: 100_000.0,
            flow: Vec2::ZERO,
        }
    }
}

/// Supplies reflected radiance from the environment or mirrored scene views.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ReflectionMode {
    /// Sample only the existing environment cubemap.
    Cubemap,
    /// Render nearby scene geometry from mirrored cameras.
    Planar {
        /// Main-view resolution multiplier, clamped to `0.1..=1.0` (`0.5` default).
        scale: f32,
        /// Wave-normal UV offset, clamped to zero or greater (`0.02` default).
        distortion: f32,
    },
}

impl Default for ReflectionMode {
    fn default() -> Self {
        Self::Planar {
            scale: 0.5,
            distortion: 0.02,
        }
    }
}

/// Controls sunlight focused onto visible underwater beds.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Caustics {
    /// Radiance multiplier, clamped to zero or greater (`0.35` by default).
    pub strength: f32,
    /// Worley-cell width in metres, clamped to at least `0.01`.
    pub scale: f32,
    /// Pattern drift in metres per second; negative values reverse direction.
    pub speed: f32,
    /// Maximum lit bed depth in metres, clamped to zero or greater.
    pub depth_max: f32,
}

impl Default for Caustics {
    fn default() -> Self {
        Self {
            strength: 0.35,
            scale: 5.0,
            speed: 0.18,
            depth_max: 6.0,
        }
    }
}

/// Multiplies caustic sunlight by host visibility, clamped to `0..=1`.
#[derive(Resource, Debug, Clone, Copy, PartialEq)]
pub struct CausticsSunVisibility(pub f32);

impl Default for CausticsSunVisibility {
    fn default() -> Self {
        Self(1.0)
    }
}

/// Marks a camera that must not drive Aqua's main-view cascade tracking.
#[doc(hidden)]
#[derive(Component, Debug, Default, Clone, Copy)]
pub struct AuxiliaryWaterView;

/// Configures ocean-surface appearance.
///
/// Insert this resource before adding the Aqua plugins to replace the
/// defaults. Its fields can also be changed at runtime.
#[derive(Resource, Debug, Clone, Copy)]
pub struct AquaSettings {
    /// Overall detail-layer strength (`0.08` default, `0` disables, clamped
    /// to `0..=2`). Scales the Crest detail normals, the world-space
    /// capillary ripples (capped at `0.5`), and the unresolved-slope
    /// roughness multiplier (12.5x, capped at `2`) together, so the default
    /// reproduces the shipped look.
    pub detail_strength: f32,
    /// Selects a coherent water scattering and extinction preset.
    pub water_optics: WaterOptics,
    /// Filter raw directional sunlight through Bevy's atmosphere before it
    /// lights the water. Enable this when the light uses `lux::RAW_SUNLIGHT`;
    /// leave it disabled for already-filtered direct-light values.
    pub atmospheric_sunlight: bool,
    /// Far-shading start distance in metres, clamped to zero or greater.
    pub far_tier_start: f32,
    /// Full far-shading distance, clamped to at least one metre past the start.
    pub far_tier_end: f32,
    pub reflections: ReflectionMode,
    /// Procedural bed caustics; `None` disables both texture samples.
    pub caustics: Option<Caustics>,
}

impl Default for AquaSettings {
    fn default() -> Self {
        Self {
            detail_strength: 0.08,
            water_optics: WaterOptics::DEEP_OCEAN,
            atmospheric_sunlight: false,
            far_tier_start: 320.0,
            far_tier_end: 512.0,
            reflections: ReflectionMode::default(),
            caustics: Some(Caustics::default()),
        }
    }
}

/// Selects the ocean surface or an Aqua diagnostic output.
#[derive(Resource, Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum AquaDebug {
    /// Render the fully shaded ocean surface.
    #[default]
    Shaded,
    /// Render camera-space water path length as grayscale.
    WaterPath,
    /// Render valid refracted samples green and cancelled samples red.
    RefractionValidity,
    /// Render only the full-resolution refracted opaque background.
    Transmission,
    /// Render only the opaque background without a refraction offset.
    TransmissionUnrefracted,
    /// Render refracted transmission with Crest's Beer-Lambert depth fog.
    BeerLambert,
    /// Render the orthographic SeaFloorDepth cache.
    SeaFloorDepth,
    /// Render the shallow-water transmission, reflection, and attenuation composite.
    ShallowComposite,
    /// Render only Fresnel-weighted environment and sun reflection.
    ReflectionSanity,
    /// Render the persistent foam density before surface breakup.
    FoamDensity,
    /// Render FFT foam density with the legacy bilinear reconstruction for A/B diagnostics.
    FoamDensityBilinear,
    /// Render displaced surface height as signed grayscale.
    WaveHeight,
    /// Render pre-exposed primary-light RGB divided by 16 for direct readback probes.
    LightRadiance,
    /// Render the far-tier weight as grayscale from near (black) to far (white).
    FarTier,
    /// Render the final per-pixel Fresnel reflection fraction as grayscale.
    ReflectionFraction,
}
