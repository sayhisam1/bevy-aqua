//! The AnimWaves uniform ABI, shared by the waves producer that fills it
//! and the WGSL consumers that bind it (`bevy_aqua_core::waves_sample`).

use bevy::ecs::resource::Resource;
use bevy::math::{UVec4, Vec2, Vec4};
use bevy::render::render_resource::ShaderType;

use crate::cascade::{GpuLayout, LOD_COUNT};

/// Fixed uniform wave-slot count; matches `WAVE_COUNT` in
/// `cascade/waves_sample.wgsl`.
pub const WAVE_SLOTS: usize = 40;

/// One Gerstner/FFT-band wave slot in the AnimWaves uniform.
#[derive(ShaderType, Clone, Copy, Debug, Default)]
pub struct GpuWave {
    /// Unit direction in world XZ.
    pub direction: Vec2,
    /// Vertical amplitude in metres.
    pub amplitude: f32,
    /// Deep-water wave number (rad/m).
    pub wave_number: f32,
    /// Deep-water angular frequency (rad/s).
    pub angular_frequency: f32,
    /// Initial phase (radians).
    pub phase: f32,
    /// Horizontal (choppy) amplitude in metres; negative pulls crests in.
    pub chop_amplitude: f32,
}

/// Render-world holder for the live AnimWaves uniform upload. The waves
/// producer writes it every frame; the query pass binds it without
/// depending on the waves crate.
#[derive(Resource, Default)]
pub struct AnimWavesUniformSlot(
    pub Option<bevy::render::render_resource::UniformBuffer<AnimWavesUniform>>,
);

impl std::fmt::Debug for AnimWavesUniformSlot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AnimWavesUniformSlot")
            .field("uploaded", &self.0.is_some())
            .finish()
    }
}

/// The live AnimWaves simulation uniform: one authoritative upload shared
/// by the wave compute passes and the query and volume passes. Layout must
/// stay aligned with `AnimWavesUniform` in `cascade/waves_sample.wgsl`.
#[derive(ShaderType, Clone, Debug)]
pub struct AnimWavesUniform {
    pub layout: GpuLayout,
    pub waves: [GpuWave; WAVE_SLOTS],
    /// Per-cascade `[start, end)` wave-slot ranges (`x`/`y`).
    pub ranges: [UVec4; LOD_COUNT],
    /// x: elapsed seconds; y: clamped shallow attenuation; z: enabled flag.
    pub time: Vec4,
    /// xy: world-space current in m/s; zw reserved.
    pub flow: Vec4,
}
