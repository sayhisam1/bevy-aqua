use bevy::{
    asset::RenderAssetUsages,
    prelude::*,
    render::render_resource::{
        Extent3d, ShaderType, TextureDimension, TextureFormat, TextureUsages,
        TextureViewDescriptor, TextureViewDimension,
    },
};

use bevy_aqua_fft::BinSpec;
pub(crate) use bevy_aqua_fft::SpectrumAuthoring;

pub(crate) const RESOLUTION: u32 = lod::RESOLUTION;
// Log-frequency bins per cascade used for local shallow-water attenuation.
// The final cascade spans five octaves; its four bins are wider.
pub(crate) const ATTENUATION_BINS: u32 = 4;
pub(crate) const FIELD_LAYERS: u32 = LOD_COUNT as u32 * ATTENUATION_BINS;

use bevy_aqua_core::LOD_COUNT;
use bevy_aqua_core::cascade as lod;

pub(crate) fn make_h0(
    layout: &lod::GpuLayout,
    amplitude_multiplier: f32,
    authoring: &SpectrumAuthoring,
) -> Image {
    let field = bevy_aqua_fft::make_h0(
        RESOLUTION,
        &cascade_specs(layout),
        amplitude_multiplier,
        authoring,
    );
    let mut image = Image::new(
        Extent3d {
            width: RESOLUTION,
            height: RESOLUTION,
            depth_or_array_layers: field.layers,
        },
        TextureDimension::D2,
        field.bytes,
        TextureFormat::Rgba32Float,
        RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
    );
    image.texture_descriptor.usage = TextureUsages::COPY_DST | TextureUsages::TEXTURE_BINDING;
    image.texture_view_descriptor = Some(TextureViewDescriptor {
        dimension: Some(TextureViewDimension::D2Array),
        ..default()
    });
    image
}

pub(crate) fn cumulative_height_bounds(
    layout: &lod::GpuLayout,
    amplitude_multiplier: f32,
    authoring: &SpectrumAuthoring,
) -> [f32; LOD_COUNT] {
    let bounds = bevy_aqua_fft::cumulative_height_bounds(
        RESOLUTION,
        &cascade_specs(layout),
        amplitude_multiplier,
        authoring,
    );
    std::array::from_fn(|lod| bounds[lod])
}

fn cascade_specs(layout: &lod::GpuLayout) -> Vec<BinSpec> {
    layout.cascades[..LOD_COUNT]
        .iter()
        .enumerate()
        .map(|(band, cascade)| BinSpec {
            texel_width: cascade.texel_width,
            texture_res: cascade.texture_res,
            min_wavelength: 0.5 * cascade.max_wavelength,
            // Keep swell below one quarter of the periodic tile so its longest
            // waves repeat at least four times, limiting tile-scale repetition.
            // This exclusive cutoff must match fft_evolve.wgsl, fft_resolve.wgsl,
            // and variance::spectral's long-band partition. Core/analytic octave
            // metadata and texture periods stay unchanged.
            max_wavelength: if band == LOD_COUNT - 1 {
                cascade.texel_width * cascade.texture_res / 4.0
            } else {
                cascade.max_wavelength
            },
        })
        .collect()
}

#[derive(ShaderType, Clone, Debug)]
pub(crate) struct Uniform {
    pub(crate) layout: lod::GpuLayout,
    // time, shallow attenuation, waves enabled, reserved.
    pub(crate) params: Vec4,
    // x: active attenuation-bin count (1 or ATTENUATION_BINS), y-z reserved.
    pub(crate) mode: Vec4,
}

// Logarithmic bins apply depth-driven shoaling across each spectral band.
// Without shoaling or a bed map, each bin is exactly vec2(1.0), so one bin is
// bit-equivalent.
pub(crate) fn active_bin_count(shallow_water_attenuation: f32, terrain_absent: bool) -> u32 {
    if shallow_water_attenuation <= 0.0 || terrain_absent {
        1
    } else {
        ATTENUATION_BINS
    }
}

pub(crate) fn make_field_texture() -> Image {
    let pixel_count = RESOLUTION as usize * RESOLUTION as usize * FIELD_LAYERS as usize;
    let mut image = Image::new(
        Extent3d {
            width: RESOLUTION,
            height: RESOLUTION,
            depth_or_array_layers: FIELD_LAYERS,
        },
        TextureDimension::D2,
        vec![0; pixel_count * 4 * size_of::<f32>()],
        TextureFormat::Rgba32Float,
        RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
    );
    image.texture_descriptor.usage =
        TextureUsages::TEXTURE_BINDING | TextureUsages::STORAGE_BINDING;
    image.texture_view_descriptor = Some(TextureViewDescriptor {
        dimension: Some(TextureViewDimension::D2Array),
        ..default()
    });
    image
}

#[cfg(test)]
#[path = "fft_tests.rs"]
mod tests;
