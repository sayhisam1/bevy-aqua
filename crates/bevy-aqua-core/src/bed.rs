//! The bed height map: one static terrain heightfield water reads for
//! shoaling, shoreline foam, and underwater optics.
//!
//! Games publish a [`BedHeightMap`] built from their terrain heightfield;
//! terra publishes one straight from its heightmap. A plain ocean needs
//! none: without the resource every sample falls outside the mapped area
//! and water keeps the deep default. Props, rocks, and piers keep their
//! screen-space depth foam/darkening path.

use bevy::{
    asset::RenderAssetUsages,
    prelude::*,
    render::{
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        render_asset::RenderAssets,
        render_resource::{Extent3d, TextureDimension, TextureFormat},
        texture::GpuImage,
    },
};

/// A negative decode span marks "no bed map" inside the shared uniforms:
/// every shader sample then takes the deep-water path untouched.
pub(crate) const NO_BED_SPAN: f32 = -1.0;

/// Depth every shader reports where the bed map has no data. Keep aligned
/// with the per-shader `NO_BED_DEPTH` constants.
#[cfg_attr(not(test), allow(dead_code))]
pub const NO_BED_DEPTH: f32 = 256.0;

/// A static bed heightfield supplied by the game.
///
/// The image is single-channel: the red channel stores height normalised
/// into [`BedHeightMap::height_range`]. Texel (0, 0) is the minimum corner:
/// world [`BedHeightMap::origin`] is the centre of that texel, and
/// [`BedHeightMap::size`] is the distance from the first texel centre to the
/// last, not the distance between the outer pixel edges. For an `N`-texel
/// axis with spacing `step`, this distance is `(N - 1) * step`.
#[derive(Resource, Debug, Clone, ExtractResource)]
pub struct BedHeightMap {
    /// Single-channel image; red channel holds normalised height in [0, 1].
    pub image: Handle<Image>,
    /// World-space XZ of the first texel's centre (the minimum corner).
    pub origin: Vec2,
    /// World-space XZ distance between the first and last texel centres.
    pub size: Vec2,
    /// Metres decoded from normalised red: `[minimum, maximum]`.
    pub height_range: [f32; 2],
}

impl BedHeightMap {
    /// Bakes a bed map from a world-XZ height function over a square grid.
    /// `origin` is the centre of texel (0, 0); `step` is the spacing between
    /// texel centres in metres.
    pub fn from_height_fn(
        images: &mut Assets<Image>,
        height: impl Fn(f32, f32) -> f32,
        resolution: u32,
        origin: Vec2,
        step: f32,
    ) -> Self {
        let mut range = [f32::MAX, f32::MIN];
        let mut bytes = Vec::with_capacity((resolution * resolution) as usize);
        let mut raw = Vec::with_capacity(bytes.capacity());
        for row in 0..resolution {
            for column in 0..resolution {
                let value = height(
                    origin.x + column as f32 * step,
                    origin.y + row as f32 * step,
                );
                range[0] = range[0].min(value);
                range[1] = range[1].max(value);
                raw.push(value);
            }
        }
        let span = (range[1] - range[0]).max(f32::MIN_POSITIVE);
        for value in &raw {
            bytes.push(
                (((value - range[0]) / span) * 255.0)
                    .round()
                    .clamp(0.0, 255.0) as u8,
            );
        }
        let image = Image::new(
            Extent3d {
                width: resolution,
                height: resolution,
                depth_or_array_layers: 1,
            },
            TextureDimension::D2,
            bytes,
            TextureFormat::R8Unorm,
            RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
        );
        Self {
            image: images.add(image),
            origin,
            size: Vec2::splat(step * (resolution - 1) as f32),
            height_range: range,
        }
    }
}

/// Fallback binding when no [`BedHeightMap`] exists. Its contents are never
/// read: shaders return the deep default before touching the texture.
#[derive(Resource, Debug, Clone, ExtractResource)]
pub struct GpuFallback(pub Handle<Image>);

impl FromWorld for GpuFallback {
    fn from_world(world: &mut World) -> Self {
        let mut images = world.resource_mut::<Assets<Image>>();
        let image = Image::new_fill(
            Extent3d::default(),
            TextureDimension::D2,
            &[0],
            TextureFormat::R8Unorm,
            RenderAssetUsages::default(),
        );
        Self(images.add(image))
    }
}

/// Resolves the GPU texture render-side consumers should bind: the bed map
/// when present, else the never-read fallback.
pub fn gpu_image<'a>(
    bed: Option<&BedHeightMap>,
    fallback: &'a GpuFallback,
    images: &'a RenderAssets<GpuImage>,
) -> Option<&'a GpuImage> {
    let handle = bed
        .map(|map| map.image.clone())
        .unwrap_or_else(|| fallback.0.clone());
    images.get(&handle)
}

/// Registers extraction for the bed map and fallback.
pub fn add(app: &mut App) {
    app.add_plugins(ExtractResourcePlugin::<BedHeightMap>::default());
    app.init_resource::<GpuFallback>();
    app.add_plugins(ExtractResourcePlugin::<GpuFallback>::default());
}
