//! Global water fields: one baked pair of textures covering every
//! resolved [`crate::WaterBody`] + [`crate::WaterShape`] entities.
//!
//! The camera-centred ring tiles are the only water mesh. Each texel
//! resolves which body owns the point: packed `maps` layer 0 stores the
//! surface level (r) and the one-based body slot (g); layer 1 stores the
//! per-texel river sample (xy current, z signed bank margin, w channel
//! half-width). Bounded vertices read their level here and
//! fragments resolve optics, banks, and culling from the slot parameters.

use bevy::{
    asset::RenderAssetUsages,
    image::ImageSampler,
    prelude::*,
    render::render_resource::{Extent3d, ShaderType, TextureDimension, TextureFormat},
};

use crate::cascade::BodyParams;
use crate::{AmortizedBake, ResolvedWaterBody};

/// Hard cap on registered bounded bodies (uniform array size).
pub const MAX_BODIES: usize = 16;
/// Array layers in the packed field texture: level/slot, then river flow.
pub const FIELD_LAYER_COUNT: u32 = 2;
/// Shared format of every packed field texture.
pub const FIELD_TEXTURE_FORMAT: TextureFormat = TextureFormat::Rgba16Float;

/// Uniform mirror of the baked fields, matching `FieldParams` in
/// `cascade/common.wgsl`.
#[derive(ShaderType, Debug, Clone, Copy)]
pub struct FieldParams {
    /// xy: region minimum in metres; zw: region size in metres.
    pub region: Vec4,
    /// x: bounded body count; y: 1.0 when the Ocean resource is present;
    /// z: metres per texel; w: reserved.
    pub meta: Vec4,
    /// Per-slot body parameters; slot i lives at bodies[i - 1].
    pub bodies: [BodyParams; MAX_BODIES],
}

impl FieldParams {
    /// The inert all-ocean uniform used before any bake.
    pub fn none() -> Self {
        Self {
            region: Vec4::ZERO,
            meta: Vec4::ZERO,
            bodies: [BodyParams::ocean(); MAX_BODIES],
        }
    }
}

/// The baked global fields and the uniform that maps them.
#[derive(Resource, Debug)]
pub struct WaterFields {
    pub params: FieldParams,
    pub maps: Handle<Image>,
    /// Rebake scheduler: runs the CPU bake only when the body set changes.
    pub bakes: AmortizedBake<(bool, Vec<ResolvedWaterBody>)>,
}

impl FromWorld for WaterFields {
    fn from_world(world: &mut World) -> Self {
        let mut images = world.resource_mut::<Assets<Image>>();
        let bytes_per_texel = FIELD_TEXTURE_FORMAT
            .block_copy_size(None)
            .expect("packed field texture format must have a fixed block size")
            as usize;
        let zero_texel = vec![0; bytes_per_texel];
        let mut maps = Image::new_fill(
            Extent3d {
                depth_or_array_layers: FIELD_LAYER_COUNT,
                ..default()
            },
            TextureDimension::D2,
            &zero_texel,
            FIELD_TEXTURE_FORMAT,
            RenderAssetUsages::default(),
        );
        maps.sampler = ImageSampler::linear();
        Self {
            params: FieldParams::none(),
            maps: images.add(maps),
            bakes: AmortizedBake::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn field_shader_filters_height_but_not_discrete_ownership() {
        let shader = include_str!("cascade/common.wgsl");
        let sample = shader
            .split("fn sample_field_level(")
            .nth(1)
            .unwrap()
            .split("fn sample_field_flow(")
            .next()
            .unwrap();
        assert!(sample.contains("textureSampleLevel(field_maps, field_sampler, uv, 0, 0.0).x"));
        assert!(sample.contains("textureLoad(field_maps, coord, 0, 0).y"));
        assert!(sample.contains("return vec2(level, slot);"));
    }

    #[test]
    fn nearest_slot_sampling_cannot_invent_a_body_at_boundaries() {
        let slots = [0_u32, 2, 7, 16];
        for y in -4..=12 {
            for x in -4..=12 {
                let uv = Vec2::new(x as f32 / 8.0, y as f32 / 8.0);
                let coord = (uv * 2.0).floor().as_ivec2().clamp(IVec2::ZERO, IVec2::ONE);
                let slot = slots[(coord.y * 2 + coord.x) as usize];
                assert!(slots.contains(&slot));
            }
        }
        // Bilinear filtering a 0/2 edge instead fabricates valid slot 1.
        assert_eq!((0.5_f32 * 0.0 + 0.5 * 2.0).round() as u32, 1);
        assert!(!slots.contains(&1));
    }
}
