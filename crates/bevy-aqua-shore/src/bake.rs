//! Global water fields for registered bodies.
//!
//! The bake returns a level/slot map, a flow map, and their mapping
//! parameters. Scheduling is handled by the umbrella crate.
#![warn(unreachable_pub)]

use bevy::{
    asset::RenderAssetUsages,
    image::ImageSampler,
    prelude::*,
    render::render_resource::{Extent3d, TextureDimension},
};
use bevy_aqua_core::{
    BodyOptics, BodyParams, FIELD_LAYER_COUNT, FIELD_TEXTURE_FORMAT, FieldParams, MAX_BODIES,
    ResolvedWaterBody, WaterShape,
};

/// Longest field side in texels; bounds VRAM for kilometre-scale regions.
const MAX_FIELD_SIDE: u32 = 2048;
/// Finest bake resolution in metres per texel.
const MIN_TEXEL: f32 = 0.25;
/// Padding around the union of body extents.
const REGION_PAD: f32 = 8.0;
/// Inside-marker for non-river bodies: saturates the 8 m bank-fade band.
const INSIDE_NO_FLOW: f32 = 8.0;
/// Minimum side that keeps a linear-filtered field map nondegenerate.
const MIN_FIELD_EXTENT: u32 = 4;

/// Converts an `f32` to its IEEE 754 binary16 bit pattern.
pub fn f32_to_f16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let magnitude = bits & 0x7FFF_FFFF;
    if magnitude >= 0x7F80_0000 {
        // Infinity and NaN keep their class.
        return sign | 0x7C00 | (u16::from(magnitude > 0x7F80_0000) * 0x0200);
    }
    if magnitude >= 0x4780_0000 {
        // Beyond half range: clamp to infinity.
        return sign | 0x7C00;
    }
    if magnitude < 0x3300_0000 {
        // Rounds to half zero.
        return sign;
    }
    let exponent = ((magnitude >> 23) & 0xFF) as i32 - 127;
    if magnitude < 0x3880_0000 {
        // Sub-normal half: quantise value * 2^24 with round-to-nearest.
        let scaled = f32::from_bits(magnitude) * 2.0_f32.powi(24) + 0.5;
        return sign | (scaled as u16);
    }
    let fraction = magnitude & 0x007F_FFFF;
    let mut half_fraction = fraction >> 13;
    let round_bit = (fraction >> 12) & 1;
    let sticky = fraction & 0x0FFF;
    if round_bit == 1 && (sticky != 0 || half_fraction & 1 == 1) {
        half_fraction += 1;
    }
    // Rounding may carry into the next exponent.
    let (half_exponent, half_fraction) = if half_fraction >= 0x400 {
        (exponent + 16, 0u32)
    } else {
        (exponent + 15, half_fraction)
    };
    sign | ((half_exponent as u16) << 10) | half_fraction as u16
}

fn push_f16(data: &mut Vec<u8>, value: f32) {
    data.extend_from_slice(&f32_to_f16_bits(value).to_le_bytes());
}

fn field_image(width: u32, height: u32, layers: u32, data: Vec<u8>) -> Image {
    let mut image = Image::new(
        Extent3d {
            width,
            height,
            depth_or_array_layers: layers,
        },
        TextureDimension::D2,
        data,
        FIELD_TEXTURE_FORMAT,
        RenderAssetUsages::default(),
    );
    image.sampler = ImageSampler::linear();
    image
}

/// Bakes packed field maps and their uniform.
///
/// Array layer 0 is level+slot; layer 1 is river flow.
pub fn bake(bodies: &[ResolvedWaterBody], has_ocean: bool) -> (FieldParams, Image) {
    let mut params = FieldParams::none();
    params.meta.y = f32::from(has_ocean);
    if bodies.is_empty() {
        let texel_count =
            MIN_FIELD_EXTENT as usize * MIN_FIELD_EXTENT as usize * FIELD_LAYER_COUNT as usize;
        let bytes_per_texel = FIELD_TEXTURE_FORMAT
            .block_copy_size(None)
            .expect("packed field texture format must have a fixed block size")
            as usize;
        let dummy = field_image(
            MIN_FIELD_EXTENT,
            MIN_FIELD_EXTENT,
            FIELD_LAYER_COUNT,
            vec![0; texel_count * bytes_per_texel],
        );
        return (params, dummy);
    }

    let extents: Vec<(Vec2, f32)> = bodies.iter().map(ResolvedWaterBody::extent).collect();
    let mut minimum = Vec2::splat(f32::MAX);
    let mut maximum = Vec2::splat(f32::MIN);
    for body in bodies {
        let (body_minimum, body_maximum) = body.aabb();
        minimum = minimum.min(body_minimum);
        maximum = maximum.max(body_maximum);
    }
    minimum -= Vec2::splat(REGION_PAD);
    maximum += Vec2::splat(REGION_PAD);
    let size = maximum - minimum;
    let texel = (size.max_element() / MAX_FIELD_SIDE as f32).max(MIN_TEXEL);
    let width = ((size.x / texel).ceil() as u32).clamp(MIN_FIELD_EXTENT, MAX_FIELD_SIDE);
    let height = ((size.y / texel).ceil() as u32).clamp(MIN_FIELD_EXTENT, MAX_FIELD_SIDE);
    params.region = minimum.extend(size.x).extend(size.y);
    params.meta.x = bodies.len() as f32;
    params.meta.z = texel;
    assert!(
        bodies.len() <= MAX_BODIES,
        "too many water bodies: {} > {MAX_BODIES}",
        bodies.len()
    );
    for (slot, ((center, half), body)) in extents.iter().zip(bodies).enumerate() {
        params.bodies[slot] = BodyParams::bounded(
            *center,
            *half,
            minimum,
            size,
            matches!(
                body.shape,
                WaterShape::River { .. } | WaterShape::Corridor { .. }
            ),
            body.optics.map(|optics| BodyOptics {
                extinction: optics.extinction,
                scatter_scale: optics.scatter_scale,
                scatter_tint: optics.scatter_tint,
                scattering_asymmetry: optics.scattering_asymmetry,
                sun_roughness: optics.sun_roughness,
            }),
        );
    }

    let mut level_id = Vec::with_capacity((width * height * 4 * 2) as usize);
    let mut flow = Vec::with_capacity((width * height * 4 * 2) as usize);
    for row in 0..height {
        let z = minimum.y + (row as f32 + 0.5) * texel;
        for column in 0..width {
            let x = minimum.x + (column as f32 + 0.5) * texel;
            let point = Vec2::new(x, z);
            let mut owner = bodies
                .iter()
                .enumerate()
                .find(|(_, body)| body.contains(point))
                .map(|(index, body)| (index, body, body.flow_at(point)));
            // River vertices need an eight-metre support band at the body's level;
            // fragments in this band are clipped by the negative margin. This
            // prevents triangles crossing an elevated bank from becoming
            // vertical water walls.
            if owner.is_none() {
                owner = bodies.iter().enumerate().find_map(|(index, body)| {
                    let flowed = body.flow_at(point)?;
                    (flowed.margin >= -8.0).then_some((index, body, Some(flowed)))
                });
            }
            let (slot, level, sample) = owner.map_or(
                (0, 0.0, [0.0_f32, 0.0, -1.0, 0.0]),
                |(index, body, flowed)| {
                    let sample = flowed.map_or([0.0, 0.0, INSIDE_NO_FLOW, 0.0], |flowed| {
                        [
                            flowed.flow.x,
                            flowed.flow.y,
                            flowed.margin,
                            flowed.half_width,
                        ]
                    });
                    (index + 1, body.level, sample)
                },
            );
            push_f16(&mut level_id, level);
            push_f16(&mut level_id, slot as f32);
            push_f16(&mut level_id, 0.0);
            push_f16(&mut level_id, 0.0);
            for channel in sample {
                push_f16(&mut flow, channel);
            }
        }
    }
    level_id.extend(flow);
    (
        params,
        field_image(width, height, FIELD_LAYER_COUNT, level_id),
    )
}

#[cfg(test)]
#[path = "bake_tests.rs"]
mod tests;
