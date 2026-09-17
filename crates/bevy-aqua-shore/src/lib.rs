//! Water-body registration and the global shoreline fields bake.
//!
//! Bounded bodies register against the shared ring tiles. Their union
//! bakes into a level/slot map and a per-texel flow map used by rendering
//! and wave queries.
#![warn(unreachable_pub)]

pub mod bake;

pub use bake::bake;
use bevy::{
    asset::{RenderAssetUsages, embedded_asset},
    image::{ImageAddressMode, ImageFilterMode, ImageSampler, ImageSamplerDescriptor},
    light::NotShadowCaster,
    prelude::*,
    render::render_resource::{Extent3d, TextureDimension, TextureFormat},
};
use bevy_aqua_core::{
    CascadeMaterial, Data, Ocean, ResolvedWaterBodies, ResolvedWaterBody, WaterBodiesResolved,
    WaterBody, WaterFields, WaterOptics, WaterShape,
};

#[derive(Resource)]
struct ShaderLibraries {
    _handles: Vec<Handle<Shader>>,
}

#[doc(hidden)]
#[derive(Resource, Debug)]
pub struct CausticsTexture(pub Handle<Image>);

/// Adds Aqua's shoreline & bed domain: ECS body resolution and
/// the fields-bake implementation.
#[derive(Debug, Default, Clone, Copy)]
pub struct AquaShorePlugin;

impl Plugin for AquaShorePlugin {
    fn build(&self, app: &mut App) {
        let texture = app
            .world_mut()
            .resource_mut::<Assets<Image>>()
            .add(make_caustics_texture());
        app.insert_resource(CausticsTexture(texture));
        // Keep the import-only shader loaded.
        embedded_asset!(app, "water.wgsl");
        let water = app
            .world()
            .resource::<AssetServer>()
            .load("embedded://bevy_aqua_shore/water.wgsl");
        app.insert_resource(ShaderLibraries {
            _handles: vec![water],
        });
        app.init_resource::<ResolvedWaterBodies>()
            .add_systems(Update, decorate_bodies)
            .add_systems(
                PostUpdate,
                (resolve_bodies, fields_bake_system)
                    .chain()
                    .after(TransformSystems::Propagate)
                    .in_set(WaterBodiesResolved),
            );
    }
}

// R8 512² plus all area-average mips: 349,525 bytes (about 341 KiB).
const CAUSTIC_TEXTURE_SIZE: u32 = 512;
// Sixteen cells delay tile repetition without increasing the neighbour search.
const CAUSTIC_CELL_COUNT: i32 = 16;
// Keeps cell borders narrow without losing them under bilinear filtering.
const CAUSTIC_RIDGE_WIDTH: f32 = 0.16;

fn caustic_feature(cell: IVec2) -> Vec2 {
    let x = cell.x.rem_euclid(CAUSTIC_CELL_COUNT) as u32;
    let y = cell.y.rem_euclid(CAUSTIC_CELL_COUNT) as u32;
    // Fixed avalanche factors make the periodic texture deterministic.
    let hash = x.wrapping_mul(0x9e37_79b9) ^ y.wrapping_mul(0x85eb_ca6b);
    Vec2::new(
        (hash.wrapping_mul(0x27d4_eb2d) & 0xffff) as f32 / 65_535.0,
        (hash.rotate_left(13).wrapping_mul(0x1656_67b1) & 0xffff) as f32 / 65_535.0,
    )
}

fn caustic_ridge(point: Vec2) -> f32 {
    let cell = point.floor().as_ivec2();
    let mut nearest = [f32::MAX; 2];
    for oy in -1..=1 {
        for ox in -1..=1 {
            let neighbour = cell + IVec2::new(ox, oy);
            let distance = point.distance(neighbour.as_vec2() + caustic_feature(neighbour));
            if distance < nearest[0] {
                nearest = [distance, nearest[0]];
            } else if distance < nearest[1] {
                nearest[1] = distance;
            }
        }
    }
    (1.0 - (nearest[1] - nearest[0]) / CAUSTIC_RIDGE_WIDTH).clamp(0.0, 1.0)
}

// Pack largest-to-smallest R8 levels. Average the unquantized values from
// the previous level, never its rounded upload bytes. For this power-of-two
// texture, f64 represents every mean of the original R8 texels exactly.
fn caustic_mip_chain(base: &[u8], size: u32) -> Vec<u8> {
    assert!(size.is_power_of_two());
    assert_eq!(base.len(), (size * size) as usize);
    let total = (0..=size.ilog2())
        .map(|level| ((size >> level) * (size >> level)) as usize)
        .sum();
    let mut packed = Vec::with_capacity(total);
    packed.extend_from_slice(base);
    let mut previous: Vec<f64> = base.iter().map(|&value| f64::from(value)).collect();
    let mut width = size as usize;
    while width > 1 {
        let next_width = width / 2;
        let mut next = Vec::with_capacity(next_width * next_width);
        for y in 0..next_width {
            for x in 0..next_width {
                let i = 2 * y * width + 2 * x;
                let mean =
                    (previous[i] + previous[i + 1] + previous[i + width] + previous[i + width + 1])
                        * 0.25;
                next.push(mean);
                packed.push(mean.round() as u8);
            }
        }
        previous = next;
        width = next_width;
    }
    packed
}

fn make_caustics_texture() -> Image {
    let mut pixels = Vec::with_capacity((CAUSTIC_TEXTURE_SIZE * CAUSTIC_TEXTURE_SIZE) as usize);
    for y in 0..CAUSTIC_TEXTURE_SIZE {
        for x in 0..CAUSTIC_TEXTURE_SIZE {
            let point = Vec2::new(x as f32, y as f32) * CAUSTIC_CELL_COUNT as f32
                / CAUSTIC_TEXTURE_SIZE as f32;
            let ridge = caustic_ridge(point);
            pixels.push((ridge * ridge * 255.0).round() as u8);
        }
    }
    let mip_pixels = caustic_mip_chain(&pixels, CAUSTIC_TEXTURE_SIZE);
    let mut image = Image::new(
        Extent3d {
            width: CAUSTIC_TEXTURE_SIZE,
            height: CAUSTIC_TEXTURE_SIZE,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        pixels,
        TextureFormat::R8Unorm,
        RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
    );
    image.texture_descriptor.mip_level_count = CAUSTIC_TEXTURE_SIZE.ilog2() + 1;
    image.data = Some(mip_pixels);
    image.sampler = ImageSampler::Descriptor(ImageSamplerDescriptor {
        address_mode_u: ImageAddressMode::Repeat,
        address_mode_v: ImageAddressMode::Repeat,
        mag_filter: ImageFilterMode::Linear,
        min_filter: ImageFilterMode::Linear,
        mipmap_filter: ImageFilterMode::Linear,
        ..default()
    });
    image
}

fn resolve_bodies(
    bodies: Query<(Entity, &WaterShape, Option<&WaterOptics>, &GlobalTransform), With<WaterBody>>,
    mut resolved: ResMut<ResolvedWaterBodies>,
    mut warned: Local<std::collections::HashSet<Entity>>,
) {
    let mut invalid = std::collections::HashSet::new();
    let mut next = bodies
        .iter()
        .filter_map(|(entity, shape, optics, transform)| {
            match ResolvedWaterBody::resolve(entity, shape, optics.copied(), transform) {
                Ok(body) => Some(body),
                Err(error) => {
                    invalid.insert(entity);
                    if warned.insert(entity) {
                        warn!(?entity, ?error, "ignoring unsupported water-body transform");
                    }
                    None
                }
            }
        })
        .collect::<Vec<_>>();
    warned.retain(|entity| invalid.contains(entity));
    next.sort_by_key(|body| std::cmp::Reverse(body.entity));
    if resolved.0 != next {
        resolved.0 = next;
    }
}

fn fields_bake_system(
    mut fields: ResMut<WaterFields>,
    bodies: Res<ResolvedWaterBodies>,
    ocean: Option<Res<Ocean>>,
    mut images: ResMut<Assets<Image>>,
    mut materials: ResMut<Assets<CascadeMaterial>>,
    data: Res<Data>,
) {
    let key = (ocean.is_some(), bodies.0.clone());
    if !fields.bakes.maintain(&key) {
        return;
    }
    let (params, maps) = bake(&bodies.0, ocean.is_some());
    let maps_handle = images.add(maps);
    let material_handle = data.material();
    if let Some(mut material) = materials.get_mut(&material_handle) {
        material.fields = params;
        material.field_maps = maps_handle.clone();
    }
    fields.params = params;
    fields.maps = maps_handle;
}

fn decorate_bodies(mut commands: Commands, bodies: Query<Entity, Added<WaterBody>>) {
    for entity in &bodies {
        commands
            .entity(entity)
            .insert((Name::new("WaterBody"), NotShadowCaster));
    }
}

#[cfg(test)]
#[path = "body_tests.rs"]
mod tests;

#[cfg(test)]
mod caustic_refraction_tests;

#[cfg(test)]
mod caustic_filter_tests;
