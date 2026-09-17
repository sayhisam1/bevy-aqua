//! Water-body registration and the global shoreline fields bake.
//!
//! Bounded bodies register against the shared ring tiles. Their union
//! bakes into a level/slot map and a per-texel flow map used by rendering
//! and wave queries.
#![warn(unreachable_pub)]

pub mod bake;

pub use bake::bake;
use bevy::{asset::embedded_asset, light::NotShadowCaster, prelude::*};
use bevy_aqua_core::{
    CascadeMaterial, Data, Ocean, ResolvedWaterBodies, ResolvedWaterBody, WaterBodiesResolved,
    WaterBody, WaterFields, WaterOptics, WaterShape,
};

#[derive(Resource)]
struct ShaderLibraries {
    _handles: Vec<Handle<Shader>>,
}

/// Adds Aqua's shoreline & bed domain: ECS body resolution and
/// the fields-bake implementation.
#[derive(Debug, Default, Clone, Copy)]
pub struct AquaShorePlugin;

impl Plugin for AquaShorePlugin {
    fn build(&self, app: &mut App) {
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
