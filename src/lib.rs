//! Camera-centred ocean rendering for Bevy.
//!
//! Aqua provides concentric ocean geometry, analytic and FFT displacement,
//! depth-aware transmission, reflections, foam, shallow-water attenuation,
//! and an underwater volume pass.
//! One world unit is one metre.
//!
//! # Setup
//!
//! Add [`AquaPlugin`], insert the [`Ocean`] resource, and spawn one active
//! [`Camera3d`]. [`Ocean::level`] sets sea level. Add
//! [`DepthPrepass`](bevy::core_pipeline::prepass::DepthPrepass) to the camera
//! for depth-aware transmission.
//!
//! Insert [`OceanWaves`] and [`AquaSettings`] before [`AquaPlugin`] to replace
//! their defaults. [`OceanWaves::sea_state`] is startup-only.
//!
//! Insert a [`BedHeightMap`] built from the terrain heightfield for wave
//! attenuation and shoreline foam. Without one, water uses the deep default.
//!
//! With the `query` feature, add `WaveQuery` to floating entities to receive
//! `WaveSurface` samples from the same cascades used for rendering.
//!
//! # Supported facade
//!
//! The supported application-facing contract is this crate's documented root:
//!
//! - setup and global configuration: [`AquaPlugin`], [`Ocean`], [`OceanWaves`],
//!   [`AquaSettings`], [`BedHeightMap`], and [`AquaDebug`];
//! - bounded-water authoring: [`WaterBody`], [`WaterShape`], [`WaterOptics`],
//!   [`RiverPath`], [`RiverPoint`], and Bevy [`Transform`];
//! - water and rendering choices: [`SeaState`], [`WaveModel`],
//!   [`ReflectionMode`], and [`Caustics`];
//! - optional feature APIs documented at the root, including `WaveQuery`,
//!   `WaveSurface`, `ReflectedInWater`, `SprayQuality`, and `SpraySettings`.
//!
//! `FlowSample`, `RiverSample`, and `CausticsSunVisibility` support advanced
//! integrations with the authored shapes and sky. Public items hidden in Aqua's
//! implementation subcrates are cross-crate render contracts, not part of this
//! application facade. They remain public because Rust has no workspace-only
//! visibility across sibling crates.

mod ocean;

use bevy::{camera::visibility::VisibilitySystems, prelude::*};

#[doc(inline)]
pub use bevy_aqua_core::{
    AquaDebug, AquaSettings, BedHeightMap, Caustics, CausticsSunVisibility, FlowSample, Ocean,
    OceanWaves, ReflectionMode, RiverPath, RiverPoint, RiverSample, SeaState, WaterBody,
    WaterOptics, WaterShape, WaveModel,
};
use bevy_aqua_core::{
    CascadeDataReady, CascadeMaterial, CascadeMaterialsUpdated, Data, OceanView, ViewDetail,
    ViewOrder, ViewPos, ViewSeaLevel, projected_detail_lod,
};
#[cfg(feature = "motion")]
pub use bevy_aqua_motion::{AquaMotionPlugin, AquaMotionSystems};
#[cfg(feature = "query")]
#[doc(inline)]
pub use bevy_aqua_query::{WaveQuery, WaveSurface};
#[cfg(feature = "reflect")]
pub use bevy_aqua_reflect::ReflectedInWater;
#[cfg(feature = "spray")]
pub use bevy_aqua_spray::{SprayQuality, SpraySettings};

/// Adds the ocean renderer and its simulation plugins.
#[derive(Debug, Default, Clone, Copy)]
pub struct AquaPlugin;

impl Plugin for AquaPlugin {
    fn build(&self, app: &mut App) {
        bevy_aqua_core::add_shader(app);
        bevy_aqua_core::bed::add(app);
        app.add_plugins(bevy::render::extract_resource::ExtractResourcePlugin::<Data>::default());
        app.init_resource::<bevy_aqua_core::WaterFields>();
        app.add_plugins(bevy_aqua_waves::AquaWavesPlugin);
        app.add_plugins(bevy_aqua_foam::AquaFoamPlugin);
        app.add_plugins(bevy_aqua_shore::AquaShorePlugin);
        app.add_plugins(bevy_aqua_volume::AquaVolumePlugin);
        #[cfg(feature = "motion")]
        app.add_plugins(bevy_aqua_motion::AquaMotionPlugin);
        #[cfg(feature = "reflect")]
        app.add_plugins(bevy_aqua_reflect::AquaReflectPlugin);
        #[cfg(feature = "spray")]
        app.add_plugins(bevy_aqua_spray::AquaSprayPlugin);
        // Root creation and level changes run before transform propagation.
        // The resolved snapshot prunes invalid or removed bounded-only roots.
        app.add_systems(Update, ocean::sync).add_systems(
            PostUpdate,
            ocean::prune.after(bevy_aqua_core::WaterBodiesResolved),
        );
        #[cfg(feature = "query")]
        app.add_plugins(bevy_aqua_query::AquaQueryPlugin);
        app.add_plugins(bevy::pbr::MaterialPlugin::<CascadeMaterial>::default())
            .init_resource::<AquaDebug>()
            .init_resource::<AquaSettings>()
            .init_resource::<CausticsSunVisibility>()
            .init_resource::<OceanWaves>()
            .init_resource::<ViewPos>()
            .init_resource::<ViewDetail>()
            .init_resource::<ViewSeaLevel>()
            .init_resource::<ViewOrder>()
            .add_systems(
                Startup,
                (
                    lod_init.in_set(CascadeDataReady),
                    ocean::init.after(lod_init),
                ),
            )
            .add_systems(
                PostUpdate,
                (update_view, lod_update, ocean::follow)
                    .chain()
                    .in_set(CascadeMaterialsUpdated)
                    .before(TransformSystems::Propagate),
            )
            .add_systems(
                PostUpdate,
                (ocean::update_bounds, ApplyDeferred)
                    .chain()
                    .after(TransformSystems::Propagate)
                    .before(VisibilitySystems::CheckVisibility),
            );
    }
}

type OceanCameraQuery<'w, 's> = Query<
    'w,
    's,
    (
        Entity,
        &'static bevy::camera::Camera,
        &'static bevy::camera::Projection,
        Option<&'static OceanView>,
    ),
    (
        With<bevy::camera::Camera3d>,
        Without<bevy_aqua_core::AuxiliaryWaterView>,
    ),
>;

fn update_view(
    mut commands: Commands,
    cameras: OceanCameraQuery,
    ocean_views: Query<Entity, With<OceanView>>,
    ocean: Option<Res<Ocean>>,
    transforms: TransformHelper,
    mut state: (
        ResMut<ViewPos>,
        ResMut<ViewDetail>,
        ResMut<ViewSeaLevel>,
        ResMut<ViewOrder>,
    ),
) {
    let Some((entity, camera, projection, marker)) =
        cameras.iter().find(|(_, camera, _, _)| camera.is_active)
    else {
        return;
    };
    for previous in &ocean_views {
        if previous != entity {
            commands.entity(previous).remove::<OceanView>();
        }
    }
    if marker.is_none() {
        commands.entity(entity).insert(OceanView);
    }
    if state.3.0 != camera.order {
        state.3.0 = camera.order;
    }
    let Ok(transform) = transforms.compute_global_transform(entity) else {
        return;
    };
    let position = transform.translation();
    if state.0.0 != position.xz() {
        state.0.0 = position.xz();
    }

    let next_sea_level = ocean.as_deref().map_or(0.0, |ocean| ocean.level);
    if state.2.0 != next_sea_level {
        state.2.0 = next_sea_level;
    }
    let next_detail = camera.physical_viewport_size().map_or(0.0, |viewport| {
        projected_detail_lod(
            (position.y - next_sea_level).abs(),
            projection,
            viewport.y as f32,
        )
    });
    if state.1.0 != next_detail {
        state.1.0 = next_detail;
    }
}

fn lod_init(
    mut commands: Commands,
    bed: Option<Res<BedHeightMap>>,
    fallback: Res<bevy_aqua_core::GpuFallback>,
    sea_level: Res<ViewSeaLevel>,
    foam_textures: Res<bevy_aqua_foam::Textures>,
    mut images: ResMut<Assets<bevy::image::Image>>,
    mut materials: ResMut<Assets<CascadeMaterial>>,
) {
    let texture = images.add(bevy_aqua_core::cascade::make_texture());
    let detail_normal = images.add(bevy_aqua_core::cascade::make_detail_normal_texture());
    let mut layout = bevy_aqua_core::GpuLayout::new(
        &bevy_aqua_core::cascade::layout(Vec2::ZERO),
        Vec2::ZERO,
        0.0,
    );
    layout.set_bed(bed.as_deref(), sea_level.0);
    let (params, maps) = bevy_aqua_shore::bake(&[], false);
    let material = materials.add(CascadeMaterial {
        texture: texture.clone(),
        layout: layout.clone(),
        surface: bevy_aqua_core::cascade::SurfaceParams::default(),
        sea_floor: bed
            .map(|map| map.image.clone())
            .unwrap_or_else(|| fallback.0.clone()),
        detail_normal,
        foam: foam_textures.surface.clone(),
        foam_pattern: foam_textures.pattern.clone(),
        fields: params,
        field_maps: images.add(maps),
        reflection_a: fallback.0.clone(),
        reflection_b: fallback.0.clone(),
        reflections: bevy_aqua_core::PlanarReflectionParams::default(),
    });
    commands.insert_resource(Data::new(material, texture, layout));
}

fn lod_update(
    inputs: bevy_aqua_core::cascade::UpdateInputs<'_>,
    data: ResMut<Data>,
    materials: ResMut<Assets<CascadeMaterial>>,
) {
    bevy_aqua_core::cascade::update(inputs, data, materials);
}

#[cfg(test)]
#[path = "lib_tests.rs"]
mod tests;
