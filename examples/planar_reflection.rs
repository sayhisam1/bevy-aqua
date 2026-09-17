//! Planar reflection of a simple procedural marker.
//!
//! Run with `cargo run --example planar_reflection`. Browser instructions are in
//! `examples/README.md`.

use bevy::{
    camera::{Exposure, Hdr},
    core_pipeline::prepass::DepthPrepass,
    light::{Atmosphere, AtmosphereEnvironmentMapLight, atmosphere::ScatteringMedium},
    pbr::AtmosphereSettings,
    prelude::*,
};
use bevy_aqua::{
    AquaPlugin, AquaSettings, Ocean, OceanWaves, ReflectedInWater, ReflectionMode, SeaState,
};

fn main() {
    App::new()
        .insert_resource(ClearColor(Color::srgb(0.58, 0.76, 0.9)))
        .insert_resource(Ocean::default())
        .insert_resource(OceanWaves {
            sea_state: SeaState::Calm,
            ..default()
        })
        .insert_resource(AquaSettings {
            reflections: ReflectionMode::Planar {
                scale: 0.5,
                distortion: 0.01,
            },
            ..default()
        })
        .add_plugins((DefaultPlugins, AquaPlugin))
        .add_systems(Startup, setup)
        .run();
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut scattering_media: ResMut<Assets<ScatteringMedium>>,
) {
    // The visible sky also supplies the water's reflected environment light.
    commands.spawn(Atmosphere::earth(
        scattering_media.add(ScatteringMedium::earth(256, 256)),
    ));
    commands.spawn((
        Camera3d::default(),
        Hdr,
        Exposure { ev100: 12.0 },
        AtmosphereSettings::default(),
        AtmosphereEnvironmentMapLight::default(),
        DepthPrepass,
        Transform::from_xyz(13.0, 5.5, 18.0).looking_at(Vec3::new(0.0, 1.2, 0.0), Vec3::Y),
    ));
    commands.spawn((
        DirectionalLight {
            illuminance: 18_000.0,
            shadow_maps_enabled: false,
            ..default()
        },
        Transform::from_rotation(Quat::from_euler(EulerRot::XYZ, -0.9, -0.5, 0.0)),
    ));

    let white = materials.add(Color::srgb(0.92, 0.88, 0.72));
    let red = materials.add(Color::srgb(0.9, 0.08, 0.04));
    let column = meshes.add(Cuboid::new(1.2, 5.0, 1.2));
    let crossbar = meshes.add(Cuboid::new(7.0, 0.8, 1.2));
    commands.spawn((
        Mesh3d(column.clone()),
        MeshMaterial3d(white.clone()),
        ReflectedInWater,
        Transform::from_xyz(-3.0, 2.2, 0.0),
    ));
    commands.spawn((
        Mesh3d(column),
        MeshMaterial3d(white),
        ReflectedInWater,
        Transform::from_xyz(3.0, 2.2, 0.0),
    ));
    commands.spawn((
        Mesh3d(crossbar),
        MeshMaterial3d(red),
        ReflectedInWater,
        Transform::from_xyz(0.0, 4.4, 0.0),
    ));
}
