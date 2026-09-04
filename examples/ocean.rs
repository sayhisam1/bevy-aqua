//! Minimal analytic ocean.
//!
//! Run with `cargo run --example ocean`. Browser instructions are in
//! `examples/README.md`.

use bevy::{
    camera::{Exposure, Hdr},
    core_pipeline::prepass::DepthPrepass,
    light::{Atmosphere, AtmosphereEnvironmentMapLight, atmosphere::ScatteringMedium},
    pbr::AtmosphereSettings,
    prelude::*,
};
use bevy_aqua::{AquaPlugin, Ocean};

fn main() {
    App::new()
        .insert_resource(ClearColor(Color::srgb(0.53, 0.75, 0.92)))
        .insert_resource(Ocean::default())
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
        Transform::from_xyz(18.0, 8.0, 22.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
    commands.spawn((
        DirectionalLight {
            illuminance: 16_000.0,
            shadow_maps_enabled: false,
            ..default()
        },
        Transform::from_rotation(Quat::from_euler(EulerRot::XYZ, -0.9, -0.6, 0.0)),
    ));
    commands.spawn((
        Mesh3d(meshes.add(Plane3d::default().mesh().size(160.0, 160.0))),
        MeshMaterial3d(materials.add(Color::srgb(0.42, 0.35, 0.22))),
        Transform::from_xyz(0.0, -4.0, 0.0),
    ));
}
