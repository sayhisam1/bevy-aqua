//! Two bounded water shapes without a global ocean.
//!
//! Run with `cargo run --example bounded_water`. Browser instructions are in
//! `examples/README.md`.

use bevy::{
    camera::{Exposure, Hdr},
    core_pipeline::prepass::DepthPrepass,
    light::{Atmosphere, AtmosphereEnvironmentMapLight, atmosphere::ScatteringMedium},
    pbr::AtmosphereSettings,
    prelude::*,
};
use bevy_aqua::{AquaPlugin, AquaSettings, ReflectionMode, WaterBody, WaterOptics, WaterShape};

fn main() {
    App::new()
        .insert_resource(ClearColor(Color::srgb(0.55, 0.75, 0.9)))
        .insert_resource(AquaSettings {
            reflections: ReflectionMode::Cubemap,
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
        Transform::from_xyz(0.0, 34.0, 42.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
    commands.spawn((
        DirectionalLight {
            illuminance: 18_000.0,
            shadow_maps_enabled: false,
            ..default()
        },
        Transform::from_rotation(Quat::from_euler(EulerRot::XYZ, -0.9, -0.7, 0.0)),
    ));

    commands.spawn((
        Mesh3d(meshes.add(Plane3d::default().mesh().size(58.0, 38.0))),
        MeshMaterial3d(materials.add(Color::srgb(0.36, 0.27, 0.15))),
        Transform::from_xyz(0.0, -1.8, 0.0),
    ));

    commands.spawn((
        WaterBody,
        WaterShape::Circle { radius: 9.0 },
        WaterOptics::CLEAR_FRESH,
        Transform::from_xyz(-13.0, 0.0, 0.0),
    ));
    commands.spawn((
        WaterBody,
        WaterShape::Polygon {
            points: vec![
                Vec2::new(-9.0, -6.0),
                Vec2::new(5.0, -8.0),
                Vec2::new(10.0, -1.0),
                Vec2::new(6.0, 7.0),
                Vec2::new(-6.0, 8.0),
                Vec2::new(-10.0, 2.0),
            ],
        },
        WaterOptics::COASTAL,
        Transform::from_xyz(13.0, 0.0, 0.0),
    ));
}
