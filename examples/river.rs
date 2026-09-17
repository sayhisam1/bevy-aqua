//! A curved, varying-width river without a global ocean.
//!
//! Run with `cargo run --example river`. Browser instructions are in
//! `examples/README.md`.

use bevy::{
    camera::{Exposure, Hdr},
    core_pipeline::prepass::DepthPrepass,
    light::{Atmosphere, AtmosphereEnvironmentMapLight, atmosphere::ScatteringMedium},
    pbr::AtmosphereSettings,
    prelude::*,
};
use bevy_aqua::{
    AquaPlugin, AquaSettings, ReflectionMode, RiverPath, RiverPoint, WaterBody, WaterOptics,
    WaterShape,
};

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
        Transform::from_xyz(0.0, 42.0, 52.0).looking_at(Vec3::new(0.0, 0.0, 0.0), Vec3::Y),
    ));
    commands.spawn((
        DirectionalLight {
            illuminance: 18_000.0,
            shadow_maps_enabled: false,
            ..default()
        },
        Transform::from_rotation(Quat::from_euler(EulerRot::XYZ, -0.9, -0.5, 0.0)),
    ));
    commands.spawn((
        Mesh3d(meshes.add(Plane3d::default().mesh().size(70.0, 70.0))),
        MeshMaterial3d(materials.add(Color::srgb(0.30, 0.22, 0.12))),
        Transform::from_xyz(0.0, -1.5, 0.0),
    ));

    commands.spawn((
        WaterBody,
        WaterShape::River {
            path: RiverPath {
                points: vec![
                    RiverPoint::new(Vec2::new(-25.0, -23.0), 7.0, 0.25),
                    RiverPoint::new(Vec2::new(-15.0, -10.0), 10.0, 0.4),
                    RiverPoint::new(Vec2::new(2.0, -4.0), 13.0, 0.55),
                    RiverPoint::new(Vec2::new(15.0, 7.0), 9.0, 0.7),
                    RiverPoint::new(Vec2::new(10.0, 21.0), 6.0, 0.85),
                ],
            },
        },
        WaterOptics::CLEAR_FRESH,
        Transform::default(),
    ));

    // A few simple stones make the bends and scale easy to read.
    let stone_mesh = meshes.add(Cuboid::new(2.4, 1.3, 2.4));
    let stone_material = materials.add(Color::srgb(0.28, 0.27, 0.24));
    for position in [
        Vec3::new(-28.0, -0.6, -18.0),
        Vec3::new(-20.0, -0.6, -5.0),
        Vec3::new(-4.0, -0.6, 3.0),
        Vec3::new(9.0, -0.6, -1.0),
        Vec3::new(21.0, -0.6, 8.0),
        Vec3::new(5.0, -0.6, 18.0),
    ] {
        commands.spawn((
            Mesh3d(stone_mesh.clone()),
            MeshMaterial3d(stone_material.clone()),
            Transform::from_translation(position),
        ));
    }
}
