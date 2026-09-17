//! Fixed spectral-wave ocean using Aqua's FFT producer.
//!
//! Run with `cargo run --example spectral_waves`. Browser instructions are in
//! `examples/README.md`.

use bevy::{
    camera::{Exposure, Hdr},
    core_pipeline::prepass::DepthPrepass,
    light::{Atmosphere, AtmosphereEnvironmentMapLight, atmosphere::ScatteringMedium},
    pbr::AtmosphereSettings,
    prelude::*,
};
use bevy_aqua::{AquaPlugin, Ocean, OceanWaves, SeaState, WaveModel};

fn main() {
    App::new()
        .insert_resource(ClearColor(Color::srgb(0.48, 0.68, 0.84)))
        .insert_resource(Ocean::default())
        .insert_resource(OceanWaves {
            model: WaveModel::Spectral,
            sea_state: SeaState::Moderate,
            wind_direction_degrees: 25.0,
            wind_speed: 14.0,
            fetch: 60_000.0,
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
    // A clear color is only a background, not light that water can reflect.
    commands.spawn(Atmosphere::earth(
        scattering_media.add(ScatteringMedium::earth(256, 256)),
    ));
    commands.spawn((
        Camera3d::default(),
        Hdr,
        // Preserve daylight highlight range before tonemapping.
        Exposure { ev100: 12.0 },
        AtmosphereSettings::default(),
        AtmosphereEnvironmentMapLight::default(),
        DepthPrepass,
        Transform::from_xyz(20.0, 5.5, 24.0).looking_at(Vec3::new(0.0, 0.3, 0.0), Vec3::Y),
    ));
    commands.spawn((
        DirectionalLight {
            illuminance: 18_000.0,
            shadow_maps_enabled: false,
            ..default()
        },
        Transform::from_rotation(Quat::from_euler(EulerRot::XYZ, -0.8, -0.5, 0.0)),
    ));

    let post = meshes.add(Cylinder::new(0.25, 5.0));
    let marker = materials.add(Color::srgb(0.95, 0.32, 0.08));
    for x in [-9.0, -3.0, 3.0, 9.0] {
        commands.spawn((
            Mesh3d(post.clone()),
            MeshMaterial3d(marker.clone()),
            Transform::from_xyz(x, 1.5, 0.0),
        ));
    }
}
