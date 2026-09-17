//! Persistent whitecaps on a rough analytic ocean.
//!
//! Run with `cargo run --example foam`. Browser instructions are in
//! `examples/README.md`.

use bevy::{
    camera::{Exposure, Hdr},
    core_pipeline::prepass::DepthPrepass,
    light::{Atmosphere, AtmosphereEnvironmentMapLight, atmosphere::ScatteringMedium},
    pbr::AtmosphereSettings,
    prelude::*,
};
use bevy_aqua::{AquaPlugin, Ocean, OceanWaves, SeaState};

fn main() {
    App::new()
        .insert_resource(ClearColor(Color::srgb(0.42, 0.62, 0.78)))
        .insert_resource(Ocean::default())
        .insert_resource(OceanWaves {
            sea_state: SeaState::Rough,
            wind_direction_degrees: 20.0,
            ..default()
        })
        .add_plugins((DefaultPlugins, AquaPlugin))
        .add_systems(Startup, setup)
        .run();
}

fn setup(mut commands: Commands, mut scattering_media: ResMut<Assets<ScatteringMedium>>) {
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
        Transform::from_xyz(14.0, 4.5, 18.0).looking_at(Vec3::new(0.0, 0.4, 0.0), Vec3::Y),
    ));
    commands.spawn((
        DirectionalLight {
            illuminance: 20_000.0,
            shadow_maps_enabled: false,
            ..default()
        },
        Transform::from_rotation(Quat::from_euler(EulerRot::XYZ, -0.8, -0.45, 0.0)),
    ));
}
