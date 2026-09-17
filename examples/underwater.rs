//! Open ocean from 20 m below the surface, looking toward an angled sun.
//!
//! Run with `cargo run --example underwater`. Browser instructions are in
//! `examples/README.md`.

use bevy::{
    camera::Exposure,
    core_pipeline::prepass::DepthPrepass,
    light::{
        Atmosphere, AtmosphereEnvironmentMapLight, atmosphere::ScatteringMedium, light_consts::lux,
    },
    pbr::AtmosphereSettings,
    prelude::*,
};
use bevy_aqua::{AquaPlugin, AquaSettings, Ocean, OceanWaves, SeaState};

fn main() {
    App::new()
        .insert_resource(ClearColor(Color::BLACK))
        .insert_resource(GlobalAmbientLight::NONE)
        .insert_resource(Ocean::default())
        .insert_resource(OceanWaves {
            sea_state: SeaState::Rough,
            ..default()
        })
        .insert_resource(AquaSettings {
            atmospheric_sunlight: true,
            ..default()
        })
        .add_plugins((DefaultPlugins, AquaPlugin))
        .add_systems(Startup, setup)
        .run();
}

fn setup(mut commands: Commands, mut scattering_mediums: ResMut<Assets<ScatteringMedium>>) {
    let sun = Transform::from_rotation(Quat::from_euler(EulerRot::XYZ, -0.7, -0.55, 0.0));
    let earth_medium = scattering_mediums.add(ScatteringMedium::earth(256, 256));
    commands.spawn(Atmosphere::earth(earth_medium));
    commands.spawn((
        Camera3d::default(),
        DepthPrepass,
        AtmosphereSettings::default(),
        AtmosphereEnvironmentMapLight::default(),
        Exposure { ev100: 13.0 },
        Transform::from_xyz(0.0, -20.0, 0.0).looking_to(sun.back(), Vec3::Y),
    ));
    commands.spawn((
        DirectionalLight {
            illuminance: lux::RAW_SUNLIGHT,
            shadow_maps_enabled: false,
            ..default()
        },
        sun,
    ));
}
