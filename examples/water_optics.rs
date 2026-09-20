//! Three identical bounded pools comparing Aqua's optics presets.
//!
//! Run with `cargo run --example water_optics`. Browser instructions are in
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
        Transform::from_xyz(0.0, 30.0, 38.0).looking_at(Vec3::new(0.0, 0.0, 1.0), Vec3::Y),
    ));
    commands.spawn((
        DirectionalLight {
            illuminance: 18_000.0,
            shadow_maps_enabled: false,
            ..default()
        },
        Transform::from_rotation(Quat::from_euler(EulerRot::XYZ, -0.85, -0.65, 0.0)),
    ));
    commands.spawn((
        Mesh3d(meshes.add(Plane3d::default().mesh().size(58.0, 30.0))),
        MeshMaterial3d(materials.add(Color::srgb(0.48, 0.39, 0.24))),
        Transform::from_xyz(0.0, -2.2, 0.0),
    ));

    for (x, optics) in [
        (-18.0, WaterOptics::DEEP_OCEAN),
        (0.0, WaterOptics::TROPICAL),
        (18.0, WaterOptics::CLEAR_FRESH),
    ] {
        commands.spawn((
            WaterBody,
            WaterShape::Circle { radius: 7.5 },
            optics,
            Transform::from_xyz(x, 0.0, 0.0),
        ));
    }

    commands
        .spawn(Node {
            position_type: PositionType::Absolute,
            top: Val::Px(24.0),
            width: Val::Percent(100.0),
            justify_content: JustifyContent::SpaceAround,
            ..default()
        })
        .with_children(|legend| {
            for label in ["DEEP OCEAN", "TROPICAL", "CLEAR FRESH"] {
                legend.spawn((
                    Text::new(label),
                    TextFont {
                        font_size: FontSize::Px(24.0),
                        ..default()
                    },
                    TextColor(Color::WHITE),
                ));
            }
        });
}
