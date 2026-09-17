//! GPU wave query driving a procedural buoy.
//!
//! Run with `cargo run --example wave_query`. Browser instructions are in
//! `examples/README.md`.

use bevy::{
    camera::{Exposure, Hdr},
    core_pipeline::prepass::DepthPrepass,
    light::{Atmosphere, AtmosphereEnvironmentMapLight, atmosphere::ScatteringMedium},
    pbr::AtmosphereSettings,
    prelude::*,
};
use bevy_aqua::{AquaPlugin, Ocean, WaveQuery, WaveSurface};

const BUOY_HEIGHT: f32 = 0.25;
const FOLLOW_RATE: f32 = 4.0;

#[derive(Component)]
struct Buoy;

fn main() {
    App::new()
        .insert_resource(ClearColor(Color::srgb(0.52, 0.73, 0.9)))
        .insert_resource(Ocean::default())
        .add_plugins((DefaultPlugins, AquaPlugin))
        .add_systems(Startup, setup)
        .add_systems(Update, follow_surface)
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
        Transform::from_xyz(10.0, 5.0, 13.0).looking_at(Vec3::new(0.0, 0.5, 0.0), Vec3::Y),
    ));
    commands.spawn((
        DirectionalLight {
            illuminance: 18_000.0,
            shadow_maps_enabled: false,
            ..default()
        },
        Transform::from_rotation(Quat::from_euler(EulerRot::XYZ, -0.9, -0.5, 0.0)),
    ));

    let hull = meshes.add(Sphere::new(1.0));
    let material = materials.add(Color::srgb(0.95, 0.18, 0.04));
    commands
        .spawn((
            Buoy,
            WaveQuery,
            Transform::from_xyz(0.0, BUOY_HEIGHT, 0.0),
            Visibility::default(),
        ))
        .with_child((
            Mesh3d(hull),
            MeshMaterial3d(material),
            Transform::from_scale(Vec3::new(0.8, 1.1, 0.8)),
        ));
}

fn follow_surface(time: Res<Time>, mut buoys: Query<(&WaveSurface, &mut Transform), With<Buoy>>) {
    let response = 1.0 - (-FOLLOW_RATE * time.delta_secs()).exp();
    for (surface, mut transform) in &mut buoys {
        if !surface.valid {
            continue;
        }
        let target_height = BUOY_HEIGHT + surface.displacement.y;
        transform.translation.y = transform.translation.y.lerp(target_height, response);
        let target_rotation = Quat::from_rotation_arc(Vec3::Y, surface.normal);
        transform.rotation = transform.rotation.slerp(target_rotation, response);
    }
}
