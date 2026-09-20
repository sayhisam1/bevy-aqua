//! A sloped beach that also supplies Aqua's bed height map.
//!
//! Run with `cargo run --example terrain_bed`. Browser instructions are in
//! `examples/README.md`.

use bevy::{
    camera::{Exposure, Hdr},
    core_pipeline::prepass::DepthPrepass,
    light::{Atmosphere, AtmosphereEnvironmentMapLight, atmosphere::ScatteringMedium},
    pbr::AtmosphereSettings,
    prelude::*,
};
use bevy_aqua::{AquaPlugin, AquaSettings, BedHeightMap, Ocean, OceanWaves, ReflectionMode};

const RESOLUTION: u32 = 65;
const STEP: f32 = 2.0;
const ORIGIN: Vec2 = Vec2::splat(-64.0);
const BEACH_SLOPE: f32 = 0.08;
const BEACH_HEIGHT: f32 = -3.0;

fn main() {
    let mut app = App::new();
    app.insert_resource(ClearColor(Color::srgb(0.55, 0.75, 0.9)))
        .add_plugins(DefaultPlugins);

    // Bevy's image storage exists after DefaultPlugins. Insert the completed
    // bed map before AquaPlugin reads it during startup.
    let bed = {
        let mut images = app.world_mut().resource_mut::<Assets<Image>>();
        BedHeightMap::from_height_fn(&mut images, beach_height, RESOLUTION, ORIGIN, STEP)
    };
    app.insert_resource(bed)
        .insert_resource(Ocean::default())
        .insert_resource(OceanWaves {
            shallow_water_attenuation: 1.0,
            ..default()
        })
        .insert_resource(AquaSettings {
            reflections: ReflectionMode::Cubemap,
            ..default()
        })
        .add_plugins(AquaPlugin)
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
        Mesh3d(meshes.add(Plane3d::default().mesh().size(128.0, 128.0))),
        MeshMaterial3d(materials.add(Color::srgb(0.46, 0.38, 0.2))),
        Transform::from_xyz(0.0, BEACH_HEIGHT, 0.0)
            .with_rotation(Quat::from_rotation_z(BEACH_SLOPE.atan())),
    ));
    commands.spawn((
        Camera3d::default(),
        Hdr,
        Exposure { ev100: 12.0 },
        AtmosphereSettings::default(),
        AtmosphereEnvironmentMapLight::default(),
        DepthPrepass,
        Transform::from_xyz(-20.0, 18.0, 45.0).looking_at(Vec3::new(20.0, 0.0, 0.0), Vec3::Y),
    ));
    commands.spawn((
        DirectionalLight {
            illuminance: 16_000.0,
            shadow_maps_enabled: false,
            ..default()
        },
        Transform::from_rotation(Quat::from_euler(EulerRot::XYZ, -0.9, -0.6, 0.0)),
    ));
}

fn beach_height(x: f32, _z: f32) -> f32 {
    BEACH_HEIGHT + BEACH_SLOPE * x
}
