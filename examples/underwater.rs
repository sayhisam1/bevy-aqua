//! Underwater ocean lighting with a procedural seabed and depth markers.
//!
//! Underwater integration is based on work contributed by Wells Crosby
//! (@wellscrosby). Run with `cargo run --example underwater --features underwater`.
//! Comparison commands and browser instructions are in `examples/README.md`.

use bevy::{
    camera::{Exposure, Hdr},
    core_pipeline::prepass::DepthPrepass,
    light::{
        Atmosphere, AtmosphereEnvironmentMapLight, atmosphere::ScatteringMedium, light_consts::lux,
    },
    pbr::AtmosphereSettings,
    prelude::*,
};
use bevy_aqua::{AquaPlugin, AquaSettings, Ocean, OceanWaves, SeaState, WaterOptics};

fn main() {
    App::new()
        .insert_resource(ClearColor(Color::srgb(0.53, 0.75, 0.92)))
        .insert_resource(GlobalAmbientLight::NONE)
        .insert_resource(Ocean::default())
        .insert_resource(OceanWaves {
            sea_state: SeaState::Moderate,
            ..default()
        })
        .insert_resource(AquaSettings {
            // This comparison uses clear water so the nearby depth markers
            // remain readable; production defaults stay unchanged.
            water_optics: WaterOptics {
                extinction: Vec3::new(0.06, 0.025, 0.015),
                scatter_scale: 0.18,
                ..WaterOptics::CLEAR_FRESH
            },
            atmospheric_sunlight: true,
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
    // The same atmosphere lights the geometry and supplies the sky seen through
    // the surface's Snell window.
    commands.spawn(Atmosphere::earth(
        scattering_media.add(ScatteringMedium::earth(256, 256)),
    ));
    commands.spawn((
        Camera3d::default(),
        Hdr,
        Projection::from(PerspectiveProjection {
            fov: 70.0_f32.to_radians(),
            ..default()
        }),
        Exposure { ev100: 12.0 },
        AtmosphereSettings::default(),
        AtmosphereEnvironmentMapLight::default(),
        DepthPrepass,
        Transform::from_xyz(8.0, -4.0, 12.0).looking_at(Vec3::new(0.0, 4.0, -2.0), Vec3::Y),
    ));
    commands.spawn((
        DirectionalLight {
            illuminance: lux::RAW_SUNLIGHT,
            shadow_maps_enabled: true,
            ..default()
        },
        Transform::from_rotation(Quat::from_euler(EulerRot::XYZ, -0.85, -0.55, 0.0)),
    ));

    let sand = materials.add(StandardMaterial {
        base_color: Color::srgb(0.46, 0.34, 0.19),
        perceptual_roughness: 0.94,
        ..default()
    });
    let rock = materials.add(StandardMaterial {
        base_color: Color::srgb(0.22, 0.29, 0.25),
        perceptual_roughness: 0.86,
        ..default()
    });
    let marker = materials.add(StandardMaterial {
        base_color: Color::srgb(0.95, 0.34, 0.08),
        metallic: 0.12,
        perceptual_roughness: 0.34,
        ..default()
    });

    // The bed is at -12 m. The three orange markers are centred at -3, -5,
    // and -7 m, so attenuation and directional lighting have stable references.
    commands.spawn((
        Mesh3d(meshes.add(Plane3d::default().mesh().size(140.0, 140.0))),
        MeshMaterial3d(sand),
        Transform::from_xyz(0.0, -12.0, -8.0),
    ));

    let column = meshes.add(Cylinder::new(0.6, 8.0));
    let marker_mesh = meshes.add(Sphere::new(0.9));
    for (x, top_y, z) in [(-7.0, -3.0, -7.0), (0.0, -5.0, -4.0), (7.0, -7.0, -1.0)] {
        let height = top_y + 12.0;
        commands.spawn((
            Mesh3d(column.clone()),
            MeshMaterial3d(rock.clone()),
            Transform::from_xyz(x, -12.0 + height * 0.5, z).with_scale(Vec3::new(
                1.0,
                height / 8.0,
                1.0,
            )),
        ));
        commands.spawn((
            Mesh3d(marker_mesh.clone()),
            MeshMaterial3d(marker.clone()),
            Transform::from_xyz(x, top_y, z),
        ));
    }

    let boulder = meshes.add(Sphere::new(1.0));
    for (position, scale) in [
        (Vec3::new(-11.0, -10.9, -6.0), Vec3::new(2.8, 1.1, 2.0)),
        (Vec3::new(10.0, -10.5, -14.0), Vec3::new(2.1, 1.5, 2.7)),
        (Vec3::new(3.5, -11.1, -20.0), Vec3::new(3.4, 0.9, 2.4)),
    ] {
        commands.spawn((
            Mesh3d(boulder.clone()),
            MeshMaterial3d(rock.clone()),
            Transform::from_translation(position).with_scale(scale),
        ));
    }
}
