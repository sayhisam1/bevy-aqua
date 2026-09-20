//! Cycles through every Aqua diagnostic view over a sloped seabed.
//!
//! Run with `cargo run --example debug_views`. Browser instructions are in
//! `examples/README.md`.

use bevy::{
    camera::{Exposure, Hdr},
    core_pipeline::prepass::DepthPrepass,
    light::{Atmosphere, AtmosphereEnvironmentMapLight, atmosphere::ScatteringMedium},
    pbr::AtmosphereSettings,
    prelude::*,
};
use bevy_aqua::{
    AquaDebug, AquaPlugin, AquaSettings, BedHeightMap, Ocean, OceanWaves, ReflectionMode, SeaState,
};

const RESOLUTION: u32 = 65;
const STEP: f32 = 2.0;
const ORIGIN: Vec2 = Vec2::splat(-64.0);
const BEACH_SLOPE: f32 = 0.06;
const BEACH_HEIGHT: f32 = -3.0;
const VIEW_SECONDS: f32 = 3.0;

const DEBUG_VIEWS: [(AquaDebug, &str); 12] = [
    (AquaDebug::WaveHeight, "Wave height"),
    (AquaDebug::FoamDensity, "Foam density"),
    (AquaDebug::WaterPath, "Water path"),
    (AquaDebug::RefractionValidity, "Refraction validity"),
    (AquaDebug::Transmission, "Transmission"),
    (
        AquaDebug::TransmissionUnrefracted,
        "Transmission (unrefracted)",
    ),
    (AquaDebug::BeerLambert, "Beer-Lambert"),
    (AquaDebug::SeaFloorDepth, "Sea-floor depth"),
    (AquaDebug::ReflectionSanity, "Reflection sanity"),
    (AquaDebug::LightRadiance, "Light radiance"),
    (AquaDebug::FarTier, "Far-tier weight"),
    (AquaDebug::ReflectionFraction, "Reflection fraction"),
];

#[derive(Resource)]
struct DebugCycle {
    timer: Timer,
    index: usize,
}

#[derive(Component)]
struct DebugLabel;

fn main() {
    let mut app = App::new();
    app.insert_resource(ClearColor(Color::srgb(0.55, 0.75, 0.9)))
        .add_plugins(DefaultPlugins);

    let bed = {
        let mut images = app.world_mut().resource_mut::<Assets<Image>>();
        BedHeightMap::from_height_fn(&mut images, beach_height, RESOLUTION, ORIGIN, STEP)
    };
    app.insert_resource(bed)
        .insert_resource(Ocean::default())
        .insert_resource(OceanWaves {
            sea_state: SeaState::Rough,
            shallow_water_attenuation: 1.0,
            ..default()
        })
        .insert_resource(AquaSettings {
            reflections: ReflectionMode::Cubemap,
            ..default()
        })
        .insert_resource(DEBUG_VIEWS[0].0)
        .insert_resource(DebugCycle {
            timer: Timer::from_seconds(VIEW_SECONDS, TimerMode::Repeating),
            index: 0,
        })
        .add_plugins(AquaPlugin)
        .add_systems(Startup, setup)
        .add_systems(Update, cycle_debug_view)
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
        Transform::from_xyz(-18.0, 13.0, 36.0).looking_at(Vec3::new(12.0, 0.0, 0.0), Vec3::Y),
    ));
    commands.spawn((
        DirectionalLight {
            illuminance: 18_000.0,
            shadow_maps_enabled: false,
            ..default()
        },
        Transform::from_rotation(Quat::from_euler(EulerRot::XYZ, -0.9, -0.6, 0.0)),
    ));
    commands.spawn((
        DebugLabel,
        Text::new(DEBUG_VIEWS[0].1),
        TextFont {
            font_size: FontSize::Px(28.0),
            ..default()
        },
        TextColor(Color::WHITE),
        BackgroundColor(Color::srgba(0.0, 0.0, 0.0, 0.8)),
        Node {
            position_type: PositionType::Absolute,
            left: Val::Px(24.0),
            top: Val::Px(20.0),
            padding: UiRect::all(Val::Px(8.0)),
            ..default()
        },
    ));
}

fn cycle_debug_view(
    time: Res<Time>,
    mut cycle: ResMut<DebugCycle>,
    mut debug: ResMut<AquaDebug>,
    mut labels: Query<&mut Text, With<DebugLabel>>,
) {
    if !cycle.timer.tick(time.delta()).just_finished() {
        return;
    }
    cycle.index = (cycle.index + 1) % DEBUG_VIEWS.len();
    let (next, label) = DEBUG_VIEWS[cycle.index];
    *debug = next;
    for mut text in &mut labels {
        **text = label.into();
    }
}

fn beach_height(x: f32, _z: f32) -> f32 {
    BEACH_HEIGHT + BEACH_SLOPE * x
}
