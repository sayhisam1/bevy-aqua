use bevy::prelude::*;
use bevy_aqua_core::{
    OceanWaves, WaterFields, bed,
    cascade::{GpuLayout, layout, make_texture},
};

fn main() {
    let mut app = App::new();
    app.insert_resource(OceanWaves {
        model: bevy_aqua_core::WaveModel::Analytic,
        ..default()
    });
    app.add_plugins((MinimalPlugins, AssetPlugin::default()));
    // No ImagePlugin headlessly: seed the stores the producers touch.
    app.init_resource::<Assets<Image>>();
    app.init_resource::<WaterFields>();
    bed::add(&mut app);
    // Stand-in for the umbrella glue this example replaces: assemble the
    // shared cascade data before any producer starts (PreStartup < Startup).
    app.add_systems(PreStartup, assemble_data);

    app.add_plugins(bevy_aqua_waves::AquaWavesPlugin);
    app.update();

    let frame = app.world().resource::<bevy_aqua_waves::Frame>();
    let ranges = frame.uniform().ranges;
    assert_eq!(ranges[0].x, 0, "first band starts at slot zero");
    for pair in ranges.windows(2) {
        assert_eq!(pair[0].y, pair[1].x, "bands partition without gaps");
    }
    assert_eq!(ranges[4].y, 40, "all WAVE_SLOTS assigned");
    println!("waves isolation OK: bands partition [0..40] across cascades");
}

fn assemble_data(mut commands: Commands, mut images: ResMut<Assets<Image>>) {
    let texture = images.add(make_texture());
    let cascades = layout(Vec2::ZERO);
    commands.insert_resource(bevy_aqua_core::Data::new(
        Handle::default(),
        texture,
        GpuLayout::new(&cascades, Vec2::ZERO, 0.0),
    ));
}
