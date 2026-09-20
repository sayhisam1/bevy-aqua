//! Runs the foam plugin headlessly with a hand-assembled cascade [`Data`].
//!
//! ```none
//! cargo run -p bevy-aqua-foam --example foam_only
//! ```

use bevy::prelude::*;
use bevy_aqua_core::{
    OceanWaves, WaterFields, bed,
    cascade::{GpuLayout, layout, make_fft_surface_texture, make_texture},
};

fn main() {
    let mut app = App::new();
    app.insert_resource(OceanWaves::default());
    app.add_plugins((MinimalPlugins, AssetPlugin::default()));
    // No ImagePlugin headlessly: seed the stores the sim touches.
    app.init_resource::<Assets<Image>>();
    app.init_resource::<WaterFields>();
    bed::add(&mut app);
    app.add_systems(PreStartup, assemble_data);

    app.add_plugins(bevy_aqua_foam::AquaFoamPlugin);
    for _ in 0..3 {
        app.update();
    }

    let textures = app.world().resource::<bevy_aqua_foam::Textures>();
    assert!(textures.state_a.is_strong() && textures.state_b.is_strong());
    assert!(textures.surface.is_strong() && textures.pattern.is_strong());
    println!("foam isolation OK: pool seeded, sim frames advanced");
}

fn assemble_data(mut commands: Commands, mut images: ResMut<Assets<Image>>) {
    let texture = images.add(make_texture());
    let fft_surface = images.add(make_fft_surface_texture());
    let cascades = layout(Vec2::ZERO);
    commands.insert_resource(bevy_aqua_core::Data::new(
        Handle::default(),
        texture,
        fft_surface,
        GpuLayout::new(&cascades, Vec2::ZERO, 0.0),
    ));
}
