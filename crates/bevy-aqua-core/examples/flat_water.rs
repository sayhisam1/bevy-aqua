//! Headless example that embeds the shaders and creates a cascade material.
//!
//! ```none
//! cargo run -p bevy-aqua-core --example flat_water
//! ```

use bevy::prelude::*;
use bevy_aqua_core::{CascadeMaterial, FieldParams, GpuLayout, layout};

fn main() {
    let mut app = App::new();
    app.add_plugins(MinimalPlugins)
        .init_resource::<Assets<CascadeMaterial>>()
        .add_plugins(AssetPlugin::default())
        .init_asset::<Shader>();
    // Registers the core contracts plus their transitive light/optics modules;
    // callers do not wire those shader libraries themselves.
    bevy_aqua_core::add_shader(&mut app);
    app.add_systems(Startup, build);
    app.update();

    let materials = app.world().resource::<Assets<CascadeMaterial>>();
    assert_eq!(materials.len(), 1, "material instantiated");
    println!("core registration OK: material and owned WGSL modules are available");
}

fn build(mut materials: ResMut<Assets<CascadeMaterial>>) {
    let cascades = layout(Vec2::ZERO);
    let material = CascadeMaterial {
        texture: Handle::default(),
        layout: GpuLayout::new(&cascades, Vec2::ZERO, 0.0),
        surface: Default::default(),
        sea_floor: Handle::default(),
        detail_normal: Handle::default(),
        foam: Handle::default(),
        foam_pattern: Handle::default(),
        fields: FieldParams::none(),
        field_maps: Handle::default(),
        reflection_a: Handle::default(),
        reflection_b: Handle::default(),
        reflections: Default::default(),
    };
    // Fixed uniform ABI: six cascade slots, last one a zero-weight sentinel.
    assert_eq!(material.layout.cascades.len(), 6);
    assert_eq!(material.layout.cascades[5].weight, 0.0);
    materials.add(material);
}
