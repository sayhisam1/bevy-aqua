#![warn(unreachable_pub)]

//! Incident and environment lighting primitives for Aqua shaders.
//!
//! The WGSL module consumes Aqua's registered cascade and material-type
//! contracts. Register it before queuing the composed water material.

use bevy::{asset::embedded_asset, prelude::*};

#[cfg(test)]
mod smith_tests;

#[derive(Resource)]
struct ShaderLibraries {
    _handles: Vec<Handle<Shader>>,
}

/// Registers and retains the import-only Aqua lighting module.
pub fn add_shader(app: &mut App) {
    embedded_asset!(app, "environment.wgsl");
    embedded_asset!(app, "incident.wgsl");
    let server = app.world().resource::<AssetServer>();
    app.insert_resource(ShaderLibraries {
        _handles: vec![
            server.load("embedded://bevy_aqua_light/environment.wgsl"),
            server.load("embedded://bevy_aqua_light/incident.wgsl"),
        ],
    });
}

#[cfg(test)]
mod environment_tests {
    #[test]
    fn environment_library_is_independent_of_material_bindings() {
        let environment = include_str!("environment.wgsl");
        assert!(environment.contains("#define_import_path aqua::light::environment"));
        assert!(!environment.contains("aqua::cascade"));
        assert!(!environment.contains("bevy_aqua_core::material"));
        assert!(!environment.contains("MATERIAL_BIND_GROUP"));
        assert!(environment.contains("fn sample_environment("));
        assert!(environment.contains("fn sample_diffuse_environment("));

        let incident = include_str!("incident.wgsl");
        assert!(!incident.contains("fn sample_environment("));
        assert!(!incident.contains("fn sample_diffuse_environment("));
    }
}
