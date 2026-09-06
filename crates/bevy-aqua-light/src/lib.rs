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
    embedded_asset!(app, "incident.wgsl");
    let server = app.world().resource::<AssetServer>();
    app.insert_resource(ShaderLibraries {
        _handles: vec![server.load("embedded://bevy_aqua_light/incident.wgsl")],
    });
}
