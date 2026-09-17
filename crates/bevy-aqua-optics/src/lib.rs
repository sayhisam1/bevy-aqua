#![warn(unreachable_pub)]

//! Depth, medium, and transmission optics for Aqua shaders.
//!
//! `medium.wgsl` is the shared water-medium integral (particle
//! Henyey-Greenstein plus molecular Rayleigh), the surface water-leaving
//! conversion (camera-ray path matching transmission, in-water radiance / n²),
//! and the water-to-air dielectric used by the underside window.
//! `optics.wgsl` consumes Aqua's registered cascade, material-type, wave,
//! foam, and shore contracts.
//! Register both before queuing the composed water material.

use bevy::{asset::embedded_asset, prelude::*};

#[derive(Resource)]
struct ShaderLibraries {
    _handles: Vec<Handle<Shader>>,
}

/// Registers and retains the layered Aqua lighting and optics modules.
pub fn add_shader(app: &mut App) {
    bevy_aqua_light::add_shader(app);
    embedded_asset!(app, "medium.wgsl");
    embedded_asset!(app, "optics.wgsl");
    let server = app.world().resource::<AssetServer>();
    app.insert_resource(ShaderLibraries {
        _handles: vec![
            server.load("embedded://bevy_aqua_optics/medium.wgsl"),
            server.load("embedded://bevy_aqua_optics/optics.wgsl"),
        ],
    });
}

#[cfg(test)]
mod refraction_tests;
