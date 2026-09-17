#![warn(unreachable_pub)]

//! Depth, medium, and transmission optics for Aqua shaders.
//!
//! The WGSL module consumes Aqua's registered cascade, material-type, wave,
//! foam, and shore contracts. Register it before queuing the composed water
//! material.

use bevy::{asset::embedded_asset, prelude::*};

#[derive(Resource)]
struct ShaderLibraries {
    _handles: Vec<Handle<Shader>>,
}

/// Registers and retains the layered Aqua lighting and optics modules.
pub fn add_shader(app: &mut App) {
    bevy_aqua_light::add_shader(app);
    embedded_asset!(app, "optics.wgsl");
    let server = app.world().resource::<AssetServer>();
    app.insert_resource(ShaderLibraries {
        _handles: vec![server.load("embedded://bevy_aqua_optics/optics.wgsl")],
    });
}

#[cfg(test)]
mod refraction_tests;

#[cfg(test)]
mod directional_exposure_tests;

#[cfg(test)]
mod fresnel_tests;
