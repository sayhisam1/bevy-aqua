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
    embedded_asset!(app, "medium.wgsl");
    embedded_asset!(app, "screen.wgsl");
    embedded_asset!(app, "ssr.wgsl");
    embedded_asset!(app, "optics.wgsl");
    let server = app.world().resource::<AssetServer>();
    app.insert_resource(ShaderLibraries {
        _handles: vec![
            server.load("embedded://bevy_aqua_optics/medium.wgsl"),
            server.load("embedded://bevy_aqua_optics/screen.wgsl"),
            server.load("embedded://bevy_aqua_optics/ssr.wgsl"),
            server.load("embedded://bevy_aqua_optics/optics.wgsl"),
        ],
    });
}

#[cfg(test)]
mod refraction_tests;

#[cfg(test)]
mod directional_exposure_tests;

#[cfg(test)]
mod fresnel_tests;

#[cfg(test)]
mod far_scatter_tests {
    #[test]
    fn far_uses_shared_medium_then_unscaled_substrate() {
        let source = include_str!("optics.wgsl");
        let far = source
            .split("fn far_field_water(")
            .nth(1)
            .unwrap()
            .split("fn empty_camera_depth_path(")
            .next()
            .unwrap();
        assert!(source.contains("fn surface_medium_radiance("));
        assert!(source.contains("water_leaving_radiance("));
        assert!(source.contains("invocation_scatter_scale()"));
        assert!(far.contains("surface_medium_radiance(vec3(0.0), to_view, t_end)"));
        assert!(far.contains("body += diffuse_irradiance * GODOT_WATER_ALBEDO;"));
        assert!(far.contains("body += lambertian * light_radiance * GODOT_WATER_ALBEDO;"));
        assert!(!far.contains("body_albedo * scatter_scale"));
        assert!(far.contains("return mix(body, reflected_radiance, reflection_weight);"));
    }
}

#[cfg(test)]
mod far_opacity_tests;

#[cfg(test)]
mod resolved_normal_tests;

#[cfg(test)]
mod caustic_receiver_tests;

#[cfg(test)]
mod medium_tests;

#[cfg(test)]
mod ssr_tests;
