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

#[cfg(test)]
mod far_scatter_tests {
    // Keep the endpoint partition visible: near scales directional_scatter,
    // then shade_water_body adds substrate diffuse/Lambert without the scale.
    #[test]
    fn far_scales_volume_scatter_but_not_substrate_or_reflection() {
        let source = include_str!("optics.wgsl");
        let far = source
            .split("fn far_field_water(")
            .nth(1)
            .unwrap()
            .split("fn camera_view_position(")
            .next()
            .unwrap();
        assert!(source.contains("invocation_extinction, invocation_scatter_scale,"));
        assert!(far.contains("let scatter_scale = invocation_scatter_scale();"));
        assert!(far.contains("body_albedo * scatter_scale + GODOT_WATER_ALBEDO"));
        assert!(far.contains("* light_radiance * GODOT_WATER_ALBEDO * scatter_scale;"));
        assert!(far.contains("body += lambertian * light_radiance * GODOT_WATER_ALBEDO;"));
        assert_eq!(far.matches("scatter_scale").count(), 4);
        assert!(far.contains("return mix(body, reflected_radiance, reflection_weight);"));
    }

    #[test]
    fn far_scatter_scale_preserves_unscaled_energy_lanes() {
        // One channel of a flat, opaque, foam-free reference pixel. Values
        // are scene-linear radiance, not screenshot/tonemap measurements.
        let volume = 0.17_f32;
        let sss = 0.04_f32;
        let substrate = 0.02_f32;
        let lambert = 0.03_f32;
        let reflection = 0.40_f32;
        let fresnel = 0.02_f32;
        let reference = |scale: f32| {
            ((volume + sss) * scale + substrate + lambert) * (1.0 - fresnel) + reflection * fresnel
        };
        for scale in [0.0_f32, 0.1, 0.18, 1.0] {
            let far = (volume * scale + substrate + lambert + sss * scale) * (1.0 - fresnel)
                + reflection * fresnel;
            assert!((far - reference(scale)).abs() < 1e-6);
        }
        let legacy = reference(1.0);
        assert!((legacy - reference(0.1) - 0.18522).abs() < 1e-6);
        assert!((reference(0.0) - 0.057).abs() < 1e-6);
    }
}
