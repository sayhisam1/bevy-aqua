#![warn(unreachable_pub)]

//! Shared physical water-medium shader contracts for Aqua.
//!
//! The embedded WGSL module provides Beer-Lambert transmittance, directional
//! in-scatter, and dielectric interface helpers under `aqua::medium`.

use bevy::{asset::embedded_asset, prelude::*};

#[derive(Resource)]
struct ShaderLibrary {
    _handle: Handle<Shader>,
}

/// Registers and retains the `aqua::medium` WGSL module.
///
/// Repeated calls leave the first registration in place.
pub fn add_shader(app: &mut App) {
    if app.world().contains_resource::<ShaderLibrary>() {
        return;
    }
    embedded_asset!(app, "medium.wgsl");
    let handle = app
        .world()
        .resource::<AssetServer>()
        .load("embedded://bevy_aqua_medium/medium.wgsl");
    app.insert_resource(ShaderLibrary { _handle: handle });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registration_retains_one_strong_handle_and_second_call_is_a_noop() {
        let mut app = App::new();
        app.add_plugins((MinimalPlugins, AssetPlugin::default()))
            .init_asset::<Shader>()
            .init_asset_loader::<bevy::shader::ShaderLoader>();
        add_shader(&mut app);
        let first = app.world().resource::<ShaderLibrary>()._handle.clone();
        assert!(matches!(first, Handle::Strong(_)));
        assert_eq!(
            first.path().unwrap().to_string(),
            "embedded://bevy_aqua_medium/medium.wgsl"
        );
        app.world_mut().clear_trackers();
        add_shader(&mut app);
        let library = app.world().get_resource_ref::<ShaderLibrary>().unwrap();
        assert_eq!(library._handle.id(), first.id());
        assert!(!library.is_changed());

        // Once registered, no asset-registry or server access is needed again.
        app.world_mut().remove_resource::<AssetServer>();
        app.world_mut()
            .remove_resource::<bevy::asset::io::embedded::EmbeddedAssetRegistry>();
        add_shader(&mut app);
        assert_eq!(
            app.world().resource::<ShaderLibrary>()._handle.id(),
            first.id()
        );
    }
}
