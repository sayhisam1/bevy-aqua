//! Mirrored scene views for Aqua water surfaces.

mod camera;
mod prepass_culling;
mod resolve;

use bevy::prelude::*;

#[derive(Debug, Default, Clone, Copy)]
pub struct AquaReflectPlugin;

impl Plugin for AquaReflectPlugin {
    fn build(&self, app: &mut App) {
        camera::add(app);
        resolve::add(app);
        prepass_culling::add(app);
    }
}

/// Includes an entity and its descendants in Aqua's planar reflection cameras.
///
/// Reflected materials must write the depth prepass. Opaque and alpha-masked
/// materials are supported; alpha-blended materials and custom materials without
/// a depth prepass are not. Empty mirror pixels fall back to environment lighting.
#[derive(Component, Debug, Default, Clone, Copy)]
pub struct ReflectedInWater;
