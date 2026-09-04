//! Mirrored scene views for Aqua water surfaces.

mod camera;
mod resolve;

use bevy::prelude::*;

#[derive(Debug, Default, Clone, Copy)]
pub struct AquaReflectPlugin;

impl Plugin for AquaReflectPlugin {
    fn build(&self, app: &mut App) {
        camera::add(app);
        resolve::add(app);
    }
}

/// Opts an entity and its descendants into Aqua's planar reflection cameras.
#[derive(Component, Debug, Default, Clone, Copy)]
pub struct ReflectedInWater;
