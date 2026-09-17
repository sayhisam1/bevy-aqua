//! Underwater volume composite for Aqua.

#![warn(unreachable_pub)]

use bevy::{
    asset::embedded_asset,
    prelude::*,
    render::extract_resource::{ExtractResource, ExtractResourcePlugin},
};
use bevy_aqua_core::{
    AquaSettings, Ocean, OceanView, ResolvedWaterBodies, ResolvedWaterBody, WaterBodiesResolved,
    WaterOptics,
};

mod render;

/// Adds the underwater volume composite after the main 3D pass.
#[derive(Debug, Default, Clone, Copy)]
pub struct AquaVolumePlugin;

impl Plugin for AquaVolumePlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "volume.wgsl");
        app.init_resource::<ExtractedVolume>()
            .add_plugins(ExtractResourcePlugin::<ExtractedVolume>::default())
            .add_systems(PostUpdate, detect_underwater.after(WaterBodiesResolved));
        render::add(app);
    }
}

/// GPU payload extracted each frame. Inactive frames skip the composite.
#[derive(Resource, Clone, ExtractResource, Debug)]
struct ExtractedVolume {
    active: bool,
    surface_level: f32,
    camera_y: f32,
    optics: WaterOptics,
}

impl Default for ExtractedVolume {
    fn default() -> Self {
        Self {
            active: false,
            surface_level: 0.0,
            camera_y: 0.0,
            optics: WaterOptics::DEEP_OCEAN,
        }
    }
}

/// Mean water plane and optics for the camera's containing body, if any.
fn sample_medium(
    camera: Vec3,
    ocean: Option<&Ocean>,
    settings: &AquaSettings,
    bodies: &[ResolvedWaterBody],
) -> Option<(f32, WaterOptics)> {
    let xz = camera.xz();
    let mut best: Option<(f32, WaterOptics)> = None;
    for body in bodies {
        if camera.y < body.level && body.contains(xz) {
            let optics = body.optics.unwrap_or(settings.water_optics);
            if best.is_none_or(|(level, _)| body.level > level) {
                best = Some((body.level, optics));
            }
        }
    }
    if let Some((level, optics)) = best {
        return Some((level, optics));
    }
    let ocean = ocean?;
    (camera.y < ocean.level).then_some((ocean.level, settings.water_optics))
}

fn detect_underwater(
    cameras: Query<&GlobalTransform, With<OceanView>>,
    ocean: Option<Res<Ocean>>,
    settings: Res<AquaSettings>,
    bodies: Res<ResolvedWaterBodies>,
    mut volume: ResMut<ExtractedVolume>,
) {
    let Some(transform) = cameras.iter().next() else {
        *volume = ExtractedVolume::default();
        return;
    };
    let camera = transform.translation();
    let Some((surface_level, optics)) =
        sample_medium(camera, ocean.as_deref(), &settings, &bodies.0)
    else {
        *volume = ExtractedVolume::default();
        return;
    };

    *volume = ExtractedVolume {
        active: true,
        surface_level,
        camera_y: camera.y,
        optics,
    };
}
