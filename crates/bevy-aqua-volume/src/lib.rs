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

/// Controls approximations used by the fullscreen underwater pass.
#[derive(Resource, Debug, Clone, Copy, PartialEq)]
pub struct UnderwaterSettings {
    /// Attenuate the already-shaded opaque scene by an estimated downwelling path.
    /// Disable this when emissive or local-light energy must remain unchanged.
    pub receiver_relighting: bool,
}

impl Default for UnderwaterSettings {
    fn default() -> Self {
        Self {
            receiver_relighting: false,
        }
    }
}

/// Adds the underwater volume composite after the main 3D pass.
#[derive(Debug, Default, Clone, Copy)]
pub struct AquaVolumePlugin;

impl Plugin for AquaVolumePlugin {
    fn build(&self, app: &mut App) {
        // Keep direct use sound: the volume shader imports `aqua::medium`.
        bevy_aqua_optics::add_shader(app);
        embedded_asset!(app, "volume.wgsl");
        app.init_resource::<UnderwaterSettings>()
            .init_resource::<ExtractedVolume>()
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
    receiver_relighting: bool,
}

impl Default for ExtractedVolume {
    fn default() -> Self {
        Self {
            active: false,
            surface_level: 0.0,
            camera_y: 0.0,
            optics: WaterOptics::DEEP_OCEAN,
            receiver_relighting: false,
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
    volume_settings: Res<UnderwaterSettings>,
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
        receiver_relighting: volume_settings.receiver_relighting,
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_aqua_core::WaterShape;

    fn body(level: f32, radius: f32, optics: Option<WaterOptics>) -> ResolvedWaterBody {
        ResolvedWaterBody::resolve(
            Entity::PLACEHOLDER,
            &WaterShape::Circle { radius },
            optics,
            &GlobalTransform::from_translation(Vec3::new(0.0, level, 0.0)),
        )
        .unwrap()
    }

    #[test]
    fn highest_containing_body_wins_before_ocean() {
        let settings = AquaSettings::default();
        let low = body(1.0, 10.0, Some(WaterOptics::COASTAL));
        let high = body(3.0, 2.0, Some(WaterOptics::TROPICAL));
        let ocean = Ocean { level: 5.0 };
        let sampled = sample_medium(
            Vec3::new(0.0, 0.0, 0.0),
            Some(&ocean),
            &settings,
            &[low, high],
        )
        .unwrap();
        assert_eq!(sampled, (3.0, WaterOptics::TROPICAL));
    }

    #[test]
    fn containment_and_mean_plane_are_strict() {
        let settings = AquaSettings::default();
        let bounded = body(2.0, 1.0, None);
        assert!(
            sample_medium(
                Vec3::new(0.0, 2.0, 0.0),
                None,
                &settings,
                &[bounded.clone()]
            )
            .is_none()
        );
        assert!(sample_medium(Vec3::new(2.0, 0.0, 0.0), None, &settings, &[bounded]).is_none());
        assert!(
            sample_medium(
                Vec3::new(0.0, 0.0, 0.0),
                Some(&Ocean { level: 0.0 }),
                &settings,
                &[],
            )
            .is_none()
        );
    }
}
