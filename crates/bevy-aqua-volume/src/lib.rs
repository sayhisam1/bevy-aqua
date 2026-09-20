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
use bevy_aqua_query::{WaveQuery, WaveSurface};

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
        // Keep direct use sound: the volume shader imports `aqua::medium`, and
        // camera admission needs the GPU query runtime as well as its types.
        bevy_aqua_optics::add_shader(app);
        if !app.is_plugin_added::<bevy_aqua_query::AquaQueryPlugin>() {
            app.add_plugins(bevy_aqua_query::AquaQueryPlugin);
        }
        embedded_asset!(app, "volume.wgsl");
        app.init_resource::<UnderwaterSettings>()
            .init_resource::<ExtractedVolume>()
            .add_plugins(ExtractResourcePlugin::<ExtractedVolume>::default())
            .add_systems(
                PostUpdate,
                (attach_camera_queries, detect_underwater)
                    .chain()
                    .after(WaterBodiesResolved),
            );
        render::add(app);
    }
}

/// GPU payload extracted each frame. Inactive frames skip the composite.
#[derive(Resource, Clone, ExtractResource, Debug)]
struct ExtractedVolume {
    active: bool,
    surface_level: f32,
    camera_y: f32,
    interface_normal: Vec3,
    unbounded_ocean: bool,
    optics: WaterOptics,
    receiver_relighting: bool,
}

impl Default for ExtractedVolume {
    fn default() -> Self {
        Self {
            active: false,
            surface_level: 0.0,
            camera_y: 0.0,
            interface_normal: Vec3::Y,
            unbounded_ocean: false,
            optics: WaterOptics::DEEP_OCEAN,
            receiver_relighting: false,
        }
    }
}

/// Mean water plane and optics for the horizontally selected medium.
///
/// Bounded bodies have priority over the ocean. Their vertical admission is
/// deliberately deferred until after the displaced surface level is known.
fn sample_medium(
    camera_xz: Vec2,
    ocean: Option<&Ocean>,
    settings: &AquaSettings,
    bodies: &[ResolvedWaterBody],
) -> Option<(f32, WaterOptics, bool)> {
    let mut best: Option<(f32, WaterOptics)> = None;
    for body in bodies {
        if body.contains(camera_xz) {
            let optics = body.optics.unwrap_or(settings.water_optics);
            if best.is_none_or(|(level, _)| body.level > level) {
                best = Some((body.level, optics));
            }
        }
    }
    best.map(|(level, optics)| (level, optics, false))
        .or_else(|| ocean.map(|ocean| (ocean.level, settings.water_optics, true)))
}

fn actual_surface_level(mean_level: f32, surface: &WaveSurface) -> f32 {
    if surface.valid && surface.displacement.y.is_finite() {
        mean_level + surface.displacement.y
    } else {
        mean_level
    }
}

fn interface_normal(surface: &WaveSurface) -> Vec3 {
    if !surface.valid || !surface.normal.is_finite() {
        return Vec3::Y;
    }
    let normal = surface.normal.normalize_or_zero();
    if normal.length_squared() > 0.0 {
        normal
    } else {
        Vec3::Y
    }
}

fn is_underwater(camera_y: f32, surface_level: f32) -> bool {
    camera_y < surface_level
}

fn attach_camera_queries(
    mut commands: Commands,
    cameras: Query<(Entity, &Camera), (With<OceanView>, Without<WaveQuery>)>,
) {
    for (entity, camera) in &cameras {
        if camera.is_active {
            commands.entity(entity).insert(WaveQuery);
        }
    }
}

fn detect_underwater(
    cameras: Query<(&Camera, &GlobalTransform, &WaveSurface), With<OceanView>>,
    ocean: Option<Res<Ocean>>,
    settings: Res<AquaSettings>,
    bodies: Res<ResolvedWaterBodies>,
    mut volume: ResMut<ExtractedVolume>,
    volume_settings: Res<UnderwaterSettings>,
) {
    let Some((_, transform, wave_surface)) = cameras.iter().find(|(camera, ..)| camera.is_active)
    else {
        *volume = ExtractedVolume::default();
        return;
    };
    let camera = transform.translation();
    let Some((mean_level, optics, unbounded_ocean)) =
        sample_medium(camera.xz(), ocean.as_deref(), &settings, &bodies.0)
    else {
        *volume = ExtractedVolume::default();
        return;
    };
    let surface_level = actual_surface_level(mean_level, wave_surface);

    *volume = ExtractedVolume {
        active: is_underwater(camera.y, surface_level),
        surface_level,
        camera_y: camera.y,
        interface_normal: interface_normal(wave_surface),
        unbounded_ocean,
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

    fn surface(displacement_y: f32, valid: bool) -> WaveSurface {
        WaveSurface {
            displacement: Vec3::Y * displacement_y,
            valid,
            ..default()
        }
    }

    #[test]
    fn above_mean_but_under_crest_is_active() {
        let level = actual_surface_level(2.0, &surface(0.75, true));
        assert_eq!(level, 2.75);
        assert!(is_underwater(2.5, level));
    }

    #[test]
    fn below_mean_but_above_trough_is_inactive() {
        let level = actual_surface_level(2.0, &surface(-0.75, true));
        assert_eq!(level, 1.25);
        assert!(!is_underwater(1.5, level));
    }

    #[test]
    fn invalid_query_falls_back_to_mean_plane() {
        let level = actual_surface_level(2.0, &surface(20.0, false));
        assert_eq!(level, 2.0);
        assert!(is_underwater(1.9, level));
        assert!(!is_underwater(2.1, level));
    }

    #[test]
    fn interface_normal_is_normalized_and_invalid_values_fall_back_up() {
        let mut sample = surface(0.0, true);
        sample.normal = Vec3::new(0.0, 2.0, 2.0);
        assert!(
            interface_normal(&sample)
                .abs_diff_eq(Vec3::new(0.0, 0.5_f32.sqrt(), 0.5_f32.sqrt()), 1e-6)
        );

        sample.normal = Vec3::ZERO;
        assert_eq!(interface_normal(&sample), Vec3::Y);
        sample.normal = Vec3::NAN;
        assert_eq!(interface_normal(&sample), Vec3::Y);
        sample.valid = false;
        sample.normal = Vec3::X;
        assert_eq!(interface_normal(&sample), Vec3::Y);
    }

    #[test]
    fn exact_displaced_boundary_is_dry() {
        let level = actual_surface_level(2.0, &surface(0.5, true));
        assert!(!is_underwater(level, level));
    }

    #[test]
    fn highest_containing_body_wins_before_ocean() {
        let settings = AquaSettings::default();
        let low = body(1.0, 10.0, Some(WaterOptics::COASTAL));
        let high = body(3.0, 2.0, Some(WaterOptics::TROPICAL));
        let ocean = Ocean { level: 5.0 };
        let sampled = sample_medium(Vec2::ZERO, Some(&ocean), &settings, &[low, high]).unwrap();
        assert_eq!(sampled, (3.0, WaterOptics::TROPICAL, false));
    }

    #[test]
    fn ocean_is_used_only_without_a_containing_body() {
        let settings = AquaSettings::default();
        let bounded = body(2.0, 1.0, Some(WaterOptics::COASTAL));
        let ocean = Ocean { level: 5.0 };
        assert_eq!(
            sample_medium(Vec2::ZERO, Some(&ocean), &settings, &[bounded.clone()]),
            Some((2.0, WaterOptics::COASTAL, false))
        );
        assert_eq!(
            sample_medium(Vec2::splat(2.0), Some(&ocean), &settings, &[bounded]),
            Some((5.0, settings.water_optics, true))
        );
    }

    #[test]
    fn active_ocean_view_gets_wave_query() {
        let mut app = App::new();
        app.add_systems(Update, attach_camera_queries);
        let active = app.world_mut().spawn((Camera::default(), OceanView)).id();
        let inactive = app
            .world_mut()
            .spawn((
                Camera {
                    is_active: false,
                    ..default()
                },
                OceanView,
            ))
            .id();

        app.update();

        assert!(app.world().get::<WaveQuery>(active).is_some());
        assert!(app.world().get::<WaveSurface>(active).is_some());
        assert!(app.world().get::<WaveQuery>(inactive).is_none());
    }
}
