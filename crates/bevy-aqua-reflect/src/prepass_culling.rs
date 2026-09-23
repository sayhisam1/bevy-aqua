//! Temporary Bevy 0.19.1 prepass view-key compatibility shim.
//! Keep upstream's key check; hide only our added bit while it compares keys.
use crate::resolve::ReflectionOutput;
use bevy::{
    pbr::{MeshPipelineKey, ViewKeyPrepassCache, check_prepass_views_need_specialization},
    prelude::*,
    render::{
        Render, RenderApp, RenderSystems,
        camera::DirtySpecializations,
        view::{ExtractedView, RetainedViewEntity},
    },
};

#[derive(Resource, Default)]
struct CompatibilityState {
    upstream_handles_inversion: bool,
    // A per-frame scratch list, not a second specialization cache.
    views: Vec<(RetainedViewEntity, Option<bool>, bool)>,
}

pub(crate) fn add(app: &mut App) {
    let Some(render) = app.get_sub_app_mut(RenderApp) else {
        return;
    };
    render.init_resource::<CompatibilityState>().add_systems(
        Render,
        (
            strip_local_bit
                .in_set(RenderSystems::PrepareAssets)
                .before(check_prepass_views_need_specialization),
            restore_local_bit
                .in_set(RenderSystems::PrepareAssets)
                .after(check_prepass_views_need_specialization)
                .after(strip_local_bit),
        ),
    );
}

fn strip_local_bit(
    cache: Option<ResMut<ViewKeyPrepassCache>>,
    mut state: ResMut<CompatibilityState>,
    views: Query<(&ExtractedView, &Msaa), With<ReflectionOutput>>,
) {
    if state.upstream_handles_inversion {
        return;
    }
    state.views.clear();
    let Some(mut cache) = cache else {
        return;
    };
    // Temporary changes must not trigger change detection on stable frames.
    let cache = cache.bypass_change_detection();
    for (view, _) in &views {
        let previous = cache.get_mut(&view.retained_view_entity).map(|key| {
            let previous = key.contains(MeshPipelineKey::INVERT_CULLING);
            key.remove(MeshPipelineKey::INVERT_CULLING);
            previous
        });
        state
            .views
            .push((view.retained_view_entity, previous, view.invert_culling));
    }
}

fn restore_local_bit(
    cache: Option<ResMut<ViewKeyPrepassCache>>,
    dirty: Option<ResMut<DirtySpecializations>>,
    mut state: ResMut<CompatibilityState>,
) {
    if state.upstream_handles_inversion {
        return;
    }
    let (Some(mut cache), Some(mut dirty)) = (cache, dirty) else {
        return;
    };
    let mut upstream_handles_inversion = false;
    let mut inversion_changed = false;
    for &(entity, previous, desired) in &state.views {
        let Some(key) = cache.bypass_change_detection().get_mut(&entity) else {
            continue;
        };
        // We stripped this bit. If it returned, upstream implements the fix.
        upstream_handles_inversion |= desired && key.contains(MeshPipelineKey::INVERT_CULLING);
        key.set(MeshPipelineKey::INVERT_CULLING, desired);
        if previous.is_some_and(|previous| previous != desired) {
            inversion_changed = true;
            dirty.views.insert(entity);
        }
        // New entries and all non-inversion changes are already dirtied upstream.
    }
    if inversion_changed {
        cache.set_changed();
    }
    state.upstream_handles_inversion = upstream_handles_inversion;
    state.views.clear();
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::{
        core_pipeline::prepass::{DepthPrepass, MotionVectorPrepass, NormalPrepass},
        render::render_resource::TextureFormat,
    };

    fn fixture(fixed: bool) -> (World, Schedule, Entity, RetainedViewEntity) {
        let mut world = World::new();
        world.init_resource::<ViewKeyPrepassCache>();
        world.init_resource::<DirtySpecializations>();
        world.init_resource::<CompatibilityState>();
        let entity = world.spawn_empty().id();
        let retained = RetainedViewEntity::new(entity.into(), None, 0);
        world.entity_mut(entity).insert((
            ExtractedView {
                retained_view_entity: retained,
                clip_from_view: Mat4::IDENTITY,
                world_from_view: GlobalTransform::IDENTITY,
                clip_from_world: None,
                target_format: TextureFormat::Rgba16Float,
                viewport: UVec4::new(0, 0, 64, 64),
                color_grading: default(),
                invert_culling: true,
            },
            Msaa::Off,
            DepthPrepass,
            ReflectionOutput(Handle::default()),
        ));
        let mut schedule = Schedule::default();
        if fixed {
            schedule.add_systems((strip_local_bit, fixed_checker, restore_local_bit).chain());
        } else {
            schedule.add_systems(
                (
                    strip_local_bit,
                    check_prepass_views_need_specialization,
                    restore_local_bit,
                )
                    .chain(),
            );
        }
        (world, schedule, entity, retained)
    }
    fn frame(world: &mut World, schedule: &mut Schedule) {
        world.resource_mut::<DirtySpecializations>().views.clear();
        schedule.run(world);
    }
    fn inverted(world: &World, retained: RetainedViewEntity) -> bool {
        world.resource::<ViewKeyPrepassCache>()[&retained].contains(MeshPipelineKey::INVERT_CULLING)
    }
    fn clean(world: &World) {
        assert!(world.resource::<DirtySpecializations>().views.is_empty());
    }

    #[test]
    fn ordinary_key_changes_and_unrelated_dirty_views_are_preserved() {
        let (mut world, mut schedule, entity, retained) = fixture(false);
        frame(&mut world, &mut schedule);
        world
            .entity_mut(entity)
            .insert((Msaa::Sample4, NormalPrepass, MotionVectorPrepass));
        frame(&mut world, &mut schedule);
        let key = world.resource::<ViewKeyPrepassCache>()[&retained];
        assert_eq!(key.msaa_samples(), 4);
        assert!(key.contains(
            MeshPipelineKey::INVERT_CULLING
                | MeshPipelineKey::NORMAL_PREPASS
                | MeshPipelineKey::MOTION_VECTOR_PREPASS
        ));
        assert!(
            world
                .resource::<DirtySpecializations>()
                .views
                .contains(&retained)
        );
        frame(&mut world, &mut schedule);
        clean(&world);
        let other = RetainedViewEntity::new(world.spawn_empty().id().into(), None, 0);
        world
            .resource_mut::<DirtySpecializations>()
            .views
            .insert(other);
        schedule.run(&mut world);
        assert!(
            world
                .resource::<DirtySpecializations>()
                .views
                .contains(&other)
        );
        assert!(
            !world
                .resource::<DirtySpecializations>()
                .views
                .contains(&retained)
        );
    }
    #[test]
    fn mixed_views_dirty_only_the_mirror_when_inversion_changes() {
        let (mut world, mut schedule, mirror, mirror_retained) = fixture(false);
        let ordinary = world.spawn_empty().id();
        let ordinary_retained = RetainedViewEntity::new(ordinary.into(), None, 0);
        world.entity_mut(ordinary).insert((
            ExtractedView {
                retained_view_entity: ordinary_retained,
                clip_from_view: Mat4::IDENTITY,
                world_from_view: GlobalTransform::IDENTITY,
                clip_from_world: None,
                target_format: TextureFormat::Rgba16Float,
                viewport: UVec4::new(0, 0, 64, 64),
                color_grading: default(),
                invert_culling: false,
            },
            Msaa::Off,
            DepthPrepass,
        ));
        frame(&mut world, &mut schedule);
        let dirty = &world.resource::<DirtySpecializations>().views;
        assert_eq!(dirty.len(), 2);
        assert!(dirty.contains(&mirror_retained));
        assert!(dirty.contains(&ordinary_retained));
        assert!(inverted(&world, mirror_retained));
        let ordinary_key = world.resource::<ViewKeyPrepassCache>()[&ordinary_retained];
        assert!(!ordinary_key.contains(MeshPipelineKey::INVERT_CULLING));
        for _ in 0..100 {
            frame(&mut world, &mut schedule);
            clean(&world);
        }

        for desired in [false, true] {
            world
                .get_mut::<ExtractedView>(mirror)
                .unwrap()
                .invert_culling = desired;
            frame(&mut world, &mut schedule);
            let dirty = &world.resource::<DirtySpecializations>().views;
            assert_eq!(dirty.len(), 1);
            assert!(dirty.contains(&mirror_retained));
            assert!(!dirty.contains(&ordinary_retained));
            assert_eq!(inverted(&world, mirror_retained), desired);
            assert_eq!(
                world.resource::<ViewKeyPrepassCache>()[&ordinary_retained],
                ordinary_key
            );
            frame(&mut world, &mut schedule);
            clean(&world);
        }
    }
    #[test]
    fn marker_removal_leaves_upstream_in_charge_and_no_scratch_state() {
        let (mut world, mut schedule, entity, retained) = fixture(false);
        frame(&mut world, &mut schedule);
        world.entity_mut(entity).remove::<ReflectionOutput>();
        frame(&mut world, &mut schedule);
        // Current upstream omits inversion for non-mirrors; shim does not touch them.
        assert!(!inverted(&world, retained));
        assert!(world.resource::<CompatibilityState>().views.is_empty());
        frame(&mut world, &mut schedule);
        clean(&world);
    }
    // Model the upstream fix, for this depth-only fixture, independently of
    // the installed upstream checker used by all other tests.
    fn fixed_checker(
        mut cache: ResMut<ViewKeyPrepassCache>,
        mut dirty: ResMut<DirtySpecializations>,
        views: Query<(&ExtractedView, &Msaa)>,
    ) {
        for (view, msaa) in &views {
            let mut key =
                MeshPipelineKey::from_msaa_samples(msaa.samples()) | MeshPipelineKey::DEPTH_PREPASS;
            key.set(MeshPipelineKey::INVERT_CULLING, view.invert_culling);
            if cache.get(&view.retained_view_entity) != Some(&key) {
                cache.insert(view.retained_view_entity, key);
                dirty.views.insert(view.retained_view_entity);
            }
        }
    }
    #[test]
    fn fixed_upstream_disables_shim_without_recurring_dirty_frames() {
        let (mut world, mut schedule, entity, retained) = fixture(true);
        world
            .get_mut::<ExtractedView>(entity)
            .unwrap()
            .invert_culling = false;
        frame(&mut world, &mut schedule);
        assert!(
            !world
                .resource::<CompatibilityState>()
                .upstream_handles_inversion
        );
        world
            .get_mut::<ExtractedView>(entity)
            .unwrap()
            .invert_culling = true;
        frame(&mut world, &mut schedule);
        assert!(
            world
                .resource::<CompatibilityState>()
                .upstream_handles_inversion
        );
        assert!(inverted(&world, retained));
        for _ in 0..100 {
            frame(&mut world, &mut schedule);
            clean(&world);
        }
        world
            .get_mut::<ExtractedView>(entity)
            .unwrap()
            .invert_culling = false;
        frame(&mut world, &mut schedule);
        assert!(!inverted(&world, retained));
        frame(&mut world, &mut schedule);
        clean(&world);
    }
}
