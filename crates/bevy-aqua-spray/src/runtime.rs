//! Probe placement, signal filtering, culling, and bounded burst dispatch.

use bevy::{ecs::system::SystemParam, prelude::*};
use bevy_aqua_core::{AuxiliaryWaterView, BedHeightMap, Ocean, ResolvedWaterBodies};
use bevy_aqua_query::{WaveQuery, WaveSurface};
use bevy_hanabi::prelude::*;

use crate::{Budget, Emitter, Probe, SprayQuality, SpraySettings, limits};

pub(super) fn configure_quality(
    mut commands: Commands,
    settings: Res<SpraySettings>,
    mut budget: ResMut<Budget>,
    probes: Query<(Entity, &Probe, Option<&WaveQuery>)>,
    mut emitters: Query<(&mut Visibility, Option<&mut EffectSpawner>), With<Emitter>>,
) {
    if budget.quality == settings.quality {
        return;
    }
    budget.quality = settings.quality;
    budget.tokens = limits(settings.quality).particles_per_second;
    let active = limits(settings.quality).probes;
    for (entity, probe, query) in &probes {
        let enabled = probe.index < active;
        if enabled && query.is_none() {
            commands.entity(entity).insert(WaveQuery);
        } else if !enabled && query.is_some() {
            commands.entity(entity).remove::<WaveQuery>();
        }
    }
    for (mut visibility, spawner) in &mut emitters {
        *visibility = if settings.quality == SprayQuality::Off {
            Visibility::Hidden
        } else {
            Visibility::Inherited
        };
        if let Some(mut spawner) = spawner {
            spawner.active = settings.quality != SprayQuality::Off;
        }
    }
}

type MainCameraQuery<'w, 's> = Query<
    'w,
    's,
    (&'static Camera, &'static GlobalTransform),
    (With<Camera3d>, Without<AuxiliaryWaterView>),
>;

type EmitterQuery<'w, 's> = Query<
    'w,
    's,
    (
        &'static mut Transform,
        &'static mut EffectProperties,
        &'static mut EffectSpawner,
    ),
    With<Emitter>,
>;

type ProbeQuery<'w, 's> = Query<
    'w,
    's,
    (
        Entity,
        &'static GlobalTransform,
        &'static WaveSurface,
        &'static mut Probe,
    ),
    With<WaveQuery>,
>;

#[derive(SystemParam)]
pub(super) struct EmitInputs<'w, 's> {
    time: Res<'w, Time>,
    settings: Res<'w, SpraySettings>,
    budget: ResMut<'w, Budget>,
    cameras: MainCameraQuery<'w, 's>,
    bed: Option<Res<'w, BedHeightMap>>,
    images: Res<'w, Assets<Image>>,
    probes: ProbeQuery<'w, 's>,
    emitters: EmitterQuery<'w, 's>,
}

pub(super) fn place_probes(
    settings: Res<SpraySettings>,
    ocean: Option<Res<Ocean>>,
    cameras: MainCameraQuery,
    bodies: Res<ResolvedWaterBodies>,
    mut probes: Query<(&Probe, &mut Transform), With<WaveQuery>>,
) {
    let Some((camera, camera_transform)) = cameras.iter().find(|(camera, _)| camera.is_active)
    else {
        return;
    };
    let _ = camera;
    let limits = limits(settings.quality);
    let forward = camera_transform.forward().as_vec3();
    let forward_xz = Vec2::new(forward.x, forward.z).normalize_or_zero();
    let right_xz = Vec2::new(-forward_xz.y, forward_xz.x);
    let origin = camera_transform.translation().xz();
    for (probe, mut transform) in &mut probes {
        let column = probe.index % limits.columns;
        let row = probe.index / limits.columns;
        let across = column as f32 - (limits.columns - 1) as f32 * 0.5;
        let ahead = row as f32 + 0.75;
        let xz = origin + right_xz * across * limits.spacing + forward_xz * ahead * limits.spacing;
        let base_y = probe_base_y(xz, ocean.as_deref().map(|ocean| ocean.level), &bodies);
        transform.translation = xz.extend(base_y.unwrap_or(0.0)).xzy();
    }
}

fn probe_base_y(xz: Vec2, ocean_level: Option<f32>, bodies: &ResolvedWaterBodies) -> Option<f32> {
    bodies
        .0
        .iter()
        .find(|body| body.contains(xz))
        .map(|body| body.level)
        .or(ocean_level)
}

fn breaking_surf(depth: Option<f32>, crest: f32, threshold: f32) -> bool {
    // Shallow depth locates surf, but is not itself a breaking event. Require
    // the compression signal so flat shore water cannot emit a blanket.
    depth.is_some_and(|depth| (0.15..=3.2).contains(&depth)) && crest >= threshold * 0.5
}

#[derive(Clone, Copy)]
struct Candidate {
    probe: Entity,
    position: Vec3,
    strength: f32,
    area: f32,
}

pub(super) fn emit_spray(inputs: EmitInputs) {
    let EmitInputs {
        time,
        settings,
        mut budget,
        cameras,
        bed,
        images,
        mut probes,
        mut emitters,
    } = inputs;
    let limits = limits(settings.quality);
    if limits.probes == 0 {
        return;
    }
    let Some((camera, camera_transform)) = cameras.iter().find(|(camera, _)| camera.is_active)
    else {
        return;
    };
    let Some(viewport) = camera.logical_viewport_size() else {
        return;
    };
    budget.tokens = (budget.tokens + time.delta_secs() * limits.particles_per_second)
        .min(limits.particles_per_second);
    let mut candidates = Vec::new();
    for (entity, transform, surface, mut probe) in &mut probes {
        if !surface.valid {
            continue;
        }
        probe.cooldown = (probe.cooldown - time.delta_secs()).max(0.0);
        let base = transform.translation();
        let position = Vec3::new(
            base.x + surface.displacement.x,
            surface.displacement.y + base.y,
            base.z + surface.displacement.z,
        );
        let crest_break = surface.crest >= settings.crest_threshold;
        let depth = bed
            .as_deref()
            .and_then(|map| sample_bed(map, &images, position.xz()))
            .map(|height| position.y - height);
        let surf_break = breaking_surf(depth, surface.crest, settings.crest_threshold);
        if probe.cooldown > 0.0 || (!crest_break && !surf_break) {
            continue;
        }
        let distance = position.distance(camera_transform.translation());
        if distance > limits.distance {
            continue;
        }
        let Ok(center) = camera.world_to_viewport(camera_transform, position) else {
            continue;
        };
        let radius_world = 0.25 + 0.35 * surface.crest;
        let offset_world = position + camera_transform.right().as_vec3() * radius_world;
        let Ok(edge) = camera.world_to_viewport(camera_transform, offset_world) else {
            continue;
        };
        let radius = center.distance(edge);
        if radius * 2.0 < 2.0
            || center.x + radius < 0.0
            || center.y + radius < 0.0
            || center.x - radius > viewport.x
            || center.y - radius > viewport.y
        {
            continue;
        }
        let area = std::f32::consts::PI * radius * radius / (viewport.x * viewport.y);
        let shore_strength = if surf_break {
            (surface.crest / settings.crest_threshold.max(1e-4)).clamp(0.15, 1.0)
        } else {
            0.0
        };
        let strength = surface.crest.max(shore_strength);
        candidates.push(Candidate {
            probe: entity,
            position,
            strength,
            area,
        });
    }
    dispatch_candidates(candidates, limits, &mut budget, &mut probes, &mut emitters);
}

// Admission means one pending CPU burst, not confirmed GPU particle creation.
fn dispatch_candidates(
    mut candidates: Vec<Candidate>,
    limits: crate::Limits,
    budget: &mut Budget,
    probes: &mut ProbeQuery,
    emitters: &mut EmitterQuery,
) {
    candidates.sort_by(|a, b| b.strength.total_cmp(&a.strength));
    let mut coverage = 0.0;
    let mut emitter_list: Vec<_> = emitters.iter_mut().collect();
    if emitter_list.is_empty() {
        return;
    }
    let mut used_emitters = 0;
    for candidate in candidates {
        if used_emitters == emitter_list.len() {
            break;
        }
        if coverage + candidate.area > limits.coverage || budget.tokens < 1.0 {
            continue;
        }
        let wanted = (2.0 + candidate.strength * (limits.burst - 2) as f32).round() as u32;
        let count = wanted.min(limits.burst).min(budget.tokens as u32).max(1);
        let index = budget.emitter_cursor % emitter_list.len();
        // Each emitter stores only one pending settings/transform/reset. Do not
        // wrap onto an emitter already written by this invocation.
        budget.emitter_cursor = (index + 1) % emitter_list.len();
        used_emitters += 1;
        let (transform, properties, spawner) = &mut emitter_list[index];
        transform.translation = candidate.position;
        properties.set("strength", (0.65 + candidate.strength).into());
        spawner.settings = SpawnerSettings::once((count as f32).into()).with_emit_on_start(false);
        spawner.active = true;
        spawner.reset();
        budget.tokens -= count as f32;
        coverage += candidate.area;
        if let Ok((_, _, _, mut probe)) = probes.get_mut(candidate.probe) {
            probe.cooldown = 0.2;
        }
    }
}

fn sample_bed(map: &BedHeightMap, images: &Assets<Image>, xz: Vec2) -> Option<f32> {
    let uv = (xz - map.origin) / map.size;
    if uv.min_element() < 0.0 || uv.max_element() > 1.0 {
        return None;
    }
    let image = images.get(&map.image)?;
    let data = image.data.as_deref()?;
    let size = image.texture_descriptor.size;
    let x = (uv.x * (size.width - 1) as f32).round() as usize;
    let y = (uv.y * (size.height - 1) as f32).round() as usize;
    let value = *data.get(y * size.width as usize + x)? as f32 / 255.0;
    Some(map.height_range[0] + value * (map.height_range[1] - map.height_range[0]))
}

#[cfg(test)]
mod tests {
    use super::*;

    // Exercise the production dispatch path with real ECS query items, without
    // a renderer, projection setup, or a duplicate admission algorithm.
    fn fixture(pool: usize, count: usize) -> (World, Vec<Candidate>) {
        let mut world = World::new();
        for _ in 0..pool {
            world.spawn((
                Emitter,
                Transform::default(),
                EffectProperties::default(),
                EffectSpawner::new(&SpawnerSettings::once(1.0.into()).with_emit_on_start(false)),
            ));
        }
        let candidates = (0..count)
            .map(|index| {
                let probe = world
                    .spawn((
                        Probe {
                            index,
                            cooldown: 0.0,
                        },
                        WaveQuery,
                        WaveSurface::default(),
                        GlobalTransform::default(),
                    ))
                    .id();
                Candidate {
                    probe,
                    position: Vec3::new(index as f32, 0.0, 0.0),
                    strength: 0.1,
                    area: 0.00001,
                }
            })
            .collect();
        (world, candidates)
    }

    fn dispatch(world: &mut World, candidates: Vec<Candidate>, budget: &mut Budget) {
        let mut state = bevy::ecs::system::SystemState::<(ProbeQuery, EmitterQuery)>::new(world);
        let (mut probes, mut emitters) = state.get_mut(world).expect("valid dispatch queries");
        dispatch_candidates(
            candidates,
            limits(SprayQuality::High),
            budget,
            &mut probes,
            &mut emitters,
        );
    }

    #[test]
    fn dispatch_reserves_each_emitter_once_and_charges_only_retained_bursts() {
        let (mut world, candidates) = fixture(24, 40);
        let original = candidates.clone();
        let mut budget = Budget {
            tokens: 160.0,
            emitter_cursor: 23,
            ..default()
        };
        dispatch(&mut world, candidates, &mut budget);
        // crest=0.1 produces round(2 + 0.1*18) = 4 particles.
        assert_eq!(budget.tokens, 64.0);
        assert_eq!(budget.emitter_cursor, 23);
        for (index, candidate) in original.iter().enumerate() {
            assert_eq!(
                world.get::<Probe>(candidate.probe).unwrap().cooldown,
                if index < 24 { 0.2 } else { 0.0 }
            );
        }
        let mut positions = Vec::new();
        let mut pending_count = 0.0;
        for (transform, spawner) in world.query::<(&Transform, &EffectSpawner)>().iter(&world) {
            positions.push(transform.translation.x as usize);
            assert!(!spawner.has_completed());
            pending_count += spawner.settings.count().range()[0];
        }
        positions.sort_unstable();
        assert_eq!(positions, (0..24).collect::<Vec<_>>());
        assert_eq!(pending_count, 160.0 - budget.tokens);
    }

    #[test]
    fn rejection_does_not_charge_cooldown_tokens_or_cursor() {
        for (pool, tokens, area) in [(0, 160.0, 0.00001), (1, 0.5, 0.00001), (1, 160.0, 0.03)] {
            let (mut world, mut candidates) = fixture(pool, 1);
            candidates[0].area = area;
            let probe = candidates[0].probe;
            let mut budget = Budget {
                tokens,
                emitter_cursor: usize::MAX,
                ..default()
            };
            dispatch(&mut world, candidates, &mut budget);
            assert_eq!(world.get::<Probe>(probe).unwrap().cooldown, 0.0);
            assert_eq!(budget.tokens, tokens);
            assert_eq!(budget.emitter_cursor, usize::MAX);
        }
    }

    #[test]
    fn partial_token_burst_charges_actual_count_and_cursor_wraps_without_overflow() {
        let (mut world, candidates) = fixture(2, 2);
        let first = candidates[0].probe;
        let second = candidates[1].probe;
        let mut budget = Budget {
            tokens: 1.5,
            emitter_cursor: usize::MAX,
            ..default()
        };
        dispatch(&mut world, candidates, &mut budget);
        assert_eq!(budget.tokens, 0.5);
        assert_eq!(budget.emitter_cursor, 0);
        assert_eq!(world.get::<Probe>(first).unwrap().cooldown, 0.2);
        assert_eq!(world.get::<Probe>(second).unwrap().cooldown, 0.0);
    }

    #[test]
    fn reservation_is_per_invocation_not_particle_lifetime() {
        let (mut world, candidates) = fixture(1, 2);
        let mut budget = Budget {
            tokens: 160.0,
            ..default()
        };
        dispatch(&mut world, vec![candidates[0]], &mut budget);
        dispatch(&mut world, vec![candidates[1]], &mut budget);
        assert_eq!(budget.tokens, 152.0);
        let transform = world
            .query_filtered::<&Transform, With<Emitter>>()
            .single(&world)
            .unwrap();
        assert_eq!(transform.translation, candidates[1].position);
    }

    #[test]
    fn bounded_probe_base_uses_resolved_surface_level() {
        let shape = bevy_aqua_core::WaterShape::Circle { radius: 4.0 };
        let body = bevy_aqua_core::ResolvedWaterBody::resolve(
            Entity::from_bits(1),
            &shape,
            None,
            &GlobalTransform::from(Transform::from_xyz(3.0, 5.0, -2.0)),
        )
        .unwrap();
        let bodies = ResolvedWaterBodies(vec![body]);
        assert_eq!(probe_base_y(Vec2::new(3.0, -2.0), None, &bodies), Some(5.0));
        assert_eq!(
            probe_base_y(Vec2::new(30.0, -2.0), Some(2.0), &bodies),
            Some(2.0)
        );
    }

    #[test]
    fn shallow_flat_water_is_not_breaking_surf() {
        assert!(!breaking_surf(Some(0.4), 0.0, 0.06));
        assert!(!breaking_surf(Some(2.0), 0.02, 0.06));
        assert!(breaking_surf(Some(2.0), 0.03, 0.06));
        assert!(!breaking_surf(Some(8.0), 0.5, 0.06));
    }
}
