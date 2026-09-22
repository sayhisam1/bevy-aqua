use super::*;

use bevy_aqua_sdf::{RiverPath, RiverPoint};

#[test]
fn probes_ride_transformed_river_flow_and_body_levels() {
    let pond = bevy_aqua_core::WaterShape::Circle { radius: 19.0 };
    let river = bevy_aqua_core::WaterShape::River {
        path: RiverPath {
            points: vec![
                RiverPoint::new(Vec2::new(-50.0, 0.0), 10.0, 1.0),
                RiverPoint::new(Vec2::new(50.0, 0.0), 10.0, 3.0),
            ],
        },
    };
    let bodies = vec![
        ResolvedWaterBody::resolve(
            Entity::from_bits(1),
            &pond,
            None,
            &GlobalTransform::from(Transform::from_xyz(45.0, 3.0, 30.0)),
        )
        .unwrap(),
        ResolvedWaterBody::resolve(
            Entity::from_bits(2),
            &river,
            None,
            &GlobalTransform::IDENTITY,
        )
        .unwrap(),
    ];
    assert!(matches!(
        probe_resolution(&bodies, Some(2.0), Vec2::new(500.0, 500.0)),
        Some(ProbeResolution::Ocean)
    ));
    assert!(probe_resolution(&bodies, None, Vec2::new(500.0, 500.0)).is_none());
    assert!(matches!(
        probe_resolution(&bodies, None, Vec2::new(45.0, 30.0)),
        Some(ProbeResolution::Body)
    ));
    match probe_resolution(&bodies, None, Vec2::ZERO) {
        Some(ProbeResolution::River { flowed }) => {
            assert!((flowed.flow.x - 2.0).abs() < 1e-4);
            assert!((flowed.margin - 5.0).abs() < 1e-4);
            assert!((flowed.half_width - 5.0).abs() < 1e-4);
        }
        other => panic!("expected river resolution, got {other:?}"),
    }
}

// Protocol fixtures use the real extraction system and readback observer.
// They supply bytes, not a CPU implementation of wave sampling.
fn query_world(count: usize) -> (World, Vec<Entity>, Entity) {
    let mut main = MainWorld::default();
    main.init_resource::<Registry>();
    main.init_resource::<ResolvedWaterBodies>();
    main.insert_resource(Ocean::default());
    let entities = (0..count)
        .map(|index| {
            main.spawn((
                WaveQuery,
                GlobalTransform::from(Transform::from_xyz(index as f32, 0.0, 0.0)),
            ))
            .id()
        })
        .collect();
    let readback = main.spawn_empty().observe(apply_results).id();
    let mut update = Schedule::new(Update);
    update.add_systems((reclaim_slots, assign_slots).chain());
    main.add_schedule(update);
    let mut render = World::new();
    render.insert_resource(main);
    let mut extract = Schedule::new(ExtractSchedule);
    extract.add_systems(extract_wave_queries);
    render.add_schedule(extract);
    (render, entities, readback)
}

fn extract(render: &mut World) -> Vec<QueryRequest> {
    render.resource_mut::<MainWorld>().run_schedule(Update);
    render.run_schedule(ExtractSchedule);
    let batch = render.resource::<Batch>();
    if batch.count == 0 {
        assert!(batch.bytes.is_empty());
        return Vec::new();
    }
    assert_eq!(batch.bytes.len(), batch.count * 32);
    bevy::render::render_resource::encase::StorageBuffer::new(batch.bytes.as_slice())
        .create()
        .unwrap()
}

// Independent storage ABI: 48 bytes; float4 displacement at 0, float4
// normal/crest at 16, uint4 slot/generation/validity/reserved at 32.
fn result_bytes(slot: u32, generation: u32, height: f32, valid: u32) -> Vec<u8> {
    let mut bytes = [0.0_f32, height, 0.0, 0.0, 0.0, 2.0, 0.0, 0.75]
        .map(f32::to_le_bytes)
        .concat();
    bytes.extend([slot, generation, valid, 0].map(u32::to_le_bytes).concat());
    bytes
}

fn deliver(render: &mut World, readback: Entity, data: Vec<u8>) {
    render
        .resource_mut::<MainWorld>()
        .trigger(ReadbackComplete {
            entity: readback,
            data,
        });
}

fn surface(render: &World, entity: Entity) -> WaveSurface {
    *render
        .resource::<MainWorld>()
        .get::<WaveSurface>(entity)
        .unwrap()
}

#[test]
fn integer_request_batch_and_result_abi_round_trip() {
    // Adjacent IDs beyond f32's exact integer range must remain distinct.
    let submissions = [
        (
            0x0100_0001,
            Vec2::new(1.25, -2.5),
            2,
            Vec4::new(3.0, 4.0, 5.0, 6.0),
        ),
        (0x0100_0002, Vec2::ZERO, 1, Vec4::ZERO),
    ];
    let (bytes, count) = pack_requests(&submissions);
    let mut expected = [1.25_f32, -2.5].map(f32::to_le_bytes).concat();
    expected.extend([0x0100_0001_u32, 2].map(u32::to_le_bytes).concat());
    expected.extend([3.0_f32, 4.0, 5.0, 6.0].map(f32::to_le_bytes).concat());
    expected.extend([0.0_f32, 0.0].map(f32::to_le_bytes).concat());
    expected.extend([0x0100_0002_u32, 1].map(u32::to_le_bytes).concat());
    expected.extend([0_u8; 16]);
    assert_eq!(count, 2);
    assert_eq!(bytes, expected);

    let mut uniform = Vec::<u8>::new();
    bevy::render::render_resource::encase::UniformBuffer::new(&mut uniform)
        .write(&QueryBatch {
            count: 65,
            generation: 0xf123_4567,
            reserved: UVec2::ZERO,
        })
        .unwrap();
    assert_eq!(
        uniform,
        [65_u32, 0xf123_4567, 0, 0].map(u32::to_le_bytes).concat()
    );

    let raw = result_bytes(0x0100_0001, 0xf123_4567, -3.5, 1);
    let results: Vec<QueryResult> =
        bevy::render::render_resource::encase::StorageBuffer::new(raw.as_slice())
            .create()
            .unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(
        results[0].metadata,
        UVec4::new(0x0100_0001, 0xf123_4567, 1, 0)
    );
    assert_eq!(results[0].displacement, Vec4::new(0.0, -3.5, 0.0, 0.0));
    assert_eq!(results[0].normal_crest, Vec4::new(0.0, 2.0, 0.0, 0.75));
    let mut round_trip = Vec::<u8>::new();
    bevy::render::render_resource::encase::StorageBuffer::new(&mut round_trip)
        .write(&results)
        .unwrap();
    assert_eq!(round_trip, raw);
}

#[test]
fn shrinking_batches_reject_old_duplicate_tails_and_zero_batch_readbacks() {
    // Includes [A, B] -> [B] -> [] and a workgroup-boundary crossing.
    for count in [2, 65] {
        let (mut render, entities, readback) = query_world(count);
        let requests = extract(&mut render);
        assert_eq!(requests.len(), count);
        let old_generation = render.resource::<Batch>().generation;
        let survivor = *entities.last().unwrap();
        let slot = render.resource::<MainWorld>().resource::<Registry>().slots[&survivor];
        for &entity in &entities[..count - 1] {
            render.resource_mut::<MainWorld>().despawn(entity);
        }
        render
            .resource_mut::<MainWorld>()
            .entity_mut(survivor)
            .insert(GlobalTransform::from(Transform::from_xyz(91.0, 0.0, -17.0)));
        let requests = extract(&mut render);
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].slot, slot);
        assert_eq!(requests[0].world_xz, Vec2::new(91.0, -17.0));
        let generation = render.resource::<Batch>().generation;
        let mut rows = result_bytes(slot, generation, 7.0, 1);
        rows.extend(result_bytes(slot, old_generation, -99.0, 1));
        rows.resize(MAX_QUERIES as usize * 48, 0);
        deliver(&mut render, readback, rows);
        assert_eq!(surface(&render, survivor).displacement.y, 7.0);
        assert_eq!(surface(&render, survivor).normal, Vec3::Y);
        assert_eq!(surface(&render, survivor).crest, 0.75);
        // Separate older readbacks and same-generation duplicates also lose.
        deliver(
            &mut render,
            readback,
            result_bytes(slot, old_generation, -80.0, 1),
        );
        deliver(
            &mut render,
            readback,
            result_bytes(slot, generation, -70.0, 1),
        );
        assert_eq!(surface(&render, survivor).displacement.y, 7.0);

        render
            .resource_mut::<MainWorld>()
            .remove_resource::<Ocean>();
        assert!(extract(&mut render).is_empty());
        assert_eq!(surface(&render, survivor), WaveSurface::default());
        deliver(
            &mut render,
            readback,
            result_bytes(slot, generation, 50.0, 1),
        );
        assert_eq!(surface(&render, survivor), WaveSurface::default());
    }
}

#[test]
fn delayed_results_survive_submission_lag_but_not_exit_and_reentry() {
    let (mut render, entities, readback) = query_world(1);
    let probe = entities[0];
    {
        let mut main = render.resource_mut::<MainWorld>();
        main.remove_resource::<Ocean>();
        let owner = main.spawn_empty().id();
        main.resource_mut::<ResolvedWaterBodies>().0.push(
            ResolvedWaterBody::resolve(
                owner,
                &bevy_aqua_core::WaterShape::Circle { radius: 10.0 },
                None,
                &GlobalTransform::IDENTITY,
            )
            .unwrap(),
        );
    }
    render
        .resource_mut::<MainWorld>()
        .resource_mut::<Registry>()
        .next_slot = 0x0100_0000;
    let slot = extract(&mut render)[0].slot;
    let first = render.resource::<Batch>().generation;
    extract(&mut render);
    let second = render.resource::<Batch>().generation;
    // A result newer than any submission must not consume the applied barrier.
    deliver(
        &mut render,
        readback,
        result_bytes(slot, second + 1, 90.0, 1),
    );
    assert!(!surface(&render, probe).valid);
    deliver(&mut render, readback, result_bytes(slot, first, 2.0, 1));
    assert_eq!(surface(&render, probe).displacement.y, 2.0);
    deliver(&mut render, readback, result_bytes(slot, second, 3.0, 0));
    assert_eq!(surface(&render, probe).displacement.y, 2.0);
    deliver(&mut render, readback, result_bytes(slot, second, 3.0, 1));
    assert_eq!(surface(&render, probe).displacement.y, 3.0);

    render
        .resource_mut::<MainWorld>()
        .entity_mut(probe)
        .insert(GlobalTransform::from(Transform::from_xyz(50.0, 0.0, 0.0)));
    assert!(extract(&mut render).is_empty());
    render
        .resource_mut::<MainWorld>()
        .entity_mut(probe)
        .insert(GlobalTransform::IDENTITY);
    assert_eq!(extract(&mut render).len(), 1);
    let reentry = render.resource::<Batch>().generation;
    deliver(&mut render, readback, result_bytes(slot, second, 99.0, 1));
    assert!(!surface(&render, probe).valid);
    deliver(&mut render, readback, result_bytes(slot, reentry, 4.0, 1));
    assert_eq!(surface(&render, probe).displacement.y, 4.0);
}

#[test]
fn removed_slots_and_invalid_transforms_cannot_accept_delayed_results() {
    let (mut render, entities, readback) = query_world(1);
    let old_entity = entities[0];
    let old_slot = extract(&mut render)[0].slot;
    let old_generation = render.resource::<Batch>().generation;
    render.resource_mut::<MainWorld>().despawn(old_entity);
    let replacement = render
        .resource_mut::<MainWorld>()
        .spawn((WaveQuery, GlobalTransform::IDENTITY))
        .id();
    let slot = extract(&mut render)[0].slot;
    assert_ne!(old_slot, slot);
    deliver(
        &mut render,
        readback,
        result_bytes(old_slot, old_generation, 100.0, 1),
    );
    assert!(!surface(&render, replacement).valid);
    let generation = render.resource::<Batch>().generation;
    deliver(
        &mut render,
        readback,
        result_bytes(slot, generation, 1.0, 1),
    );
    assert!(surface(&render, replacement).valid);
    render
        .resource_mut::<MainWorld>()
        .entity_mut(replacement)
        .insert(GlobalTransform::from(Transform::from_xyz(
            f32::NAN,
            0.0,
            0.0,
        )));
    assert!(extract(&mut render).is_empty());
    deliver(
        &mut render,
        readback,
        result_bytes(slot, generation, 100.0, 1),
    );
    assert!(!surface(&render, replacement).valid);
    render
        .resource_mut::<MainWorld>()
        .entity_mut(replacement)
        .remove::<GlobalTransform>();
    assert!(extract(&mut render).is_empty());
    deliver(
        &mut render,
        readback,
        result_bytes(slot, generation, 100.0, 1),
    );
    assert!(!surface(&render, replacement).valid);
}

#[test]
fn same_frame_query_removal_and_readdition_gets_a_new_slot() {
    let (mut render, entities, readback) = query_world(1);
    let probe = entities[0];
    let old_slot = extract(&mut render)[0].slot;
    let generation = render.resource::<Batch>().generation;
    deliver(
        &mut render,
        readback,
        result_bytes(old_slot, generation, 3.0, 1),
    );
    assert!(surface(&render, probe).valid);
    render
        .resource_mut::<MainWorld>()
        .entity_mut(probe)
        .remove::<WaveQuery>();
    render
        .resource_mut::<MainWorld>()
        .entity_mut(probe)
        .insert(WaveQuery);
    let slot = extract(&mut render)[0].slot;
    assert_ne!(slot, old_slot);
    deliver(
        &mut render,
        readback,
        result_bytes(old_slot, generation, 100.0, 1),
    );
    assert!(!surface(&render, probe).valid);
}

#[test]
fn generation_wrap_keeps_delayed_results_ordered() {
    let (mut render, entities, readback) = query_world(1);
    let probe = entities[0];
    render
        .resource_mut::<MainWorld>()
        .resource_mut::<Registry>()
        .generation = u32::MAX - 1;
    let slot = extract(&mut render)[0].slot;
    extract(&mut render);
    assert_eq!(render.resource::<Batch>().generation, 0);
    deliver(&mut render, readback, result_bytes(slot, u32::MAX, 1.0, 1));
    assert_eq!(surface(&render, probe).displacement.y, 1.0);
    deliver(&mut render, readback, result_bytes(slot, 0, 2.0, 1));
    deliver(&mut render, readback, result_bytes(slot, u32::MAX, 9.0, 1));
    assert_eq!(surface(&render, probe).displacement.y, 2.0);
}

#[test]
fn over_budget_probes_keep_their_previous_sample() {
    let (mut render, entities, _) = query_world(MAX_QUERIES as usize + 1);
    extract(&mut render);
    let previous = WaveSurface {
        displacement: Vec3::new(0.0, 8.0, 0.0),
        valid: true,
        ..default()
    };
    for &entity in &entities {
        render
            .resource_mut::<MainWorld>()
            .entity_mut(entity)
            .insert(previous);
    }
    let requests = extract(&mut render);
    assert_eq!(requests.len(), MAX_QUERIES as usize);
    let main = render.resource::<MainWorld>();
    let registry = main.resource::<Registry>();
    let skipped: Vec<_> = entities
        .iter()
        .filter(|&&entity| {
            let slot = registry.slots[&entity];
            !requests.iter().any(|request| request.slot == slot)
        })
        .collect();
    assert_eq!(skipped.len(), 1);
    assert_eq!(surface(&render, *skipped[0]), previous);
    assert!(!registry.probes[&registry.slots[skipped[0]]].accepting);
}

#[test]
fn unsampled_surface_starts_explicitly_invalid() {
    let surface = WaveSurface::default();
    assert!(!surface.valid);
    assert_eq!(surface.displacement, Vec3::ZERO);
    assert_eq!(surface.normal, Vec3::Y);
}

// Structural guard only; this does not execute the GPU query pipeline.
#[test]
fn ocean_query_ownership_is_world_anchored_and_sampling_remains_advected() {
    let shader = include_str!("wave_query.wgsl");
    assert!(shader.contains("var lod = select_lod(request.world_xz);"));
    assert!(
        shader.contains(
            "var alpha = lod_alpha(request.world_xz, params.cascade_layout.cascades[lod]);"
        )
    );
    assert!(shader.contains("let world_xz = request.world_xz - params.flow.xy * params.time.x;"));
    assert!(shader.contains("let displacement = sample_displacement(world_xz, lod, alpha);"));
    assert!(shader.contains("world_xz + vec2(texel_width, 0.0)"));
    assert!(shader.contains("world_xz + vec2(0.0, texel_width)"));
    assert!(shader.contains("if request.flow.w > 0.0"));
    assert!(shader.contains("river_surface(request.world_xz, request)"));
    assert!(shader.contains("if request.kind != 0u"));
    assert!(shader.contains("alpha = max(alpha, detail_alpha);"));
}
