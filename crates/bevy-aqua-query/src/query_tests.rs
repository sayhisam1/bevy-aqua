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

// world_xz(2), slot, kind, flow(4): eight floats per request.
const REQUEST_FLOATS: usize = 8;

const FIRST_SLOT: u32 = 1;

fn registry_with(entities: &[Entity]) -> (Registry, Vec<u32>) {
    let mut registry = Registry::default();
    let slots = entities
        .iter()
        .map(|entity| registry.assign(*entity).unwrap())
        .collect();
    (registry, slots)
}

fn test_entities(count: usize) -> Vec<Entity> {
    let mut world = World::new();
    (0..count).map(|_| world.spawn_empty().id()).collect()
}

#[test]
fn slots_are_unique_and_reclaimed() {
    let entities = test_entities(3);
    let [a, b, c] = entities[..] else {
        panic!("three entities")
    };
    let (mut registry, slots) = registry_with(&[a, b]);
    assert_eq!(slots, vec![FIRST_SLOT, FIRST_SLOT + 1]);
    assert_eq!(registry.entities[&slots[0]], a);

    registry.reclaim(a);
    assert!(!registry.slots.contains_key(&a));
    assert!(!registry.entities.contains_key(&slots[0]));

    // The freed slot is not reused; live probes keep stable identities.
    assert_eq!(registry.assign(b), None);
    assert_eq!(registry.assign(c), Some(3));
}

#[test]
fn request_packing_round_trips_and_caps_capacity() {
    let submissions: Vec<(u32, Vec2, f32, Vec4)> = (0..MAX_QUERIES + 3)
        .map(|index| {
            (
                index + 1,
                Vec2::new(index as f32, -(index as f32)),
                0.0,
                Vec4::ZERO,
            )
        })
        .collect();
    let (bytes, count) = pack_requests(&submissions);
    assert_eq!(count, MAX_QUERIES as usize);
    assert_eq!(
        bytes.len(),
        count * QueryRequest::SHADER_SIZE.get() as usize
    );

    let floats: Vec<f32> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| f32::from_le_bytes(*chunk))
        .collect();
    assert_eq!(floats[0], 0.0);
    assert_eq!(floats[1], 0.0);
    assert_eq!(floats[2], 1.0); // first slot echo
    let stride = REQUEST_FLOATS;
    assert_eq!(floats[stride], 1.0); // second request x
    assert_eq!(floats[stride + 2], 2.0); // second slot echo
}

#[test]
fn result_decode_filters_invalid_and_unknown_rows() {
    let live = test_entities(1)[0];
    let (registry, slots) = registry_with(&[live]);
    let slot = slots[0] as f32;

    let make_record =
        |displacement: [f32; 3], echo: f32, normal: [f32; 3], validity: f32, crest: f32| {
            let record = [
                displacement[0],
                displacement[1],
                displacement[2],
                echo,
                normal[0],
                normal[1],
                normal[2],
                validity,
                crest,
                0.0,
                0.0,
                0.0,
            ];
            record.map(f32::to_le_bytes).concat()
        };
    let mut data = Vec::new();
    data.extend(make_record(
        [1.0, 2.0, 3.0],
        slot,
        [0.0, 1.0, 0.0],
        1.0,
        0.75,
    ));
    data.extend(make_record(
        [9.0, 9.0, 9.0],
        999.0,
        [0.0, 1.0, 0.0],
        1.0,
        0.0,
    ));
    data.extend(make_record(
        [8.0, 8.0, 8.0],
        slot,
        [0.0, 1.0, 0.0],
        0.0,
        0.0,
    ));

    let samples = decode_results(&data, &registry.entities);
    assert_eq!(samples.len(), 1);
    assert_eq!(samples[0].0, live);
    assert_eq!(samples[0].1, Vec3::new(1.0, 2.0, 3.0));
    assert_eq!(samples[0].2, Vec3::new(0.0, 1.0, 0.0));
    assert_eq!(samples[0].3, 0.75);
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
    assert!(shader.contains("if request.kind > 0.5"));
    assert!(shader.contains("alpha = max(alpha, detail_alpha);"));
}
