use super::*;

#[test]
fn every_patch_has_finite_static_bounds() {
    for patch in geometry::Patch::ALL {
        let bounds = geometry::build_patch(patch)
            .compute_aabb()
            .expect("patch bounds");
        assert!(bounds.center.is_finite());
        assert!(bounds.half_extents.is_finite());
        assert!(bounds.half_extents.x > 0.0 && bounds.half_extents.z > 0.0);
    }
}

#[test]
fn expanded_bounds_convert_world_displacement_to_tile_local_space() {
    let base = Aabb::from_min_max(Vec3::splat(-0.5), Vec3::splat(0.5));
    let displacement = waves::DisplacementBounds {
        horizontal: 3.3,
        vertical: 2.1,
    };
    let lod = 2;
    let scale = lod_scale(lod);
    let padding = identity_parent_padding(lod, displacement);
    let expanded = expanded_bounds(base, padding);
    let snap_world = scale * SNAP_AND_MORPH_CELLS / TILE_RESOLUTION as f32;
    assert!(
        ((expanded.half_extents.x - base.half_extents.x) * scale
            - displacement.horizontal
            - snap_world)
            .abs()
            < 1e-5
    );
    assert_eq!(
        expanded.half_extents.y - base.half_extents.y,
        displacement.vertical
    );
    assert_eq!(expanded.half_extents.x, expanded.half_extents.z);
    assert_eq!(expanded.center, base.center);
}

#[test]
fn inverse_affine_padding_supports_transformed_ancestors() {
    let transform = GlobalTransform::from(
        Transform::from_rotation(Quat::from_euler(EulerRot::YXZ, 0.4, -0.2, 0.1))
            .with_scale(Vec3::new(2.0, 0.75, 1.5)),
    );
    let world_padding = Vec3A::new(7.0, 3.0, 5.0);
    let local_padding = world_to_local_padding(&transform, world_padding).unwrap();
    let linear = transform.affine().matrix3;
    let absolute_linear = Mat3A::from_cols(
        linear.x_axis.abs(),
        linear.y_axis.abs(),
        linear.z_axis.abs(),
    );
    let enclosed_world = absolute_linear * local_padding;
    assert!((enclosed_world.cmpge(world_padding)).all());

    let singular = GlobalTransform::from(Transform::from_scale(Vec3::new(1.0, 0.0, 1.0)));
    assert!(world_to_local_padding(&singular, world_padding).is_none());
}

#[test]
fn root_tracks_ocean_and_only_valid_resolved_bodies() {
    let mut app = App::new();
    app.add_plugins(bevy::transform::TransformPlugin)
        .init_resource::<Assets<Mesh>>()
        .insert_resource(OceanWaves::default())
        .insert_resource(waves::StartupAmplitude(1.0))
        .insert_resource(ResolvedWaterBodies::default())
        .insert_resource(lod::Data::new(
            Handle::default(),
            Handle::default(),
            Handle::default(),
            lod::GpuLayout::new(&lod::layout(Vec2::ZERO), Vec2::ZERO, 0.0),
        ))
        .add_systems(Startup, init)
        .add_systems(Update, sync);

    app.update();
    assert_eq!(
        app.world_mut().query::<&Root>().iter(app.world()).count(),
        0
    );

    app.world_mut().insert_resource(Ocean { level: 4.0 });
    app.update();
    let mut roots = app.world_mut().query::<(&Root, &Transform)>();
    let transforms = roots.iter(app.world()).collect::<Vec<_>>();
    assert_eq!(transforms.len(), 1);
    assert_eq!(transforms[0].1.translation.y, 4.0);
    let root = app
        .world_mut()
        .query_filtered::<Entity, With<Root>>()
        .single(app.world())
        .unwrap();
    assert_eq!(
        app.world()
            .get::<GlobalTransform>(root)
            .unwrap()
            .translation()
            .y,
        4.0
    );
    let tile = app
        .world_mut()
        .query_filtered::<Entity, With<Tile>>()
        .iter(app.world())
        .next()
        .unwrap();
    assert_eq!(
        app.world()
            .get::<GlobalTransform>(tile)
            .unwrap()
            .translation()
            .y,
        4.0
    );

    app.world_mut().resource_mut::<Ocean>().level = 6.0;
    app.update();
    assert_eq!(
        app.world()
            .get::<GlobalTransform>(root)
            .unwrap()
            .translation()
            .y,
        6.0
    );
    assert_eq!(
        app.world()
            .get::<GlobalTransform>(tile)
            .unwrap()
            .translation()
            .y,
        6.0
    );

    app.world_mut().remove_resource::<Ocean>();
    app.update();
    assert_eq!(
        app.world_mut().query::<&Root>().iter(app.world()).count(),
        0
    );

    let body = app
        .world_mut()
        .spawn((
            WaterBody,
            WaterShape::Circle { radius: 2.0 },
            Transform::default(),
        ))
        .id();
    app.update();
    assert_eq!(
        app.world_mut().query::<&Root>().iter(app.world()).count(),
        1
    );

    app.world_mut().entity_mut(body).remove::<WaterShape>();
    app.update();
    assert_eq!(
        app.world_mut().query::<&Root>().iter(app.world()).count(),
        0
    );
}
