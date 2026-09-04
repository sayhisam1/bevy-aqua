use bevy::{
    asset::RenderAssetUsages,
    camera::{
        CameraProjection, CameraUpdateSystems, Exposure, Hdr, RenderTarget,
        visibility::RenderLayers,
    },
    core_pipeline::{
        prepass::{DeferredPrepass, DepthPrepass},
        tonemapping::Tonemapping,
    },
    ecs::system::SystemParam,
    image::{ImageFilterMode, ImageSampler, ImageSamplerDescriptor},
    math::reflection_matrix,
    pbr::AtmosphereSettings,
    prelude::*,
    render::render_resource::{Extent3d, TextureDimension, TextureFormat, TextureUsages},
};
use bevy_aqua_core::{
    AquaSettings, AuxiliaryWaterView, CascadeMaterial, CascadeMaterialsUpdated, Data, Ocean,
    OceanView, PlanarReflectionView, ReflectionMode, ResolvedWaterBodies,
};

use crate::ReflectedInWater;

// Reserved for mirror cameras; existing host layers remain attached.
const REFLECTION_LAYER: usize = 31;
// Bounds mirror render cost to the nearest two distinct water levels.
const VIEW_LIMIT: usize = 2;
const MIN_TARGET_SCALE: f32 = 0.1;
const MAX_TARGET_SCALE: f32 = 1.0;
const LEVEL_EPSILON_METRES: f32 = 0.01;

#[derive(Component, Debug, Clone, Copy)]
struct MirrorCamera;

#[derive(Debug)]
struct MirrorSlot {
    entity: Entity,
    image: Handle<Image>,
    raw: Handle<Image>,
}

#[derive(Resource, Debug, Default)]
struct Mirrors {
    slots: Vec<MirrorSlot>,
    size: UVec2,
}

#[derive(SystemParam)]
#[expect(clippy::type_complexity, reason = "mirror-camera query bundle")]
struct Scene<'w, 's> {
    main_camera: Query<
        'w,
        's,
        (
            &'static Camera,
            &'static Projection,
            &'static GlobalTransform,
            Option<&'static Exposure>,
            Option<&'static AtmosphereSettings>,
            Option<&'static EnvironmentMapLight>,
        ),
        (With<OceanView>, Without<AuxiliaryWaterView>),
    >,
    ocean: Option<Res<'w, Ocean>>,
    bodies: Res<'w, ResolvedWaterBodies>,
    mirror_cameras: Query<
        'w,
        's,
        (
            &'static mut Camera,
            &'static mut Transform,
            &'static mut GlobalTransform,
            &'static mut Projection,
            &'static mut Exposure,
        ),
        (With<MirrorCamera>, With<AuxiliaryWaterView>),
    >,
    mirrors: ResMut<'w, Mirrors>,
}

pub(super) fn add(app: &mut App) {
    app.init_resource::<Mirrors>().add_systems(
        PostUpdate,
        (
            include_marked,
            (reset_mirror_activity, sync_mirrors)
                .chain()
                .after(CascadeMaterialsUpdated)
                .after(TransformSystems::Propagate)
                .after(bevy_aqua_core::WaterBodiesResolved)
                .before(CameraUpdateSystems),
        ),
    );
}

#[derive(Default)]
struct ReflectionLayerOwnership {
    owned: std::collections::HashSet<Entity>,
    wanted: std::collections::HashSet<Entity>,
    stack: Vec<Entity>,
}

fn include_marked(
    mut commands: Commands,
    roots: Query<Entity, With<ReflectedInWater>>,
    lights: Query<Entity, With<DirectionalLight>>,
    hierarchy: Query<(Option<&RenderLayers>, Option<&Children>)>,
    mut state: Local<ReflectionLayerOwnership>,
) {
    let ReflectionLayerOwnership {
        owned,
        wanted,
        stack,
    } = &mut *state;
    wanted.clear();
    stack.clear();
    stack.extend(roots.iter());
    while let Some(entity) = stack.pop() {
        if !wanted.insert(entity) {
            continue;
        }
        if let Ok((_, Some(children))) = hierarchy.get(entity) {
            stack.extend(children.iter());
        }
    }
    // Lights participate directly, not by claiming their whole hierarchy.
    wanted.extend(lights.iter());
    let reflection = RenderLayers::layer(REFLECTION_LAYER);
    owned.retain(|entity| {
        if wanted.contains(entity) {
            return true;
        }
        if let Ok((Some(layers), _)) = hierarchy.get(*entity) {
            if layers.intersects(&reflection) {
                commands
                    .entity(*entity)
                    .insert(layers.clone().without(REFLECTION_LAYER));
            }
        }
        false // Includes despawned entities: no stale ownership remains.
    });
    for &entity in wanted.iter() {
        let Ok((layers, _)) = hierarchy.get(entity) else {
            continue;
        };
        if !layers.is_some_and(|layers| layers.intersects(&reflection)) {
            commands
                .entity(entity)
                .insert(layers.cloned().unwrap_or_default().with(REFLECTION_LAYER));
            owned.insert(entity);
        }
        // Pre-existing host layer31 is not ours to revoke. Hosts that want
        // permanent ownership must set that bit before opting into Aqua.
    }
}

// Run before synchronization, including when its material/view prerequisites
// are missing. Only fully validated synchronization can reactivate a mirror.
fn reset_mirror_activity(mut cameras: Query<&mut Camera, With<MirrorCamera>>) {
    for mut camera in &mut cameras {
        camera.is_active = false;
    }
}

fn sync_mirrors(
    mut commands: Commands,
    settings: Res<AquaSettings>,
    data: Res<Data>,
    mut images: ResMut<Assets<Image>>,
    mut materials: ResMut<Assets<CascadeMaterial>>,
    mut scene: Scene,
) {
    let Some(mut material) = materials.get_mut(&data.material()) else {
        return;
    };
    material.reflections.view_count = 0;
    let ReflectionMode::Planar { scale, distortion } = settings.reflections else {
        return;
    };
    let Ok((camera, projection, camera_transform, exposure, atmosphere, environment)) =
        scene.main_camera.single()
    else {
        material.reflections.view_count = 0;
        return;
    };
    let Projection::Perspective(projection) = projection else {
        material.reflections.view_count = 0;
        return;
    };
    if !camera.is_active
        || !scale.is_finite()
        || !distortion.is_finite()
        || !camera_transform.to_matrix().inverse().is_finite()
        || !projection.get_clip_from_view().is_finite()
    {
        return;
    }
    let Some(main_size) = camera
        .physical_viewport_size()
        .filter(|s| s.x > 0 && s.y > 0)
    else {
        return;
    };
    let target_size = (main_size.as_vec2() * scale.clamp(MIN_TARGET_SCALE, MAX_TARGET_SCALE))
        .round()
        .as_uvec2()
        .max(UVec2::ONE);
    let cameras_intact = scene
        .mirrors
        .slots
        .iter()
        .all(|slot| scene.mirror_cameras.contains(slot.entity));
    let rebuilt = ensure_slots(
        &mut commands,
        &mut images,
        &mut scene.mirrors,
        target_size,
        cameras_intact,
    );
    material.reflection_a = scene.mirrors.slots[0].image.clone();
    material.reflection_b = scene.mirrors.slots[1].image.clone();
    if rebuilt {
        material.reflections.view_count = 0;
        return;
    }

    let main_transform = camera_transform.compute_transform();
    let levels = visible_levels(&scene, camera_transform.translation());
    let count = levels.len();
    let mirror_order = camera.order.saturating_sub(1);
    for (index, slot) in scene.mirrors.slots.iter().enumerate() {
        let active = index < count;
        let level = levels.get(index).copied().unwrap_or_default();
        let (transform, mut mirror_projection) = mirror_view(&main_transform, projection, level);
        // Match the render target after integer scaling; resolution never changes UV coverage.
        mirror_projection.aspect_ratio = target_size.x as f32 / target_size.y as f32;
        let view_projection =
            mirror_projection.get_clip_from_view() * transform.to_matrix().inverse();
        material.reflections.views[index] = PlanarReflectionView {
            view_projection,
            level,
        };
        if let Ok((
            mut camera,
            mut camera_transform,
            mut camera_global_transform,
            mut camera_projection,
            mut mirror_exposure,
        )) = scene.mirror_cameras.get_mut(slot.entity)
        {
            camera.order = mirror_order;
            camera.is_active = active;
            *camera_transform = transform;
            // Mirror cameras are controlled root entities. Synchronization runs after
            // propagation so it can read this frame's main camera world transform;
            // publish the matching mirror world transform for same-frame extraction.
            *camera_global_transform = GlobalTransform::from(transform);
            *camera_projection = Projection::Perspective(mirror_projection);
            *mirror_exposure = exposure.cloned().unwrap_or_default();
        }
        // The main environment lights reflected geometry too. Coverage export
        // still rejects the mirror sky; this does not double-light the water.
        if let Some(environment) = environment {
            commands.entity(slot.entity).insert(environment.clone());
        } else {
            commands.entity(slot.entity).remove::<EnvironmentMapLight>();
        }
        if let Some(atmosphere) = atmosphere {
            commands.entity(slot.entity).insert(atmosphere.clone());
        } else {
            commands.entity(slot.entity).remove::<AtmosphereSettings>();
        }
    }
    material.reflections.view_count = count as u32;
    material.reflections.distortion = distortion.max(0.0);
}

fn ensure_slots(
    commands: &mut Commands,
    images: &mut Assets<Image>,
    mirrors: &mut Mirrors,
    size: UVec2,
    cameras_intact: bool,
) -> bool {
    if cameras_intact
        && mirrors.slots.len() == VIEW_LIMIT
        && mirrors
            .slots
            .iter()
            .all(|slot| images.get(&slot.image).is_some() && images.get(&slot.raw).is_some())
    {
        if mirrors.size != size {
            let extent = Extent3d {
                width: size.x,
                height: size.y,
                depth_or_array_layers: 1,
            };
            for slot in &mirrors.slots {
                images
                    .get_mut(&slot.raw)
                    .expect("raw mirror image")
                    .resize(extent);
                // Resize must regenerate the mip count, not only the extent.
                *images
                    .get_mut(&slot.image)
                    .expect("mirror image existence checked above") = reflection_mip_image(size);
            }
            mirrors.size = size;
        }
        return false;
    }
    for slot in mirrors.slots.drain(..) {
        // Hosts may already have despawned a slot; rebuild without queuing
        // failing commands against its old generation.
        commands.entity(slot.entity).try_despawn();
        images.remove(slot.image.id());
        images.remove(slot.raw.id());
    }
    mirrors.size = size;
    for _ in 0..VIEW_LIMIT {
        let image = images.add(reflection_mip_image(size));
        let raw = images.add(reflection_image(size));
        let entity = commands
            .spawn((
                Camera3d::default(),
                Camera {
                    order: -1,
                    is_active: false,
                    invert_culling: true,
                    clear_color: ClearColorConfig::Custom(Color::NONE),
                    ..default()
                },
                RenderTarget::Image(raw.clone().into()),
                crate::resolve::ReflectionOutput(image.clone()),
                Hdr,
                DepthPrepass,
                DeferredPrepass,
                Msaa::Off,
                Tonemapping::None,
                RenderLayers::layer(REFLECTION_LAYER),
                AuxiliaryWaterView,
                MirrorCamera,
            ))
            .id();
        mirrors.slots.push(MirrorSlot { entity, image, raw });
    }
    true
}

fn visible_levels(scene: &Scene, camera_position: Vec3) -> Vec<f32> {
    let camera_xz = camera_position.xz();
    let mut candidates = Vec::new();
    if let Some(ocean) = &scene.ocean {
        if usable_level(camera_position.y, ocean.level) {
            candidates.push((0.0, ocean.level));
        }
    }
    for body in &scene.bodies.0 {
        let (center, _) = body.extent();
        if usable_level(camera_position.y, body.level) {
            candidates.push((center.distance_squared(camera_xz), body.level));
        }
    }
    candidates.sort_by(|left, right| left.0.total_cmp(&right.0));
    let mut levels: Vec<f32> = Vec::with_capacity(VIEW_LIMIT);
    for (_, level) in candidates {
        if levels
            .iter()
            .all(|other| (other - level).abs() > LEVEL_EPSILON_METRES)
        {
            levels.push(level);
            if levels.len() == VIEW_LIMIT {
                break;
            }
        }
    }
    levels
}

// These cameras render above-surface reflection, not underwater optics.
fn usable_level(camera_y: f32, level: f32) -> bool {
    camera_y.is_finite() && level.is_finite() && camera_y - level > LEVEL_EPSILON_METRES
}

fn mirror_view(
    main: &Transform,
    projection: &PerspectiveProjection,
    level: f32,
) -> (Transform, PerspectiveProjection) {
    let plane = Vec3::Y * level;
    let reflection = Mat4::from_translation(plane)
        * Mat4::from_mat3a(reflection_matrix(Vec3::Y))
        * Mat4::from_translation(-plane);
    let transform = Transform::from_matrix(reflection * main.to_matrix());
    let distance = level - main.translation.y;
    let view_from_world = main.compute_affine().matrix3.inverse();
    let normal = (view_from_world * Vec3::NEG_Y).normalize();
    let projection = PerspectiveProjection {
        near_clip_plane: normal.extend(distance),
        ..projection.clone()
    };
    (transform, projection)
}

fn reflection_image(size: UVec2) -> Image {
    let mut image = Image::new_uninit(
        Extent3d {
            width: size.x,
            height: size.y,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        TextureFormat::Rgba16Float,
        RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
    );
    image.texture_descriptor.usage |= TextureUsages::RENDER_ATTACHMENT;
    image.sampler = ImageSampler::Descriptor(ImageSamplerDescriptor {
        mag_filter: ImageFilterMode::Linear,
        min_filter: ImageFilterMode::Linear,
        ..default()
    });
    image
}

// The camera renders only to the single-level raw image. Water samples this
// separate full-chain image; compute and clear use explicit one-level views.
fn reflection_mip_image(size: UVec2) -> Image {
    let mut image = reflection_image(size);
    image.texture_descriptor.mip_level_count = size.x.max(size.y).max(1).ilog2() + 1;
    image.texture_descriptor.usage |= TextureUsages::STORAGE_BINDING;
    image.texture_view_descriptor = None; // Default sampled view spans ALL levels.
    image.sampler = ImageSampler::Descriptor(ImageSamplerDescriptor {
        mag_filter: ImageFilterMode::Linear,
        min_filter: ImageFilterMode::Linear,
        mipmap_filter: ImageFilterMode::Linear,
        ..default()
    });
    image
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reflection_mip_descriptors_cover_odd_skinny_and_resized_targets() {
        for (size, levels) in [
            (UVec2::ONE, 1),
            (UVec2::new(7, 5), 3),
            (UVec2::new(1, 7), 3),
            (UVec2::new(7, 1), 3),
            (UVec2::new(1023, 687), 10),
            (UVec2::new(1025, 687), 11),
        ] {
            let image = reflection_mip_image(size);
            assert_eq!(image.texture_descriptor.mip_level_count, levels);
            assert_eq!(image.texture_descriptor.size.width, size.x);
            assert_eq!(image.texture_descriptor.size.height, size.y);
            assert!(image.texture_view_descriptor.is_none());
            assert!(image.data.is_none());
            assert!(
                image
                    .texture_descriptor
                    .usage
                    .contains(TextureUsages::STORAGE_BINDING)
            );
            assert_eq!(reflection_image(size).texture_descriptor.mip_level_count, 1);
            let ImageSampler::Descriptor(sampler) = image.sampler else {
                panic!("explicit sampler");
            };
            assert_eq!(sampler.mipmap_filter, ImageFilterMode::Linear);
        }
    }

    #[test]
    fn planar_slope_projection_matches_uv_differential_across_camera_bases() {
        let projection = PerspectiveProjection::default();
        for yaw in [0.0_f32, 90.0, 180.0, 270.0] {
            for pitch in [-30.0_f32, -5.0, 5.0] {
                for roll in [-20.0_f32, 0.0, 20.0] {
                    let main = Transform::from_xyz(0.0, 8.0, 0.0).with_rotation(Quat::from_euler(
                        EulerRot::YXZ,
                        yaw.to_radians(),
                        pitch.to_radians(),
                        roll.to_radians(),
                    ));
                    let (mirror, p) = mirror_view(&main, &projection, 0.0);
                    let matrix = p.get_clip_from_view() * mirror.to_matrix().inverse();
                    let point = mirror
                        .to_matrix()
                        .transform_point3(Vec3::new(1.0, -0.5, -25.0));
                    let slope = Vec3::new(0.17, 0.0, -0.23);
                    let clip = matrix * point.extend(1.0);
                    let ndc = clip.truncate() / clip.w;
                    let delta = matrix * slope.extend(0.0);
                    let expected =
                        0.5 * Vec2::new(delta.x - ndc.x * delta.w, -delta.y + ndc.y * delta.w);
                    let uv = |p: Vec3| {
                        let c = matrix * p.extend(1.0);
                        c.xy() / c.w * Vec2::new(0.5, -0.5) + Vec2::splat(0.5)
                    };
                    let epsilon = 0.002;
                    let finite = (uv(point + slope * clip.w * epsilon)
                        - uv(point - slope * clip.w * epsilon))
                        / (2.0 * epsilon);
                    assert!(
                        expected.distance(finite) < 0.0001,
                        "yaw {yaw} pitch {pitch} roll {roll}: {expected:?} vs {finite:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn reset_disables_owned_cameras_without_touching_main_views() {
        let mut app = App::new();
        app.add_systems(Update, reset_mirror_activity);
        let mirror = app
            .world_mut()
            .spawn((Camera::default(), MirrorCamera))
            .id();
        let main = app.world_mut().spawn(Camera::default()).id();
        for _ in 0..2 {
            app.world_mut().get_mut::<Camera>(mirror).unwrap().is_active = true;
            app.update();
            assert!(!app.world().get::<Camera>(mirror).unwrap().is_active);
            assert!(app.world().get::<Camera>(main).unwrap().is_active);
        }
    }

    #[test]
    fn mirror_level_requires_eye_above_mean_surface() {
        assert!(usable_level(10.1, 10.0));
        assert!(!usable_level(10.0, 10.0));
        assert!(!usable_level(9.0, 10.0));
        assert!(!usable_level(10.005, 10.0));
        assert!(!usable_level(11.0, f32::NAN));
        assert!(!usable_level(f32::INFINITY, 10.0));
    }

    #[test]
    fn missing_mirror_entity_rebuilds_slots_and_releases_old_images() {
        fn maintain(
            mut commands: Commands,
            mut images: ResMut<Assets<Image>>,
            mut mirrors: ResMut<Mirrors>,
            cameras: Query<Entity, With<MirrorCamera>>,
        ) {
            let intact = mirrors
                .slots
                .iter()
                .all(|slot| cameras.contains(slot.entity));
            ensure_slots(
                &mut commands,
                &mut images,
                &mut mirrors,
                UVec2::new(32, 17),
                intact,
            );
        }
        let mut app = App::new();
        app.init_resource::<Mirrors>()
            .init_resource::<Assets<Image>>()
            .add_systems(Update, maintain);
        app.update();
        let old: Vec<_> = app
            .world()
            .resource::<Mirrors>()
            .slots
            .iter()
            .map(|s| (s.entity, s.image.id(), s.raw.id()))
            .collect();
        app.world_mut().entity_mut(old[0].0).despawn();
        app.update();
        let new: Vec<_> = app
            .world()
            .resource::<Mirrors>()
            .slots
            .iter()
            .map(|s| s.entity)
            .collect();
        assert_eq!(new.len(), VIEW_LIMIT);
        for (entity, image, raw) in old {
            assert!(app.world().get_entity(entity).is_err());
            assert!(app.world().resource::<Assets<Image>>().get(image).is_none());
            assert!(app.world().resource::<Assets<Image>>().get(raw).is_none());
        }
        for &entity in &new {
            assert!(app.world().get::<MirrorCamera>(entity).is_some());
        }
        app.update();
        assert_eq!(
            new,
            app.world()
                .resource::<Mirrors>()
                .slots
                .iter()
                .map(|s| s.entity)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn mirror_projection_preserves_surface_uv_across_pitch() {
        let projection = PerspectiveProjection {
            aspect_ratio: 16.0 / 9.0,
            ..default()
        };
        let level = 60.0;
        for pitch_degrees in [-25.0_f32, -15.0, -5.0, 5.0] {
            let main = Transform::from_xyz(522.0, 70.0, 205.0).with_rotation(Quat::from_euler(
                EulerRot::YXZ,
                270.0_f32.to_radians(),
                pitch_degrees.to_radians(),
                0.0,
            ));
            let (mirror, mirror_projection) = mirror_view(&main, &projection, level);
            let main_vp = projection.get_clip_from_view() * main.to_matrix().inverse();
            let mirror_vp = mirror_projection.get_clip_from_view() * mirror.to_matrix().inverse();
            for offset in [Vec2::ZERO, Vec2::new(80.0, 40.0), Vec2::new(-60.0, 120.0)] {
                let point = Vec4::new(522.0 + offset.x, level, 205.0 + offset.y, 1.0);
                let main_clip = main_vp * point;
                let mirror_clip = mirror_vp * point;
                let main_ndc = main_clip.xy() / main_clip.w;
                let mirror_ndc = mirror_clip.xy() / mirror_clip.w;
                let tolerance = 2e-4 * (1.0 + main_ndc.length());
                assert!(
                    main_ndc.distance(mirror_ndc) <= tolerance,
                    "pitch {pitch_degrees}: {main_ndc:?} != {mirror_ndc:?}"
                );
            }
        }
    }

    #[test]
    fn removed_markers_and_reparented_children_release_only_owned_bits() {
        let mut app = App::new();
        app.add_systems(Update, include_marked);
        let root = app
            .world_mut()
            .spawn((ReflectedInWater, RenderLayers::layer(7)))
            .id();
        let child = app
            .world_mut()
            .spawn((ChildOf(root), RenderLayers::layer(8)))
            .id();
        let host = app
            .world_mut()
            .spawn((ChildOf(root), RenderLayers::layer(31).with(9)))
            .id();
        app.update();
        app.world_mut().entity_mut(child).remove::<ChildOf>();
        app.update();
        let layers = app.world().get::<RenderLayers>(child).unwrap();
        assert_eq!(*layers, RenderLayers::layer(8));
        app.world_mut()
            .entity_mut(root)
            .remove::<ReflectedInWater>();
        app.update();
        assert_eq!(
            *app.world().get::<RenderLayers>(root).unwrap(),
            RenderLayers::layer(7)
        );
        assert_eq!(
            *app.world().get::<RenderLayers>(host).unwrap(),
            RenderLayers::layer(31).with(9)
        );
    }

    #[test]
    fn nested_markers_keep_membership_and_removed_lights_release_it() {
        let mut app = App::new();
        app.add_systems(Update, include_marked);
        let root = app.world_mut().spawn(ReflectedInWater).id();
        let child = app
            .world_mut()
            .spawn((ChildOf(root), ReflectedInWater))
            .id();
        let light = app
            .world_mut()
            .spawn((DirectionalLight::default(), RenderLayers::layer(4)))
            .id();
        app.update();
        app.world_mut()
            .entity_mut(root)
            .remove::<ReflectedInWater>();
        app.world_mut()
            .entity_mut(light)
            .remove::<DirectionalLight>();
        app.update();
        assert!(
            app.world()
                .get::<RenderLayers>(child)
                .unwrap()
                .intersects(&RenderLayers::layer(31))
        );
        assert_eq!(
            *app.world().get::<RenderLayers>(light).unwrap(),
            RenderLayers::layer(4)
        );
        app.world_mut().entity_mut(root).despawn();
        app.update(); // Despawned owned entries must not issue stale commands.
    }

    #[test]
    fn marked_subtrees_adopt_existing_and_late_descendants() {
        let mut app = App::new();
        app.add_systems(Update, include_marked);
        let root = app
            .world_mut()
            .spawn((ReflectedInWater, RenderLayers::layer(7)))
            .id();
        let child = app
            .world_mut()
            .spawn((ChildOf(root), RenderLayers::default()))
            .id();

        app.update();
        let reflection = RenderLayers::layer(REFLECTION_LAYER);
        let root_layers = app.world().get::<RenderLayers>(root).unwrap();
        let child_layers = app.world().get::<RenderLayers>(child).unwrap();
        assert!(root_layers.intersects(&RenderLayers::layer(7)));
        assert!(root_layers.intersects(&reflection));
        assert!(child_layers.intersects(&RenderLayers::default()));
        assert!(child_layers.intersects(&reflection));

        let late = app.world_mut().spawn(ChildOf(child)).id();
        app.update();
        assert!(
            app.world()
                .get::<RenderLayers>(late)
                .unwrap()
                .intersects(&reflection)
        );
    }
}
