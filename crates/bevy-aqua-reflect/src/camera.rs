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
    app.init_resource::<Mirrors>()
        .add_systems(Update, include_directional_lights)
        .add_systems(
            PostUpdate,
            (
                include_marked,
                remove_mirror_atmosphere.before(sync_mirrors),
                sync_mirror_environment
                    .after(sync_mirrors)
                    .before(CameraUpdateSystems),
                sync_mirrors
                    .after(CascadeMaterialsUpdated)
                    .after(TransformSystems::Propagate)
                    .after(bevy_aqua_core::WaterBodiesResolved)
                    .before(CameraUpdateSystems),
            ),
        );
}

fn include_marked(
    mut commands: Commands,
    roots: Query<Entity, With<ReflectedInWater>>,
    hierarchy: Query<(Option<&RenderLayers>, Option<&Children>)>,
) {
    let reflection = RenderLayers::layer(REFLECTION_LAYER);
    let mut stack = Vec::new();
    for root in &roots {
        stack.push(root);
        while let Some(entity) = stack.pop() {
            let Ok((layers, children)) = hierarchy.get(entity) else {
                continue;
            };
            if !layers.is_some_and(|layers| layers.intersects(&reflection)) {
                commands
                    .entity(entity)
                    .insert(layers.cloned().unwrap_or_default().with(REFLECTION_LAYER));
            }
            if let Some(children) = children {
                stack.extend(children.iter());
            }
        }
    }
}

fn include_directional_lights(
    mut commands: Commands,
    lights: Query<(Entity, Option<&RenderLayers>), Added<DirectionalLight>>,
) {
    for (entity, layers) in &lights {
        commands
            .entity(entity)
            .insert(layers.cloned().unwrap_or_default().with(REFLECTION_LAYER));
    }
}

// Keep this independent of synchronization's material/view prerequisites.
// Mirror targets contain geometry only. Atmosphere generation from a reflected,
// potentially below-ground eye can overwrite the main view's environment map.
fn remove_mirror_atmosphere(
    mut commands: Commands,
    mirrors: Query<Entity, (With<MirrorCamera>, With<AtmosphereSettings>)>,
) {
    for entity in &mirrors {
        commands.entity(entity).remove::<AtmosphereSettings>();
    }
}

// Inherit lighting, not the atmosphere generator or sky renderer. Running after
// synchronization also initializes newly spawned mirrors in the same frame.
fn sync_mirror_environment(
    mut commands: Commands,
    main: Query<
        Option<&EnvironmentMapLight>,
        (With<Camera>, With<OceanView>, Without<AuxiliaryWaterView>),
    >,
    mirrors: Query<Entity, (With<MirrorCamera>, With<AuxiliaryWaterView>)>,
) {
    let environment = main.single().ok().flatten();
    for entity in &mirrors {
        if let Some(environment) = environment {
            commands.entity(entity).insert(environment.clone());
        } else {
            commands.entity(entity).remove::<EnvironmentMapLight>();
        }
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
    let ReflectionMode::Planar { scale, distortion } = settings.reflections else {
        material.reflections.view_count = 0;
        for slot in &scene.mirrors.slots {
            if let Ok((mut camera, ..)) = scene.mirror_cameras.get_mut(slot.entity) {
                camera.is_active = false;
            }
        }
        return;
    };
    let Ok((camera, projection, camera_transform, exposure)) = scene.main_camera.single() else {
        material.reflections.view_count = 0;
        return;
    };
    let Projection::Perspective(projection) = projection else {
        material.reflections.view_count = 0;
        return;
    };
    let Some(main_size) = camera.physical_viewport_size() else {
        return;
    };
    let target_size = (main_size.as_vec2() * scale.clamp(MIN_TARGET_SCALE, MAX_TARGET_SCALE))
        .round()
        .as_uvec2()
        .max(UVec2::ONE);
    let rebuilt = ensure_slots(&mut commands, &mut images, &mut scene.mirrors, target_size);
    material.reflection_a = scene.mirrors.slots[0].image.clone();
    material.reflection_b = scene.mirrors.slots[1].image.clone();
    if rebuilt {
        material.reflections.view_count = 0;
        return;
    }

    let main_transform = camera_transform.compute_transform();
    let levels = visible_levels(&scene, camera_transform.translation().xz());
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
    }
    material.reflections.view_count = count as u32;
    material.reflections.distortion = distortion.max(0.0);
}

fn ensure_slots(
    commands: &mut Commands,
    images: &mut Assets<Image>,
    mirrors: &mut Mirrors,
    size: UVec2,
) -> bool {
    if mirrors.slots.len() == VIEW_LIMIT
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
                    .expect("raw mirror image existence checked above")
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
        commands.entity(slot.entity).despawn();
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

fn visible_levels(scene: &Scene, camera_xz: Vec2) -> Vec<f32> {
    let mut candidates = Vec::new();
    if let Some(ocean) = &scene.ocean {
        candidates.push((0.0, ocean.level));
    }
    for body in &scene.bodies.0 {
        let (center, _) = body.extent();
        candidates.push((center.distance_squared(camera_xz), body.level));
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
    fn mirrors_drop_stale_atmosphere_without_changing_main_view_or_activity() {
        let mut app = App::new();
        // No material, main-view projection, or target resources: cleanup must
        // not depend on any prerequisite that can make sync_mirrors return.
        app.add_systems(PostUpdate, remove_mirror_atmosphere);
        let main = app
            .world_mut()
            .spawn((Camera::default(), OceanView, AtmosphereSettings::default()))
            .id();
        let mirrors: Vec<_> = [true, false]
            .into_iter()
            .map(|is_active| {
                app.world_mut()
                    .spawn((
                        Camera {
                            is_active,
                            ..default()
                        },
                        MirrorCamera,
                        AuxiliaryWaterView,
                    ))
                    .id()
            })
            .collect();
        for _ in 0..2 {
            for &mirror in &mirrors {
                app.world_mut()
                    .entity_mut(mirror)
                    .insert(AtmosphereSettings::default());
            }
            app.update();
            assert!(app.world().get::<AtmosphereSettings>(main).is_some());
            for (&mirror, expected_active) in mirrors.iter().zip([true, false]) {
                assert!(app.world().get::<AtmosphereSettings>(mirror).is_none());
                assert_eq!(
                    app.world().get::<Camera>(mirror).unwrap().is_active,
                    expected_active
                );
            }
        }
    }

    #[test]
    fn mirrors_inherit_replace_and_remove_main_environment_without_atmosphere() {
        let mut app = App::new();
        app.add_systems(
            PostUpdate,
            (remove_mirror_atmosphere, sync_mirror_environment).chain(),
        );
        let main = app
            .world_mut()
            .spawn((Camera::default(), OceanView, AtmosphereSettings::default()))
            .id();
        let mirror = app
            .world_mut()
            .spawn((Camera::default(), MirrorCamera, AuxiliaryWaterView))
            .id();
        let mut images = Assets::<Image>::default();
        for intensity in [250.0, 750.0] {
            let environment = EnvironmentMapLight {
                diffuse_map: images.add(reflection_image(UVec2::ONE)),
                specular_map: images.add(reflection_image(UVec2::ONE)),
                intensity,
                rotation: Quat::from_rotation_y(intensity / 1000.0),
                affects_lightmapped_mesh_diffuse: intensity < 500.0,
            };
            app.world_mut().entity_mut(main).insert(environment.clone());
            app.world_mut()
                .entity_mut(mirror)
                .insert(AtmosphereSettings::default());
            app.update();
            for entity in [main, mirror] {
                let actual = app.world().get::<EnvironmentMapLight>(entity).unwrap();
                assert_eq!(actual.diffuse_map, environment.diffuse_map);
                assert_eq!(actual.specular_map, environment.specular_map);
                assert_eq!(actual.intensity, environment.intensity);
                assert_eq!(actual.rotation, environment.rotation);
                assert_eq!(
                    actual.affects_lightmapped_mesh_diffuse,
                    environment.affects_lightmapped_mesh_diffuse
                );
            }
            assert!(app.world().get::<AtmosphereSettings>(main).is_some());
            assert!(app.world().get::<AtmosphereSettings>(mirror).is_none());
        }
        app.world_mut()
            .entity_mut(main)
            .remove::<EnvironmentMapLight>();
        app.update();
        assert!(app.world().get::<EnvironmentMapLight>(mirror).is_none());
        assert!(app.world().get::<EnvironmentMapLight>(main).is_none());
        assert!(app.world().get::<AtmosphereSettings>(main).is_some());
        assert!(app.world().get::<AtmosphereSettings>(mirror).is_none());
    }

    #[test]
    fn mirror_slots_keep_raw_targets_separate_from_coverage_outputs() {
        use bevy::ecs::system::RunSystemOnce;

        let mut world = World::new();
        world.init_resource::<Assets<Image>>();
        world.init_resource::<Mirrors>();
        for size in [UVec2::new(64, 32), UVec2::new(128, 64)] {
            world
                .run_system_once(
                    move |mut commands: Commands,
                          mut images: ResMut<Assets<Image>>,
                          mut mirrors: ResMut<Mirrors>| {
                        ensure_slots(&mut commands, &mut images, &mut mirrors, size);
                    },
                )
                .unwrap();
            let mirrors = world.resource::<Mirrors>();
            let images = world.resource::<Assets<Image>>();
            assert_eq!(mirrors.slots.len(), VIEW_LIMIT);
            for slot in &mirrors.slots {
                assert_ne!(slot.raw, slot.image);
                let RenderTarget::Image(target) = world.get::<RenderTarget>(slot.entity).unwrap()
                else {
                    panic!("mirror must render to its raw image");
                };
                assert_eq!(target.handle, slot.raw);
                assert_eq!(
                    world
                        .get::<crate::resolve::ReflectionOutput>(slot.entity)
                        .unwrap()
                        .0,
                    slot.image
                );
                assert!(world.get::<DepthPrepass>(slot.entity).is_some());
                assert_eq!(*world.get::<Msaa>(slot.entity).unwrap(), Msaa::Off);
                for handle in [&slot.raw, &slot.image] {
                    let descriptor = &images.get(handle).unwrap().texture_descriptor;
                    assert_eq!(descriptor.size.width, size.x);
                    assert_eq!(descriptor.size.height, size.y);
                    assert_eq!(descriptor.format, TextureFormat::Rgba16Float);
                    let is_output = handle == &slot.image;
                    assert_eq!(
                        descriptor.mip_level_count,
                        if is_output { size.x.max(size.y).ilog2() + 1 } else { 1 }
                    );
                    assert!(descriptor.usage.contains(
                        TextureUsages::TEXTURE_BINDING | TextureUsages::RENDER_ATTACHMENT
                    ));
                    if is_output {
                        assert!(descriptor.usage.contains(TextureUsages::STORAGE_BINDING));
                    }
                }
            }
        }
    }

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
