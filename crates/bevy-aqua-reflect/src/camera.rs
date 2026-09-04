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
    app.init_resource::<Mirrors>()
        .add_systems(Update, include_directional_lights)
        .add_systems(
            PostUpdate,
            (
                include_marked,
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
                    .expect("raw mirror image")
                    .resize(extent);
                images
                    .get_mut(&slot.image)
                    .expect("mirror image existence checked above")
                    .resize(extent);
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
        let image = images.add(reflection_image(size));
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
    image.texture_descriptor.usage |=
        TextureUsages::RENDER_ATTACHMENT | TextureUsages::STORAGE_BINDING;
    image.sampler = ImageSampler::Descriptor(ImageSamplerDescriptor {
        mag_filter: ImageFilterMode::Linear,
        min_filter: ImageFilterMode::Linear,
        ..default()
    });
    image
}

#[cfg(test)]
mod tests {
    use super::*;

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
