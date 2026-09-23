//! Render-world fullscreen underwater volume composite.

use bevy::{
    asset::load_embedded_asset,
    core_pipeline::{
        FullscreenShader,
        schedule::{Core3d, Core3dSystems},
    },
    pbr::{
        MeshPipelineSystems, MeshPipelineViewLayoutKey, MeshPipelineViewLayouts, MeshViewBindGroup,
        ViewKeyCache,
    },
    prelude::*,
    render::{
        GpuResourceAppExt, Render, RenderApp, RenderStartup, RenderSystems,
        diagnostic::RecordDiagnostics,
        render_resource::{
            BindGroup, BindGroupEntries, BindGroupLayoutDescriptor, BindGroupLayoutEntries,
            CachedRenderPipelineId, ColorTargetState, ColorWrites, FilterMode, FragmentState,
            LoadOp, Operations, PipelineCache, RenderPassColorAttachment, RenderPassDescriptor,
            RenderPipelineDescriptor, Sampler, SamplerBindingType, SamplerDescriptor, ShaderStages,
            ShaderType, SpecializedRenderPipeline, SpecializedRenderPipelines, StoreOp,
            TextureFormat, TextureSampleType, TextureUsages, TextureView, TextureViewId,
            UniformBuffer,
            binding_types::{
                sampler, texture_2d, texture_depth_2d, texture_depth_2d_multisampled,
                uniform_buffer,
            },
        },
        renderer::{
            RenderAdapter, RenderAdapterInfo, RenderContext, RenderDevice, RenderQueue, ViewQuery,
            WgpuWrapper,
        },
        settings::WgpuFeatures,
        view::{ExtractedView, ViewDepthTexture, ViewTarget},
    },
    shader::ShaderDefVal,
};
use bevy_aqua_core::{OceanView, pass};

use super::ExtractedVolume;

// Mirrors Bevy's SSR capability gate. Bevy keeps its helper crate-private.
fn binding_arrays_are_usable(render_device: &RenderDevice, render_adapter: &RenderAdapter) -> bool {
    let adapter_info = RenderAdapterInfo(WgpuWrapper::new(render_adapter.get_info()));
    bevy::render::get_adreno_model(&adapter_info).is_none_or(|model| model > 610)
        && render_device
            .limits()
            .max_binding_array_elements_per_shader_stage
            >= 24
        && render_device
            .limits()
            .max_binding_array_sampler_elements_per_shader_stage
            >= 24
        && render_device.features().contains(
            WgpuFeatures::TEXTURE_BINDING_ARRAY
                | WgpuFeatures::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING,
        )
}

pub(super) fn add(app: &mut App) {
    let Some(render) = app.get_sub_app_mut(RenderApp) else {
        return;
    };
    render
        .init_gpu_resource::<SpecializedRenderPipelines<VolumePipeline>>()
        .add_systems(RenderStartup, init_pipeline.after(MeshPipelineSystems))
        .add_systems(
            Render,
            (
                prepare_pipelines.in_set(RenderSystems::Prepare),
                prepare_depth_usages
                    .in_set(RenderSystems::Prepare)
                    .before(bevy::core_pipeline::core_3d::prepare_core_3d_depth_textures),
                prepare_bind_groups.in_set(RenderSystems::PrepareBindGroups),
            ),
        )
        .add_systems(
            Core3d,
            draw_volume
                .after(Core3dSystems::MainPass)
                .before(Core3dSystems::EarlyPostProcess),
        );
}

#[derive(ShaderType, Clone, Copy, Debug, Default)]
struct VolumeUniform {
    extinction: Vec4,
    scatter: Vec4,
    environment: Vec4,
    sea: Vec4,
    interface_normal: Vec4,
}

#[derive(Resource)]
struct VolumePipeline {
    mesh_view_layouts: MeshPipelineViewLayouts,
    sampler: Sampler,
    layout: BindGroupLayoutDescriptor,
    layout_msaa: BindGroupLayoutDescriptor,
    fullscreen_shader: FullscreenShader,
    fragment_shader: Handle<Shader>,
    binding_arrays_are_usable: bool,
}

#[derive(Resource)]
struct Prepared {
    uniform: Option<UniformBuffer<VolumeUniform>>,
}

impl VolumePipeline {
    fn new(
        render_device: &RenderDevice,
        mesh_view_layouts: MeshPipelineViewLayouts,
        fullscreen_shader: FullscreenShader,
        fragment_shader: Handle<Shader>,
        binding_arrays_are_usable: bool,
    ) -> Self {
        let entries = BindGroupLayoutEntries::sequential(
            ShaderStages::FRAGMENT,
            (
                texture_2d(TextureSampleType::Float { filterable: true }),
                texture_depth_2d(),
                sampler(SamplerBindingType::Filtering),
                uniform_buffer::<VolumeUniform>(false),
            ),
        );
        let entries_msaa = BindGroupLayoutEntries::sequential(
            ShaderStages::FRAGMENT,
            (
                texture_2d(TextureSampleType::Float { filterable: true }),
                texture_depth_2d_multisampled(),
                sampler(SamplerBindingType::Filtering),
                uniform_buffer::<VolumeUniform>(false),
            ),
        );
        let sampler = render_device.create_sampler(&SamplerDescriptor {
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            ..default()
        });
        Self {
            mesh_view_layouts,
            sampler,
            layout: BindGroupLayoutDescriptor::new("aqua_volume_layout", &entries),
            layout_msaa: BindGroupLayoutDescriptor::new("aqua_volume_layout_msaa", &entries_msaa),
            fullscreen_shader,
            fragment_shader,
            binding_arrays_are_usable,
        }
    }
}

fn init_pipeline(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    render_adapter: Res<RenderAdapter>,
    mesh_view_layouts: Res<MeshPipelineViewLayouts>,
    fullscreen_shader: Res<FullscreenShader>,
    asset_server: Res<AssetServer>,
) {
    let fragment_shader = load_embedded_asset!(asset_server.as_ref(), "volume.wgsl");
    commands.insert_resource(VolumePipeline::new(
        &render_device,
        mesh_view_layouts.clone(),
        fullscreen_shader.clone(),
        fragment_shader,
        binding_arrays_are_usable(&render_device, &render_adapter),
    ));
    commands.insert_resource(Prepared { uniform: None });
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct VolumePipelineKey {
    mesh_pipeline_view_key: MeshPipelineViewLayoutKey,
    target_format: TextureFormat,
    samples: u32,
}

impl SpecializedRenderPipeline for VolumePipeline {
    type Key = VolumePipelineKey;

    fn specialize(&self, key: Self::Key) -> RenderPipelineDescriptor {
        let view_layout = self
            .mesh_view_layouts
            .get_view_layout(key.mesh_pipeline_view_key);
        let volume_layout = if key.samples > 1 {
            self.layout_msaa.clone()
        } else {
            self.layout.clone()
        };
        let mut shader_defs = Vec::new();
        if key.samples > 1 {
            shader_defs.push(ShaderDefVal::from("MULTISAMPLED"));
        }
        if key
            .mesh_pipeline_view_key
            .contains(MeshPipelineViewLayoutKey::ENVIRONMENT_MAP)
        {
            shader_defs.push(ShaderDefVal::from("ENVIRONMENT_MAP"));
        }
        if self.binding_arrays_are_usable {
            shader_defs.push(ShaderDefVal::from("MULTIPLE_LIGHT_PROBES_IN_ARRAY"));
        }
        if key
            .mesh_pipeline_view_key
            .contains(MeshPipelineViewLayoutKey::ATMOSPHERE)
        {
            shader_defs.push(ShaderDefVal::from("ATMOSPHERE"));
        }
        RenderPipelineDescriptor {
            label: Some("aqua_volume_pipeline".into()),
            layout: vec![
                view_layout.main_layout,
                view_layout.binding_array_layout,
                volume_layout,
            ],
            vertex: self.fullscreen_shader.to_vertex_state(),
            fragment: Some(FragmentState {
                shader: self.fragment_shader.clone(),
                shader_defs,
                targets: vec![Some(ColorTargetState {
                    format: key.target_format,
                    blend: None,
                    write_mask: ColorWrites::ALL,
                })],
                ..default()
            }),
            ..default()
        }
    }
}

#[derive(Component)]
struct VolumePipelineId(CachedRenderPipelineId);

fn prepare_pipelines(
    mut commands: Commands,
    pipeline_cache: Res<PipelineCache>,
    mut pipelines: ResMut<SpecializedRenderPipelines<VolumePipeline>>,
    pipeline: Res<VolumePipeline>,
    view_key_cache: Res<ViewKeyCache>,
    cameras: Query<(Entity, &ExtractedView, &Msaa), With<OceanView>>,
) {
    for (entity, view, msaa) in &cameras {
        let Some(mesh_pipeline_key) = view_key_cache.get(&view.retained_view_entity) else {
            continue;
        };
        let pipeline_id = pipelines.specialize(
            &pipeline_cache,
            &pipeline,
            VolumePipelineKey {
                mesh_pipeline_view_key: (*mesh_pipeline_key).into(),
                target_format: view.target_format,
                samples: msaa.samples(),
            },
        );
        commands
            .entity(entity)
            .insert(VolumePipelineId(pipeline_id));
    }
}

fn prepare_depth_usages(mut cameras: Query<&mut Camera3d, With<OceanView>>) {
    for mut camera in &mut cameras {
        camera.depth_texture_usages.0 |= TextureUsages::TEXTURE_BINDING.bits();
    }
}

fn volume_uniform(volume: &ExtractedVolume) -> VolumeUniform {
    let optics = volume.optics;
    VolumeUniform {
        extinction: optics.extinction.extend(optics.scatter_scale.max(0.0)),
        scatter: optics.scatter_tint.max(Vec3::ZERO).extend(0.0),
        environment: Vec4::new(
            volume.optics.scattering_asymmetry,
            if volume.receiver_relighting { 1.0 } else { 0.0 },
            0.0,
            0.0,
        ),
        sea: Vec4::new(volume.surface_level, volume.camera_y, 0.0, 0.0),
        interface_normal: volume.interface_normal.extend(0.0),
    }
}

struct CachedGroup {
    color: TextureViewId,
    depth: TextureViewId,
    group: BindGroup,
}

#[derive(Component)]
struct VolumeBindGroups {
    samples: u32,
    a: CachedGroup,
    b: CachedGroup,
}

fn prepare_bind_groups(
    mut commands: Commands,
    volume: Res<ExtractedVolume>,
    pipeline: Res<VolumePipeline>,
    pipeline_cache: Res<PipelineCache>,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    mut prepared: ResMut<Prepared>,
    views: Query<
        (
            Entity,
            &ViewTarget,
            &ViewDepthTexture,
            &Msaa,
            Option<&VolumeBindGroups>,
        ),
        With<OceanView>,
    >,
) {
    if !volume.active {
        return;
    }
    pass::write_uniform(
        &mut prepared.uniform,
        volume_uniform(&volume),
        &device,
        &queue,
    );
    let Some(uniform) = prepared.uniform.as_ref().and_then(UniformBuffer::binding) else {
        return;
    };

    for (entity, target, depth, msaa, cached) in &views {
        let samples = msaa.samples();
        let layout = if samples > 1 {
            &pipeline.layout_msaa
        } else {
            &pipeline.layout
        };
        let depth_view = depth.view();
        let a_color = target.main_texture_view();
        let b_color = target.main_texture_other_view();
        let fresh = cached.is_none_or(|groups| {
            groups.samples != samples
                || groups.a.color != a_color.id()
                || groups.a.depth != depth_view.id()
                || groups.b.color != b_color.id()
                || groups.b.depth != depth_view.id()
        });
        if !fresh {
            continue;
        }
        let make = |color: &TextureView| CachedGroup {
            color: color.id(),
            depth: depth_view.id(),
            group: device.create_bind_group(
                Some("aqua_volume_bind_group"),
                &pipeline_cache.get_bind_group_layout(layout),
                &BindGroupEntries::sequential((
                    color,
                    depth_view,
                    &pipeline.sampler,
                    uniform.clone(),
                )),
            ),
        };
        commands.entity(entity).insert(VolumeBindGroups {
            samples,
            a: make(a_color),
            b: make(b_color),
        });
    }
}

fn draw_volume(
    view: ViewQuery<(
        Option<&OceanView>,
        Option<&VolumePipelineId>,
        Option<&VolumeBindGroups>,
        Option<&MeshViewBindGroup>,
        &ViewTarget,
    )>,
    volume: Res<ExtractedVolume>,
    pipeline_cache: Res<PipelineCache>,
    mut ctx: RenderContext,
) {
    let (ocean_view, pipeline_id, bind_groups, view_bind_group, view_target) = view.into_inner();
    if ocean_view.is_none() || !volume.active {
        return;
    }
    let Some(pipeline_id) = pipeline_id else {
        return;
    };
    let Some(bind_groups) = bind_groups else {
        return;
    };
    let Some(view_bind_group) = view_bind_group else {
        return;
    };
    let Some(gpu_pipeline) = pipeline_cache.get_render_pipeline(pipeline_id.0) else {
        return;
    };

    let post_process = view_target.post_process_write();
    let bind_group = if bind_groups.a.color == post_process.source.id() {
        &bind_groups.a.group
    } else {
        &bind_groups.b.group
    };

    let diagnostics = ctx.diagnostic_recorder();
    let diagnostics = diagnostics.as_deref();
    let mut render_pass = ctx.begin_tracked_render_pass(RenderPassDescriptor {
        label: Some("aqua_volume"),
        color_attachments: &[Some(RenderPassColorAttachment {
            view: post_process.destination,
            depth_slice: None,
            resolve_target: None,
            ops: Operations {
                // The fullscreen shader copies source pixels outside this view's
                // viewport, so the ping-pong destination is fully initialized.
                load: LoadOp::Clear(Default::default()),
                store: StoreOp::Store,
            },
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });
    let pass_span = diagnostics.pass_span(&mut render_pass, "aqua_volume");
    render_pass.set_render_pipeline(gpu_pipeline);
    render_pass.set_bind_group(0, &view_bind_group.main, &view_bind_group.main_offsets);
    render_pass.set_bind_group(1, &view_bind_group.binding_array, &[]);
    render_pass.set_bind_group(2, bind_group, &[]);
    render_pass.draw(0..3, 0..1);
    pass_span.end(&mut render_pass);
}

#[cfg(test)]
mod tests {
    use bevy::prelude::Vec3;
    #[test]
    fn fullscreen_copy_initializes_ping_pong_destination() {
        let source = include_str!("render.rs");
        let render = source.split("#[cfg(test)]").next().unwrap();
        let shader = include_str!("volume.wgsl");
        assert!(render.contains("load: LoadOp::Clear"));
        assert!(!render.contains("render_pass.set_viewport("));
        assert!(shader.contains("if any(in.position.xy < viewport_min)"));
        assert!(shader.contains("return vec4(scene, 1.0);"));
    }

    #[test]
    fn multisampled_depth_is_resolved_for_single_sample_post_target() {
        let shader = include_str!("volume.wgsl");
        assert!(shader.contains("textureNumSamples(depth_texture)"));
        assert!(shader.contains("raw_depth = max(raw_depth"));
        assert!(!shader.contains("@builtin(sample_index)"));
    }

    #[test]
    fn upward_depth_hit_retains_displaced_surface_endpoint() {
        let shader = include_str!("volume.wgsl");
        assert!(shader.contains("dot(rd_world, facet_up) > 1e-6 && raw_depth <= 0.0"));
        assert!(!shader.contains("t_surface < t_scene"));
    }

    #[test]
    fn near_clip_fills_sub_near_and_distant_gaps_stay_open() {
        fn local_exit(camera_y: f32, level: f32, ray: Vec3, normal: Vec3) -> f32 {
            (level - camera_y) * normal.y / ray.dot(normal)
        }
        let normal = Vec3::new(0.0, 1.0, 1.0).normalize();
        let near = 0.25;
        let clipped = local_exit(-0.1, 0.0, Vec3::Y, normal);
        let visible = local_exit(-1.0, 0.0, Vec3::Y, normal);
        assert!(clipped > 0.0 && clipped <= near + 1e-4);
        assert!(visible > near + 1e-4);

        let shader = include_str!("volume.wgsl");
        assert!(shader.contains("t_surface > 0.0 && t_surface <= ray.near_distance + 1e-4"));
        assert!(shader.contains("if clipped_before_near {"));
        assert!(!shader.contains("volume.environment.z"));
        assert!(!shader.contains("scene = vec3(0.0)"));
        assert!(shader.contains("raw_depth <= 0.0"));
        assert!(shader.contains("dot(rd_world, facet_up) > 1e-6"));
    }

    #[test]
    fn synthesized_boundary_matches_exact_underside_interface_physics() {
        let shader = include_str!("volume.wgsl");
        let boundary = shader
            .split("fn terminal_underside_radiance(")
            .nth(1)
            .unwrap()
            .split("@fragment")
            .next()
            .unwrap();
        assert!(boundary.contains("refract(incident, water_normal, N_WATER)"));
        assert!(boundary.contains("fresnel_water_to_air(cos_water)"));
        assert!(boundary.contains("medium_radiance_oriented("));
        assert!(boundary.contains("facet_up,"));
        assert!(boundary.contains("sample_environment(transmitted_direction, facet_up, 0.0)"));
        assert!(boundary.contains("N_WATER * N_WATER"));
        let tir = boundary.find("if dot(transmitted_direction").unwrap();
        let environment = boundary.find("sample_environment(").unwrap();
        assert!(
            tir < environment,
            "TIR must return before environment sampling"
        );
        assert!(!boundary.contains("screen_texture"));
        assert!(!boundary.contains("max(window"));
    }

    #[test]
    fn environment_layout_uses_main_binding_array_and_volume_groups() {
        let source = include_str!("render.rs");
        let render = source.split("#[cfg(test)]").next().unwrap();
        let shader = include_str!("volume.wgsl");
        assert!(render.contains("view_layout.binding_array_layout"));
        assert!(render.contains("layout: vec!["));
        assert!(render.contains("MeshPipelineViewLayoutKey::ENVIRONMENT_MAP"));
        assert!(render.contains("MULTIPLE_LIGHT_PROBES_IN_ARRAY"));
        assert!(render.contains("binding_arrays_are_usable(&render_device, &render_adapter)"));
        assert!(render.contains("set_bind_group(1, &view_bind_group.binding_array"));
        assert!(render.contains("set_bind_group(2, bind_group"));
        assert!(shader.contains("@group(2) @binding(0)"));
    }
}
