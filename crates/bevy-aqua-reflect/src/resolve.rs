//! Depth-authoritative planar color export. Background stays invalid.
use bevy::{
    asset::embedded_asset,
    core_pipeline::{Core3dSystems, prepass::ViewPrepassTextures, schedule::Core3d},
    prelude::*,
    render::{
        RenderApp, RenderStartup,
        extract_component::{ExtractComponent, ExtractComponentPlugin},
        render_asset::RenderAssets,
        render_resource::{
            binding_types::{texture_2d, texture_depth_2d, texture_storage_2d},
            *,
        },
        renderer::{RenderContext, RenderDevice, ViewQuery},
        texture::GpuImage,
        view::ViewTarget,
    },
};
use bevy_aqua_core::pass::{self, PassSpec, Passes, ShaderSource, Span, Step};

#[derive(Component, Clone, Debug, ExtractComponent)]
pub(crate) struct ReflectionOutput(pub Handle<Image>);

#[derive(Resource, Debug)]
struct ResolvePipeline(Passes);

pub(crate) fn add(app: &mut App) {
    embedded_asset!(app, "resolve.wgsl");
    embedded_asset!(app, "mips.wgsl");
    app.add_plugins(ExtractComponentPlugin::<ReflectionOutput>::default());
    let Some(render) = app.get_sub_app_mut(RenderApp) else {
        return;
    };
    render.add_systems(RenderStartup, init).add_systems(
        Core3d,
        resolve
            .after(Core3dSystems::MainPass)
            .before(Core3dSystems::EarlyPostProcess),
    );
}

fn init(mut commands: Commands, server: Res<AssetServer>, cache: Res<PipelineCache>) {
    commands.insert_resource(ResolvePipeline(Passes::new(
        &server,
        &cache,
        vec![
            PassSpec {
                key: "aqua_planar_resolve",
                shader: ShaderSource::Path("embedded://bevy_aqua_reflect/resolve.wgsl"),
                entry_points: &["main"],
                shader_defs: &[],
                wgsl_entry: None,
                layout: BindGroupLayoutDescriptor::new(
                    "aqua_planar_resolve",
                    &BindGroupLayoutEntries::sequential(
                        ShaderStages::COMPUTE,
                        (
                            texture_2d(TextureSampleType::Float { filterable: false }),
                            texture_depth_2d(),
                            texture_storage_2d(
                                TextureFormat::Rgba16Float,
                                StorageTextureAccess::WriteOnly,
                            ),
                        ),
                    ),
                ),
            },
            PassSpec {
                key: "aqua_planar_mip",
                shader: ShaderSource::Path("embedded://bevy_aqua_reflect/mips.wgsl"),
                entry_points: &["main"],
                shader_defs: &[],
                wgsl_entry: None,
                layout: BindGroupLayoutDescriptor::new(
                    "aqua_planar_mip",
                    &BindGroupLayoutEntries::sequential(
                        ShaderStages::COMPUTE,
                        (
                            texture_2d(TextureSampleType::Float { filterable: false }),
                            texture_storage_2d(
                                TextureFormat::Rgba16Float,
                                StorageTextureAccess::WriteOnly,
                            ),
                        ),
                    ),
                ),
            },
        ],
    )));
}

fn resolve(
    view: ViewQuery<(
        &ViewTarget,
        Option<&ViewPrepassTextures>,
        Option<&ReflectionOutput>,
    )>,
    mut context: RenderContext,
    images: Res<RenderAssets<GpuImage>>,
    pipelines: Res<ResolvePipeline>,
    cache: Res<PipelineCache>,
    device: Res<RenderDevice>,
) {
    let (target, prepass, output) = view.into_inner();
    let Some(output) = output.and_then(|o| images.get(&o.0)) else {
        return;
    };
    let levels = output.texture_descriptor.mip_level_count;
    let views: Vec<_> = (0..levels)
        .map(|level| {
            output.texture.create_view(&TextureViewDescriptor {
                label: Some("aqua_planar_single_mip"),
                dimension: Some(TextureViewDimension::D2),
                base_mip_level: level,
                mip_level_count: Some(1),
                ..default()
            })
        })
        .collect();
    let ready = prepass
        .and_then(ViewPrepassTextures::depth_view)
        .zip(pipelines.0.ready(&cache, "aqua_planar_resolve", "main"))
        .zip(pipelines.0.ready(&cache, "aqua_planar_mip", "main"));
    let Some(((depth, pipeline), mip_pipeline)) = ready else {
        // Fail closed at EVERY sampled level. No clear on a successful resolve.
        use bevy::render::diagnostic::RecordDiagnostics;
        let recorder = context.diagnostic_recorder();
        let diagnostics = recorder.as_deref();
        for view in &views {
            let span =
                diagnostics.time_span(context.command_encoder(), "aqua_planar_validity_clear");
            {
                let attachments = [Some(RenderPassColorAttachment {
                    view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: Operations {
                        load: LoadOp::Clear(Default::default()),
                        store: StoreOp::Store,
                    },
                })];
                let _pass = context
                    .command_encoder()
                    .begin_render_pass(&RenderPassDescriptor {
                        label: Some("aqua_planar_validity_clear"),
                        color_attachments: &attachments,
                        ..default()
                    });
            }
            span.end(context.command_encoder());
        }
        return;
    };
    let group = pass::bind_group(
        &device,
        &cache,
        &pipelines.0,
        "aqua_planar_resolve",
        "aqua_planar_resolve",
        &BindGroupEntries::sequential((target.main_texture_view(), depth, &views[0])),
    );
    let size = output.texture.size();
    pass::run_spans(
        &mut context,
        &[Span::new(
            "aqua_planar_resolve",
            vec![Step::Dispatch {
                pipeline,
                group: &group,
                workgroups: [size.width.div_ceil(8), size.height.div_ceil(8), 1],
            }],
        )],
    );
    for level in 1..levels {
        // Disjoint single-mip views allow sampled input + storage output on
        // the same texture. A separate compute pass orders each dependency.
        let group = pass::bind_group(
            &device,
            &cache,
            &pipelines.0,
            "aqua_planar_mip",
            "aqua_planar_mip",
            &BindGroupEntries::sequential((&views[level as usize - 1], &views[level as usize])),
        );
        let width = (size.width >> level).max(1);
        let height = (size.height >> level).max(1);
        pass::run_spans(
            &mut context,
            &[Span::new(
                "aqua_planar_mip",
                vec![Step::Dispatch {
                    pipeline: mip_pipeline,
                    group: &group,
                    workgroups: [width.div_ceil(8), height.div_ceil(8), 1],
                }],
            )],
        );
    }
}
