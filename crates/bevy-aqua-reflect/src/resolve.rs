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
        vec![PassSpec {
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
        }],
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
    // Fail closed on cold pipelines or missing depth. Clear has its own GPU span.
    use bevy::render::diagnostic::RecordDiagnostics;
    let recorder = context.diagnostic_recorder();
    let diagnostics = recorder.as_deref();
    let span = diagnostics.time_span(context.command_encoder(), "aqua_planar_validity_clear");
    {
        let attachments = [Some(RenderPassColorAttachment {
            view: &output.texture_view,
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
    let Some(depth) = prepass.and_then(ViewPrepassTextures::depth_view) else {
        return;
    };
    let Some(pipeline) = pipelines.0.ready(&cache, "aqua_planar_resolve", "main") else {
        return;
    };
    let group = pass::bind_group(
        &device,
        &cache,
        &pipelines.0,
        "aqua_planar_resolve",
        "aqua_planar_resolve",
        &BindGroupEntries::sequential((target.main_texture_view(), depth, &output.texture_view)),
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
}
