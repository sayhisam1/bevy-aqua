use bevy::{
    core_pipeline::{Core3dSystems, schedule::Core3d},
    prelude::*,
    render::{
        Render, RenderApp, RenderStartup, RenderSystems,
        extract_component::ExtractComponentPlugin,
        render_asset::RenderAssets,
        render_resource::{
            binding_types::{
                sampler, texture_2d, texture_2d_array, texture_storage_2d_array, uniform_buffer,
            },
            *,
        },
        renderer::{RenderContext, RenderDevice, RenderQueue, ViewQuery},
        texture::GpuImage,
    },
};

use super::{
    COMBINE_SHADER_PATH, FFT_EVOLVE_SHADER_PATH, FFT_RESOLVE_SHADER_PATH, FFT_SURFACE_SHADER_PATH,
    Frame, GATHER_SHADER_PATH, SHADER_PATH, Uniform, WORKGROUP_SIZE, fft,
};
use bevy_aqua_core::cascade as lod;
use bevy_aqua_core::{
    AnimWavesStatus, AnimWavesUniformSlot, AnimWavesWritten, LOD_COUNT, OceanView, WaveModel, bed,
    pass,
};

// Pass-table row keys; bind groups reuse them as labels where unique.
const GENERATE: &str = "AnimWaves generate";
const COMBINE: &str = "AnimWaves combine";
const GATHER: &str = "AnimWaves gather";
const EVOLVE: &str = "FFT evolve";
const TRANSFORM: &str = "FFT transform";
const RESOLVE: &str = "FFT resolve";
const SURFACE: &str = "FFT surface resolve";

const COMBINE_ENTRIES: &[&str] = &[
    "combine_0",
    "combine_1",
    "combine_2",
    "combine_3",
    "combine_4",
];
const HORIZONTAL_GROUPS: [&str; 2] = ["FFT horizontal 0", "FFT horizontal 1"];
const VERTICAL_GROUPS: [&str; 2] = ["FFT vertical 0", "FFT vertical 1"];

fn pass_table(stockham: &StockhamShader) -> Vec<pass::PassSpec> {
    let compute = ShaderStages::COMPUTE;
    let storage_rgba16 =
        || texture_storage_2d_array(TextureFormat::Rgba16Float, StorageTextureAccess::WriteOnly);
    let storage_rgba32 =
        || texture_storage_2d_array(TextureFormat::Rgba32Float, StorageTextureAccess::WriteOnly);
    let float_array = || texture_2d_array(TextureSampleType::Float { filterable: false });
    vec![
        pass::PassSpec {
            key: GENERATE,
            shader: pass::ShaderSource::Path(SHADER_PATH),
            entry_points: &["generate"],
            shader_defs: &[],
            wgsl_entry: None,
            layout: BindGroupLayoutDescriptor::new(
                GENERATE,
                &BindGroupLayoutEntries::sequential(
                    compute,
                    (
                        storage_rgba16(),
                        uniform_buffer::<Uniform>(false),
                        texture_2d(TextureSampleType::Float { filterable: false }),
                    ),
                ),
            ),
        },
        pass::PassSpec {
            key: COMBINE,
            shader: pass::ShaderSource::Path(COMBINE_SHADER_PATH),
            entry_points: COMBINE_ENTRIES,
            shader_defs: &[
                &["COMBINE_0"],
                &["COMBINE_1"],
                &["COMBINE_2"],
                &["COMBINE_3"],
                &["COMBINE_4"],
            ],
            wgsl_entry: Some("main"),
            layout: BindGroupLayoutDescriptor::new(
                COMBINE,
                &BindGroupLayoutEntries::sequential(
                    compute,
                    (
                        texture_2d_array(TextureSampleType::Float { filterable: true }),
                        texture_2d_array(TextureSampleType::Float { filterable: true }),
                        sampler(SamplerBindingType::Filtering),
                        storage_rgba16(),
                        uniform_buffer::<Uniform>(false),
                    ),
                ),
            ),
        },
        pass::PassSpec {
            key: GATHER,
            shader: pass::ShaderSource::Path(GATHER_SHADER_PATH),
            entry_points: &["gather"],
            shader_defs: &[],
            wgsl_entry: None,
            layout: BindGroupLayoutDescriptor::new(
                GATHER,
                &BindGroupLayoutEntries::sequential(
                    compute,
                    (float_array(), float_array(), storage_rgba16()),
                ),
            ),
        },
        pass::PassSpec {
            key: EVOLVE,
            shader: pass::ShaderSource::Path(FFT_EVOLVE_SHADER_PATH),
            entry_points: &["evolve"],
            shader_defs: &[],
            wgsl_entry: None,
            layout: BindGroupLayoutDescriptor::new(
                EVOLVE,
                &BindGroupLayoutEntries::sequential(
                    compute,
                    (
                        float_array(),
                        storage_rgba32(),
                        storage_rgba32(),
                        uniform_buffer::<fft::Uniform>(false),
                    ),
                ),
            ),
        },
        pass::PassSpec {
            key: TRANSFORM,
            shader: pass::ShaderSource::Handle(stockham.0.clone()),
            entry_points: &["horizontal", "vertical"],
            shader_defs: &[&[], &["FFT_VERTICAL"]],
            wgsl_entry: Some("main"),
            layout: BindGroupLayoutDescriptor::new(
                TRANSFORM,
                &BindGroupLayoutEntries::sequential(compute, (float_array(), storage_rgba32())),
            ),
        },
        pass::PassSpec {
            key: RESOLVE,
            shader: pass::ShaderSource::Path(FFT_RESOLVE_SHADER_PATH),
            entry_points: &["resolve"],
            shader_defs: &[],
            wgsl_entry: None,
            layout: BindGroupLayoutDescriptor::new(
                RESOLVE,
                &BindGroupLayoutEntries::sequential(
                    compute,
                    (
                        float_array(),
                        float_array(),
                        texture_2d(TextureSampleType::Float { filterable: false }),
                        storage_rgba16(),
                        uniform_buffer::<fft::Uniform>(false),
                    ),
                ),
            ),
        },
        pass::PassSpec {
            key: SURFACE,
            shader: pass::ShaderSource::Path(FFT_SURFACE_SHADER_PATH),
            entry_points: &["resolve_surface"],
            shader_defs: &[],
            wgsl_entry: None,
            layout: BindGroupLayoutDescriptor::new(
                SURFACE,
                &BindGroupLayoutEntries::sequential(
                    compute,
                    (
                        float_array(),
                        storage_rgba16(),
                        uniform_buffer::<fft::Uniform>(false),
                    ),
                ),
            ),
        },
    ]
}

#[derive(Resource, Clone, Debug)]
struct StockhamShader(Handle<Shader>);

/// Queued pipelines, per-frame uniforms, and bind groups for the wave
/// write node. The query pass reads the live uniform through `uniform`.
#[derive(Resource)]
pub struct Prepared {
    passes: pass::Passes,
    uniform: Option<UniformBuffer<Uniform>>,
    fft_uniform: Option<UniformBuffer<fft::Uniform>>,
    groups: pass::Groups,
}

impl std::fmt::Debug for Prepared {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Prepared")
            .field("has_uniform", &self.uniform.is_some())
            .finish_non_exhaustive()
    }
}

impl Prepared {
    /// The live AnimWaves uniform, shared with the wave-query pass so both
    /// always read one authoritative cascade layout upload.
    pub fn uniform(&self) -> Option<&UniformBuffer<Uniform>> {
        self.uniform.as_ref()
    }
}

#[derive(Debug)]
pub(crate) struct WaveRenderPlugin(pub Handle<Shader>);

impl Plugin for WaveRenderPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(ExtractComponentPlugin::<OceanView>::default());
        let Some(render) = app.get_sub_app_mut(RenderApp) else {
            // Logic-only consumers (isolation examples) run without a render
            // app; the write node simply never registers.
            return;
        };
        render
            .insert_resource(StockhamShader(self.0.clone()))
            .init_resource::<AnimWavesStatus>()
            .add_systems(RenderStartup, init_pipeline)
            .add_systems(
                Render,
                prepare_bind_groups.in_set(RenderSystems::PrepareBindGroups),
            )
            .add_systems(
                Core3d,
                write_anim_waves
                    .in_set(AnimWavesWritten)
                    .before(Core3dSystems::MainPass),
            );
    }
}

fn init_pipeline(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    stockham: Res<StockhamShader>,
    cache: Res<PipelineCache>,
) {
    commands.insert_resource(Prepared {
        passes: pass::Passes::new(&asset_server, &cache, pass_table(&stockham)),
        uniform: None,
        fft_uniform: None,
        groups: pass::Groups::default(),
    });
    // Shared slot: the query pass binds this buffer through the contract.
    commands.insert_resource(AnimWavesUniformSlot::default());
}

fn prepare_bind_groups(
    frame: Res<Frame>,
    world_slot: ResMut<AnimWavesUniformSlot>,
    bed: Option<Res<bed::BedHeightMap>>,
    fallback: Res<bed::GpuFallback>,
    images: Res<RenderAssets<GpuImage>>,
    device: (Res<RenderDevice>, Res<RenderQueue>, Res<PipelineCache>),
    mut prepared: ResMut<Prepared>,
) {
    // The bed is a world-static heightmap; scenes without one bind the
    // cleared fallback so shaders take their deep default.
    let Some(bed_gpu) = bed::gpu_image(bed.as_deref(), &fallback, &images) else {
        return;
    };
    let (Some(output), Some(raw), Some(bed), Some(scratch_a), Some(scratch_b)) = (
        images.get(&frame.output),
        images.get(&frame.raw),
        Some(bed_gpu),
        images.get(&frame.scratch_a),
        images.get(&frame.scratch_b),
    ) else {
        return;
    };
    let Some(h0_jonswap) = images.get(&frame.h0[0]) else {
        return;
    };
    let (Some(state_0), Some(state_1), Some(fft_scratch_0), Some(fft_scratch_1)) = (
        images.get(&frame.fft_state[0]),
        images.get(&frame.fft_state[1]),
        images.get(&frame.fft_scratch[0]),
        images.get(&frame.fft_scratch[1]),
    ) else {
        return;
    };

    // Uniforms refresh every frame, even after the groups exist.
    let Prepared {
        passes,
        uniform,
        fft_uniform,
        groups,
    } = &mut *prepared;
    pass::write_uniform(uniform, frame.uniform.clone(), &device.0, &device.1);
    pass::write_uniform(fft_uniform, frame.fft_uniform.clone(), &device.0, &device.1);
    // Publish into the contract slot for consumers (wave query).
    let slot = world_slot.into_inner();
    if slot.0.is_none() {
        slot.0 = Some(bevy::render::render_resource::UniformBuffer::from(
            frame.uniform.clone(),
        ));
    }
    if let Some(buffer) = slot.0.as_mut() {
        buffer.set(frame.uniform.clone());
        buffer.write_buffer(&device.0, &device.1);
    }
    if groups.created() {
        return;
    }
    let uniform = uniform.as_ref().unwrap();
    let fft_uniform = fft_uniform.as_ref().unwrap();
    let lod_layers = LOD_COUNT as u32;
    let displacement_sampled = cascade_array_view(
        &output.texture,
        0,
        lod_layers,
        TextureUsages::TEXTURE_BINDING,
    );
    let displacement_storage = cascade_array_view(
        &output.texture,
        0,
        lod_layers,
        TextureUsages::STORAGE_BINDING,
    );
    let surface_storage = cascade_array_view(
        &output.texture,
        lod_layers,
        lod_layers,
        TextureUsages::STORAGE_BINDING,
    );
    // Registers one bind group built from a table row's sequential layout.
    macro_rules! group {
        ($key:expr, $label:expr, $entries:expr) => {
            groups.register(
                $key,
                pass::bind_group(&device.0, &device.2, passes, $label, $label, $entries),
            )
        };
    }

    group!(
        "generate",
        GENERATE,
        &BindGroupEntries::sequential((&raw.texture_view, uniform, &bed.texture_view))
    );
    group!(
        "to_a",
        COMBINE,
        &BindGroupEntries::sequential((
            &raw.texture_view,
            &scratch_b.texture_view,
            &scratch_b.sampler,
            &scratch_a.texture_view,
            uniform
        ))
    );
    group!(
        "to_b",
        COMBINE,
        &BindGroupEntries::sequential((
            &raw.texture_view,
            &scratch_a.texture_view,
            &scratch_a.sampler,
            &scratch_b.texture_view,
            uniform
        ))
    );
    group!(
        "gather",
        GATHER,
        &BindGroupEntries::sequential((
            &scratch_a.texture_view,
            &scratch_b.texture_view,
            &displacement_storage
        ))
    );
    group!(
        "evolve",
        EVOLVE,
        &BindGroupEntries::sequential((
            &h0_jonswap.texture_view,
            &state_0.texture_view,
            &state_1.texture_view,
            fft_uniform
        ))
    );
    for (key, (source, target)) in HORIZONTAL_GROUPS
        .iter()
        .zip([(state_0, fft_scratch_0), (state_1, fft_scratch_1)])
    {
        group!(
            key,
            TRANSFORM,
            &BindGroupEntries::sequential((&source.texture_view, &target.texture_view))
        );
    }
    for (key, (source, target)) in VERTICAL_GROUPS
        .iter()
        .zip([(fft_scratch_0, state_0), (fft_scratch_1, state_1)])
    {
        group!(
            key,
            TRANSFORM,
            &BindGroupEntries::sequential((&source.texture_view, &target.texture_view))
        );
    }
    group!(
        "resolve",
        RESOLVE,
        &BindGroupEntries::sequential((
            &state_0.texture_view,
            &state_1.texture_view,
            &bed.texture_view,
            &raw.texture_view,
            fft_uniform
        ))
    );
    group!(
        "surface",
        SURFACE,
        &BindGroupEntries::sequential((&displacement_sampled, &surface_storage, fft_uniform,))
    );
}

fn cascade_array_view(
    texture: &Texture,
    base_array_layer: u32,
    array_layer_count: u32,
    usage: TextureUsages,
) -> TextureView {
    texture.create_view(&TextureViewDescriptor {
        format: Some(TextureFormat::Rgba16Float),
        dimension: Some(TextureViewDimension::D2Array),
        usage: Some(usage),
        base_array_layer,
        array_layer_count: Some(array_layer_count),
        ..default()
    })
}

fn cascade_grid(layers: u32) -> [u32; 3] {
    [
        lod::RESOLUTION / WORKGROUP_SIZE,
        lod::RESOLUTION / WORKGROUP_SIZE,
        layers,
    ]
}

fn transform_grid(layers: u32) -> [u32; 3] {
    [1, fft::RESOLUTION, layers]
}

fn gerstner_spans<'a>(
    prepared: &'a Prepared,
    cache: &'a PipelineCache,
) -> Option<Vec<pass::Span<'a>>> {
    let groups = &prepared.groups;
    let (Some(to_a), Some(to_b), Some(gather_group)) =
        (groups.get("to_a"), groups.get("to_b"), groups.get("gather"))
    else {
        return None;
    };
    let gather = prepared.passes.ready(cache, GATHER, "gather")?;
    let mut combines = Vec::new();
    for entry in COMBINE_ENTRIES {
        combines.push(prepared.passes.ready(cache, COMBINE, entry)?);
    }
    let (Some(group), Some(generate)) = (
        groups.get("generate"),
        prepared.passes.ready(cache, GENERATE, "generate"),
    ) else {
        return None;
    };
    let grid = [
        lod::RESOLUTION / WORKGROUP_SIZE,
        lod::RESOLUTION / WORKGROUP_SIZE,
        LOD_COUNT as u32,
    ];
    let mut steps = Vec::with_capacity(LOD_COUNT + 2);
    steps.push(pass::Step::Dispatch {
        pipeline: generate,
        group,
        workgroups: grid,
    });
    for (slice, combine) in (0..LOD_COUNT).rev().zip(combines.iter().rev()) {
        let target = if slice.is_multiple_of(2) { to_b } else { to_a };
        steps.push(pass::Step::Dispatch {
            pipeline: combine,
            group: target,
            workgroups: [grid[0], grid[1], 1],
        });
    }
    steps.push(pass::Step::Dispatch {
        pipeline: gather,
        group: gather_group,
        workgroups: grid,
    });
    Some(vec![pass::Span::new("aqua_cascade_compute", steps)])
}

fn fft_spans<'a>(
    frame: &Frame,
    prepared: &'a Prepared,
    cache: &'a PipelineCache,
) -> Option<Vec<pass::Span<'a>>> {
    // Deep-water scenes collapse the four shoaling bins to one.
    let field_layers = LOD_COUNT as u32 * frame.fft_bins;
    let mut requested = vec![
        (EVOLVE, "evolve"),
        (TRANSFORM, "horizontal"),
        (TRANSFORM, "vertical"),
        (RESOLVE, "resolve"),
        (GATHER, "gather"),
        (SURFACE, "resolve_surface"),
    ];
    requested.extend(COMBINE_ENTRIES.iter().map(|entry| (COMBINE, *entry)));
    let ready = prepared.passes.ready_all(cache, &requested)?;
    let groups = &prepared.groups;

    // One dispatch step each, keyed by (pass row, entry point, bind group).
    let single = |label, key, entry, group_key, workgroups| -> Option<pass::Span> {
        Some(pass::Span::new(
            label,
            vec![pass::Step::Dispatch {
                pipeline: ready.get(key, entry),
                group: groups.get(group_key)?,
                workgroups,
            }],
        ))
    };
    // Stockham transforms run two passes over doubled buffer slots.
    let butterfly = |label, entry| -> Option<pass::Span> {
        let keys: &[&str] = if entry == "horizontal" {
            &HORIZONTAL_GROUPS
        } else {
            &VERTICAL_GROUPS
        };
        let mut steps = Vec::with_capacity(keys.len());
        for key in keys {
            steps.push(pass::Step::Dispatch {
                pipeline: ready.get(TRANSFORM, entry),
                group: groups.get(key)?,
                workgroups: transform_grid(field_layers),
            });
        }
        Some(pass::Span::new(label, steps))
    };
    let mut combine_steps: Vec<pass::Step> = Vec::with_capacity(LOD_COUNT + 1);
    for slice in (0..LOD_COUNT).rev() {
        combine_steps.push(pass::Step::Dispatch {
            pipeline: ready.get(COMBINE, COMBINE_ENTRIES[slice]),
            group: groups.get(if slice.is_multiple_of(2) {
                "to_b"
            } else {
                "to_a"
            })?,
            workgroups: cascade_grid(1),
        });
    }
    combine_steps.push(pass::Step::Dispatch {
        pipeline: ready.get(GATHER, "gather"),
        group: groups.get("gather")?,
        workgroups: cascade_grid(LOD_COUNT as u32),
    });

    Some(vec![
        single(
            "aqua_fft_evolve",
            EVOLVE,
            "evolve",
            "evolve",
            cascade_grid(field_layers),
        )?,
        butterfly("aqua_fft_horizontal", "horizontal")?,
        butterfly("aqua_fft_vertical", "vertical")?,
        single(
            "aqua_fft_resolve",
            RESOLVE,
            "resolve",
            "resolve",
            cascade_grid(LOD_COUNT as u32),
        )?,
        pass::Span::new("aqua_cascade_combine", combine_steps),
        single(
            "aqua_fft_surface",
            SURFACE,
            "resolve_surface",
            "surface",
            cascade_grid(LOD_COUNT as u32),
        )?,
    ])
}

fn write_anim_waves(
    view: ViewQuery<Option<&OceanView>>,
    mut context: RenderContext,
    resources: (Res<Frame>, Res<Prepared>, Res<PipelineCache>),
    mut status: ResMut<AnimWavesStatus>,
) {
    if view.into_inner().is_none() {
        return;
    }
    status.written = false;
    let (frame, prepared, cache) = resources;
    let spans = match frame.model {
        WaveModel::Analytic => gerstner_spans(&prepared, &cache),
        WaveModel::Spectral => fft_spans(&frame, &prepared, &cache),
    };
    let Some(spans) = spans else { return };
    pass::run_spans(&mut context, &spans);
    status.written = true;
}
