//! Render-world compute pipeline for persistent foam.

use bevy::{
    core_pipeline::{Core3dSystems, schedule::Core3d},
    prelude::*,
    render::{
        Render, RenderApp, RenderStartup, RenderSystems,
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

use super::{Frame, MAX_CATCH_UP_STEPS, RESOLUTION, SHADER_PATH, Uniform, WORKGROUP_SIZE};
use bevy_aqua_core::{LOD_COUNT, OceanView};
use bevy_aqua_core::{bed, pass};

const UPDATE: &str = "Aqua foam update";
const PREVIOUS: &str = "update_previous_layout";
const PREVIOUS_ZERO: &str = "reproject_previous_layout";
const CURRENT: &str = "update_current_layout";

pub(super) fn add(app: &mut App) {
    let Some(render) = app.get_sub_app_mut(RenderApp) else {
        // Logic-only consumers (isolation examples) run without a render
        // app; the write node simply never registers.
        return;
    };
    render
        .add_systems(RenderStartup, init_pipeline)
        .add_systems(
            Render,
            prepare_bind_groups.in_set(RenderSystems::PrepareBindGroups),
        )
        .add_systems(
            Core3d,
            write_foam
                .after(bevy_aqua_core::AnimWavesWritten)
                .before(Core3dSystems::MainPass),
        );
}

fn pass_table() -> Vec<pass::PassSpec> {
    vec![pass::PassSpec {
        key: UPDATE,
        shader: pass::ShaderSource::Path(SHADER_PATH),
        entry_points: &[PREVIOUS, PREVIOUS_ZERO, CURRENT],
        shader_defs: &[&[], &["FOAM_REPROJECT_PREVIOUS"], &["FOAM_UPDATE_CURRENT"]],
        wgsl_entry: Some("main"),
        layout: BindGroupLayoutDescriptor::new(
            UPDATE,
            &BindGroupLayoutEntries::sequential(
                ShaderStages::COMPUTE,
                (
                    texture_2d_array(TextureSampleType::Float { filterable: true }),
                    sampler(SamplerBindingType::Filtering),
                    texture_2d_array(TextureSampleType::Float { filterable: true }),
                    sampler(SamplerBindingType::Filtering),
                    texture_2d(TextureSampleType::Float { filterable: false }),
                    texture_storage_2d_array(
                        TextureFormat::Rgba16Float,
                        StorageTextureAccess::WriteOnly,
                    ),
                    uniform_buffer::<Uniform>(false),
                ),
            ),
        ),
    }]
}

#[derive(Resource)]
struct Prepared {
    passes: pass::Passes,
    uniform: Option<UniformBuffer<Uniform>>,
    groups: pass::Groups,
    state_is_a: bool,
    completed_tick: u32,
    state_layout: Option<bevy_aqua_core::GpuLayout>,
}

fn init_pipeline(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    cache: Res<PipelineCache>,
) {
    commands.insert_resource(Prepared {
        passes: pass::Passes::new(&asset_server, &cache, pass_table()),
        uniform: None,
        groups: pass::Groups::default(),
        state_is_a: true,
        completed_tick: 0,
        state_layout: None,
    });
}

fn prepare_bind_groups(
    frame: Res<Frame>,
    bed: Option<Res<bed::BedHeightMap>>,
    fallback: Res<bed::GpuFallback>,
    images: Res<RenderAssets<GpuImage>>,
    device: (Res<RenderDevice>, Res<RenderQueue>, Res<PipelineCache>),
    mut prepared: ResMut<Prepared>,
) {
    let Some(depth) = bed::gpu_image(bed.as_deref(), &fallback, &images) else {
        return;
    };
    let (Some(state_a), Some(state_b), Some(waves)) = (
        images.get(&frame.state_a),
        images.get(&frame.state_b),
        images.get(&frame.waves),
    ) else {
        return;
    };

    let mut uniform_value = frame.uniform.clone();
    if let Some(layout) = prepared.state_layout.as_ref() {
        uniform_value.source_layout = layout.clone();
    }
    pass::write_uniform(&mut prepared.uniform, uniform_value, &device.0, &device.1);

    if prepared.groups.created() {
        return;
    }
    let Prepared {
        passes,
        uniform,
        groups,
        ..
    } = &mut *prepared;
    let uniform = uniform.as_ref().unwrap();
    macro_rules! direction {
        ($key:expr, $label:expr, $write:expr, $keep:expr) => {
            groups.register(
                $key,
                pass::bind_group(
                    &device.0,
                    &device.2,
                    passes,
                    UPDATE,
                    $label,
                    &BindGroupEntries::sequential((
                        &$keep.texture_view,
                        &$keep.sampler,
                        &waves.texture_view,
                        &waves.sampler,
                        &depth.texture_view,
                        &$write.texture_view,
                        uniform,
                    )),
                ),
            );
        };
    }
    direction!("a_to_b", "Aqua foam A to B", state_b, state_a);
    direction!("b_to_a", "Aqua foam B to A", state_a, state_b);
}

fn write_foam(
    view: ViewQuery<Option<&OceanView>>,
    mut context: RenderContext,
    mut prepared: ResMut<Prepared>,
    resources: (Res<Frame>, Res<PipelineCache>, Res<RenderAssets<GpuImage>>),
    statuses: (Res<bevy_aqua_core::AnimWavesStatus>,),
) {
    let (frame, cache, images) = resources;
    let (waves_status,) = statuses;
    if view.into_inner().is_none() {
        return;
    }
    if !waves_status.written {
        return;
    }
    let (Some(state_a), Some(state_b), Some(surface)) = (
        images.get(&frame.state_a),
        images.get(&frame.state_b),
        images.get(&frame.surface),
    ) else {
        return;
    };
    // Compute the outcome before borrowing bind groups from `prepared`.
    let pending_steps = frame.uniform.step.x.saturating_sub(prepared.completed_tick);
    let dispatch_count = pending_steps.clamp(1, MAX_CATCH_UP_STEPS);
    let state_is_a = prepared.state_is_a;
    let new_state_is_a = state_is_a != (dispatch_count % 2 == 1);
    prepared.state_is_a = new_state_is_a;
    prepared.completed_tick = prepared
        .completed_tick
        .saturating_add(pending_steps.min(MAX_CATCH_UP_STEPS));
    prepared.state_layout = Some(frame.uniform.target_layout.clone());

    let (Some(group_a_to_b), Some(group_b_to_a)) =
        (prepared.groups.get("a_to_b"), prepared.groups.get("b_to_a"))
    else {
        return;
    };
    // Resolve every entry point up front so a cold pipeline frame opens no
    // diagnostics span.
    let Some(ready) = prepared.passes.ready_all(
        &cache,
        &[
            (UPDATE, PREVIOUS),
            (UPDATE, PREVIOUS_ZERO),
            (UPDATE, CURRENT),
        ],
    ) else {
        return;
    };

    let workgroups = [
        RESOLUTION / WORKGROUP_SIZE,
        RESOLUTION / WORKGROUP_SIZE,
        LOD_COUNT as u32,
    ];
    let mut state_is_a = state_is_a;
    let mut steps = Vec::with_capacity(dispatch_count as usize + 1);
    for step in 0..dispatch_count {
        let group = if state_is_a {
            group_a_to_b
        } else {
            group_b_to_a
        };
        let update = if pending_steps == 0 {
            ready.get(UPDATE, PREVIOUS_ZERO)
        } else if step == 0 {
            ready.get(UPDATE, PREVIOUS)
        } else {
            ready.get(UPDATE, CURRENT)
        };
        steps.push(pass::Step::Dispatch {
            pipeline: update,
            group,
            workgroups,
        });
        state_is_a = !state_is_a;
    }
    let source = if state_is_a { state_a } else { state_b };
    steps.push(pass::Step::CopyTexture {
        source: &source.texture,
        target: &surface.texture,
        extent: Extent3d {
            width: RESOLUTION,
            height: RESOLUTION,
            depth_or_array_layers: LOD_COUNT as u32,
        },
    });
    pass::run_spans(&mut context, &[pass::Span::new("aqua_foam_compute", steps)]);
}
