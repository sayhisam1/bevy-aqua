use bevy_aqua_core::{
    AnimWavesStatus, AnimWavesUniform, AnimWavesWritten, GpuLayout, LOD_COUNT, OceanView,
    RESOLUTION, pass,
};
use bevy_aqua_waves::Frame;

use crate::MotionEpoch;

const MAX_HISTORY_DELTA_SECONDS: f32 = 0.25;
use bevy::{
    core_pipeline::{Core3dSystems, prepass::ViewPrepassTextures, schedule::Core3d},
    prelude::*,
    render::{
        Render, RenderApp, RenderStartup, RenderSystems,
        diagnostic::RecordDiagnostics,
        render_asset::RenderAssets,
        render_resource::{
            BindGroup, BindGroupEntries, BindGroupLayout, BindGroupLayoutEntries, Extent3d,
            Origin3d, Sampler, SamplerBindingType, SamplerDescriptor, ShaderStages, ShaderType,
            TexelCopyTextureInfo, Texture, TextureAspect, TextureDescriptor, TextureDimension,
            TextureFormat, TextureSampleType, TextureUsages, TextureView, TextureViewDescriptor,
            TextureViewDimension, UniformBuffer,
            binding_types::{sampler, texture_2d_array, uniform_buffer},
        },
        renderer::{RenderContext, RenderDevice, RenderQueue, ViewQuery},
        texture::GpuImage,
    },
};

pub(crate) const HISTORY_EXTENT: Extent3d = Extent3d {
    width: RESOLUTION,
    height: RESOLUTION,
    depth_or_array_layers: LOD_COUNT as u32,
};

#[derive(Clone, Debug, ShaderType)]
pub(crate) struct PreviousFrame {
    pub(crate) layout: GpuLayout,
    pub(crate) time: Vec4,
    pub(crate) flow: Vec4,
    // x: 1 when previous displacement and frame state are valid.
    pub(crate) valid: Vec4,
}

#[derive(Resource)]
pub(crate) struct History {
    pub(crate) texture: Texture,
    pub(crate) view: TextureView,
    pub(crate) sampler: Sampler,
    pub(crate) layout: BindGroupLayout,
    pub(crate) group: Option<BindGroup>,
    pub(crate) uniform: Option<UniformBuffer<PreviousFrame>>,
    pub(crate) previous: Option<AnimWavesUniform>,
    previous_epoch: Option<u64>,
}

pub(crate) fn add_render_systems(app: &mut App) {
    let Some(render) = app.get_sub_app_mut(RenderApp) else {
        return;
    };
    render
        .add_systems(RenderStartup, init_history)
        .add_systems(
            Render,
            prepare_history.in_set(RenderSystems::PrepareBindGroups),
        )
        .configure_sets(Core3d, AnimWavesWritten.before(Core3dSystems::Prepass))
        .add_systems(
            Core3d,
            capture_previous
                .before(AnimWavesWritten)
                .run_if(main_motion_view),
        )
        .add_systems(
            Core3d,
            commit_previous
                .after(AnimWavesWritten)
                .before(Core3dSystems::Prepass)
                .run_if(main_motion_view),
        );
}

pub(crate) fn init_history(mut commands: Commands, device: Res<RenderDevice>) {
    let texture = device.create_texture(&TextureDescriptor {
        label: Some("aqua_previous_displacement"),
        size: HISTORY_EXTENT,
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::Rgba16Float,
        usage: TextureUsages::COPY_DST | TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = texture.create_view(&TextureViewDescriptor {
        label: Some("aqua_previous_displacement"),
        dimension: Some(TextureViewDimension::D2Array),
        ..default()
    });
    let history_sampler = device.create_sampler(&SamplerDescriptor {
        label: Some("aqua_previous_displacement"),
        mag_filter: bevy::render::render_resource::FilterMode::Linear,
        min_filter: bevy::render::render_resource::FilterMode::Linear,
        ..default()
    });
    let layout = device.create_bind_group_layout(
        "aqua_motion_history",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::VERTEX_FRAGMENT,
            (
                texture_2d_array(TextureSampleType::Float { filterable: true }),
                sampler(SamplerBindingType::Filtering),
                uniform_buffer::<PreviousFrame>(false),
            ),
        ),
    );
    commands.insert_resource(History {
        texture,
        view,
        sampler: history_sampler,
        layout,
        group: None,
        uniform: None,
        previous: None,
        previous_epoch: None,
    });
}

fn prepare_history(
    frame: Res<Frame>,
    epoch: Res<MotionEpoch>,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    mut history: ResMut<History>,
) {
    let valid = history
        .previous
        .as_ref()
        .is_some_and(|saved| history_is_contiguous(frame.uniform(), saved))
        && history.previous_epoch == Some(epoch.0);
    let saved = history.previous.as_ref().unwrap_or_else(|| frame.uniform());
    let value = PreviousFrame {
        layout: saved.layout.clone(),
        time: saved.time,
        flow: saved.flow,
        valid: Vec4::new(if valid { 1.0 } else { 0.0 }, 0.0, 0.0, 0.0),
    };
    pass::write_uniform(&mut history.uniform, value, &device, &queue);
    if history.group.is_some() {
        return;
    }
    let Some(uniform) = history.uniform.as_ref() else {
        return;
    };
    history.group = Some(device.create_bind_group(
        "aqua_motion_history",
        &history.layout,
        &BindGroupEntries::sequential((&history.view, &history.sampler, uniform)),
    ));
}

fn main_motion_view(view: ViewQuery<(Option<&OceanView>, Option<&ViewPrepassTextures>)>) -> bool {
    let (ocean, prepass) = view.into_inner();
    ocean.is_some() && prepass.is_some_and(|textures| textures.motion_vectors.is_some())
}

fn capture_previous(
    mut context: RenderContext,
    frame: Res<Frame>,
    images: Res<RenderAssets<GpuImage>>,
    history: Res<History>,
) {
    let Some(source) = images.get(&frame.output()) else {
        return;
    };
    let recorder = context.diagnostic_recorder();
    let diagnostics = recorder.as_deref();
    let span = diagnostics.time_span(context.command_encoder(), "aqua_motion_history_copy");
    context.command_encoder().copy_texture_to_texture(
        TexelCopyTextureInfo {
            texture: &source.texture,
            mip_level: 0,
            origin: Origin3d::ZERO,
            aspect: TextureAspect::All,
        },
        TexelCopyTextureInfo {
            texture: &history.texture,
            mip_level: 0,
            origin: Origin3d::ZERO,
            aspect: TextureAspect::All,
        },
        HISTORY_EXTENT,
    );
    span.end(context.command_encoder());
}

fn commit_previous(
    frame: Res<Frame>,
    epoch: Res<MotionEpoch>,
    status: Res<AnimWavesStatus>,
    mut history: ResMut<History>,
) {
    if status.written {
        history.previous = Some(frame.uniform().clone());
        history.previous_epoch = Some(epoch.0);
    }
}

fn history_is_contiguous(current: &AnimWavesUniform, previous: &AnimWavesUniform) -> bool {
    let delta = current.time.x - previous.time.x;
    let finest_scale = current.layout.cascades[0].scale.max(1.0);
    let centre_delta =
        current.layout.center.truncate().truncate() - previous.layout.center.truncate().truncate();
    let detail_delta = (current.layout.center.z - previous.layout.center.z).abs();
    let bed_transform_delta = (current.layout.bed_transform - previous.layout.bed_transform)
        .abs()
        .max_element();
    let bed_range_delta = (current.layout.bed_range - previous.layout.bed_range)
        .abs()
        .max_element();
    let flow_delta = (current.flow - previous.flow).abs().max_element();

    delta.is_finite()
        && (-f32::EPSILON..=MAX_HISTORY_DELTA_SECONDS).contains(&delta)
        && centre_delta.is_finite()
        && centre_delta.length() <= finest_scale
        && detail_delta.is_finite()
        && detail_delta <= 1.0
        && bed_transform_delta.is_finite()
        && bed_transform_delta <= f32::EPSILON
        && bed_range_delta.is_finite()
        && bed_range_delta <= f32::EPSILON
        && flow_delta.is_finite()
        && flow_delta <= f32::EPSILON
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retained_displacement_is_exactly_two_and_a_half_mibibytes() {
        let texels = HISTORY_EXTENT.width as usize
            * HISTORY_EXTENT.height as usize
            * HISTORY_EXTENT.depth_or_array_layers as usize;
        let bytes = texels * 4 * size_of::<u16>();
        assert_eq!(bytes, 2_621_440);
        assert_eq!(bytes, 5 * 1024 * 1024 / 2);
    }

    fn uniform(time: f32, centre: Vec2) -> AnimWavesUniform {
        let mut cascades = [bevy_aqua_core::GpuCascade::default(); LOD_COUNT + 1];
        cascades[0].scale = 24.0;
        AnimWavesUniform {
            layout: GpuLayout {
                cascades,
                center: centre.extend(0.0).extend(LOD_COUNT as f32),
                bed_transform: Vec4::ZERO,
                bed_range: Vec4::ZERO,
            },
            waves: [bevy_aqua_core::GpuWave::default(); bevy_aqua_core::WAVE_SLOTS],
            ranges: [UVec4::ZERO; LOD_COUNT],
            time: Vec4::new(time, 0.0, 1.0, 0.0),
            flow: Vec4::ZERO,
        }
    }

    #[test]
    fn history_continuity_rejects_temporal_and_layout_discontinuities() {
        let previous = uniform(10.0, Vec2::ZERO);
        assert!(history_is_contiguous(
            &uniform(10.0 + 1.0 / 60.0, Vec2::new(1.0, 0.0)),
            &previous,
        ));
        assert!(history_is_contiguous(&uniform(10.0, Vec2::ZERO), &previous));
        assert!(!history_is_contiguous(&uniform(9.0, Vec2::ZERO), &previous));
        assert!(!history_is_contiguous(
            &uniform(10.251, Vec2::ZERO),
            &previous
        ));
        assert!(!history_is_contiguous(
            &uniform(10.0 + 1.0 / 60.0, Vec2::new(24.01, 0.0)),
            &previous,
        ));
    }

    #[test]
    fn history_continuity_rejects_changed_wave_or_bed_state() {
        let previous = uniform(10.0, Vec2::ZERO);
        let mut changed_flow = uniform(10.0 + 1.0 / 60.0, Vec2::ZERO);
        changed_flow.flow.x = 0.1;
        assert!(!history_is_contiguous(&changed_flow, &previous));

        let mut changed_bed = uniform(10.0 + 1.0 / 60.0, Vec2::ZERO);
        changed_bed.layout.bed_range.z = 1.0;
        assert!(!history_is_contiguous(&changed_bed, &previous));

        let mut changed_detail = uniform(10.0 + 1.0 / 60.0, Vec2::ZERO);
        changed_detail.layout.center.z = 1.01;
        assert!(!history_is_contiguous(&changed_detail, &previous));
    }
}

#[cfg(test)]
mod ocean_presence_tests {
    #[test]
    fn absent_ocean_vertices_keep_current_position_in_history() {
        let shader = include_str!("motion_prepass.wgsl");
        let current = shader
            .find("var previous_world_position = out.world_position;")
            .unwrap();
        let river = shader.find("if deformation.river {").unwrap();
        let ocean = shader
            .find("else if !deformation.bounded && field_params.info.y >= 0.5 {")
            .expect("previous ocean displacement requires an Ocean");
        assert!(current < river && river < ocean);
        assert!(!shader.contains("else if !deformation.bounded {"));
    }
}
