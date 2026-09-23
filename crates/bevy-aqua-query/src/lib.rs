//! GPU-sampled wave queries for gameplay such as buoyancy.

#![warn(unreachable_pub)]
//!
//! Insert [`WaveQuery`] on any entity with a transform to receive the rendered
//! water displacement and surface normal at its origin in [`WaveSurface`].
//! Samples use the AnimWaves cascades and detail-LOD clamps; mesh morphing
//! and shading normals can differ from this gameplay sample. Results arrive
//! asynchronously, usually with roughly one frame of latency. A skipped wave
//! or query pass retains the last result rather than publishing a new sample.
//!
//! Every probe — ocean, bounded pond, or river body — goes through the one
//! compute dispatch and readback: extraction resolves each probe against
//! the registered bodies (`bevy-aqua-sdf`) into a request carrying its surface
//! level and, inside rivers, the baked current plus channel geometry. The
//! shader synthesizes river waves in closed form (the shared
//! `bevy_aqua_core::river` module) instead of sampling the cascades there,
//! mirroring the material's vertex path.
//!
//! Sample points beyond the coarsest cascade ring return zero displacement,
//! matching the horizon fade of the rendered waves. At most `MAX_QUERIES`
//! sample points are submitted per frame; over-budget entities keep their
//! previous sample.

use std::collections::HashMap;

use bevy::{
    asset::{RenderAssetUsages, embedded_asset},
    core_pipeline::{Core3dSystems, schedule::Core3d},
    prelude::*,
    render::{
        MainWorld, Render, RenderApp, RenderStartup, RenderSystems,
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        gpu_readback::{Readback, ReadbackComplete},
        render_asset::RenderAssets,
        render_resource::{
            ShaderType,
            binding_types::{
                sampler, storage_buffer, storage_buffer_read_only, texture_2d_array, uniform_buffer,
            },
            *,
        },
        renderer::{RenderContext, RenderDevice, RenderQueue, ViewQuery},
        storage::{GpuShaderBuffer, ShaderBuffer},
        texture::GpuImage,
    },
};

use bevy_aqua_core::{
    AnimWavesStatus, AnimWavesUniformSlot, Data, Ocean, OceanView, ResolvedWaterBodies,
    ResolvedWaterBody, pass,
};
use bevy_aqua_sdf::FlowSample;

const SHADER_PATH: &str = "embedded://bevy_aqua_query/wave_query.wgsl";
pub(crate) const MAX_QUERIES: u32 = 256;
const WORKGROUP_SIZE: u32 = 64;
// Serial comparisons require fewer than 2^31 batches between a result and
// its submission/re-entry/last-applied barrier (about 207 days at 120 Hz).
const GENERATION_HALF_RANGE: u32 = 1 << 31;

/// Marks an entity for per-frame water sampling at its world origin.
///
/// The entity needs a transform; child entities inherit their parent and make
/// multi-point hulls straightforward. Inserting this component automatically
/// inserts [`WaveSurface`].
///
/// # Examples
///
/// ```
/// use bevy::prelude::*;
/// use bevy_aqua_core::Ocean;
/// use bevy_aqua_query::{AquaQueryPlugin, WaveQuery};
///
/// # fn spawn_buoy(mut commands: Commands, mut meshes: ResMut<Assets<Mesh>>) {
///     commands.spawn((
///         WaveQuery,
///         Mesh3d(meshes.add(Sphere::new(0.5))),
///         Transform::from_xyz(4.0, 0.0, -2.0),
///     ));
/// }
/// ```
#[derive(Component, Debug, Default, Clone, Copy, PartialEq)]
#[require(WaveSurface)]
pub struct WaveQuery;

/// The sampled water surface at a [`WaveQuery`] entity's origin.
///
/// Updated when a fresh GPU result arrives. Until the first result, or while
/// no surface contains the query, [`WaveSurface::valid`] is false. Skipped
/// dispatches retain the previous sample.
#[derive(Component, Debug, Clone, Copy, PartialEq)]
pub struct WaveSurface {
    /// World-space wave displacement relative to the owning surface's mean
    /// plane. Add this to a query transform placed at that plane's level.
    pub displacement: Vec3,
    /// Unit surface normal at the sample point.
    pub normal: Vec3,
    /// Whether a sample has arrived since this probe last entered water.
    pub valid: bool,
    /// Instantaneous breaking-crest source in `[0, 1]`, derived from the
    /// same horizontal-displacement compression that feeds persistent foam.
    pub crest: f32,
}

impl Default for WaveSurface {
    fn default() -> Self {
        Self {
            displacement: Vec3::ZERO,
            normal: Vec3::Y,
            valid: false,
            crest: 0.0,
        }
    }
}

#[derive(ShaderType, Clone, Copy, Debug, Default)]
struct QueryRequest {
    world_xz: Vec2,
    slot: u32,
    // 0: ocean cascade, 1: flat bounded body, 2: analytic river.
    kind: u32,
    // River synthesis inputs: xy local current (m/s), z signed bank
    // margin (m), w channel half width (m). Positive width selects the
    // river analytic path even when authored current is zero.
    flow: Vec4,
}

#[derive(ShaderType, Clone, Copy, Debug, Default)]
struct QueryResult {
    displacement: Vec4,
    normal_crest: Vec4,
    // x: integer slot, y: batch generation, z: validity, w: reserved.
    metadata: UVec4,
}

#[derive(ShaderType, Clone, Copy, Debug, Default)]
struct QueryBatch {
    count: u32,
    generation: u32,
    reserved: UVec2,
}

#[derive(Debug)]
struct RegisteredProbe {
    entity: Entity,
    accepting: bool,
    first_generation: u32,
    latest_submitted: u32,
    latest_applied: Option<u32>,
}

#[derive(Resource, Debug, Default)]
struct Registry {
    next_slot: u32,
    generation: u32,
    slots: HashMap<Entity, u32>,
    probes: HashMap<u32, RegisteredProbe>,
}

impl Registry {
    fn assign(&mut self, entity: Entity) -> Option<u32> {
        if self.slots.contains_key(&entity) {
            return None;
        }
        // Do not recycle identities while an old readback may still exist.
        let slot = self.next_slot.checked_add(1)?;
        self.next_slot = slot;
        self.slots.insert(entity, slot);
        self.probes.insert(
            slot,
            RegisteredProbe {
                entity,
                accepting: false,
                first_generation: 0,
                latest_submitted: 0,
                latest_applied: None,
            },
        );
        Some(slot)
    }

    fn reclaim(&mut self, entity: Entity) {
        if let Some(slot) = self.slots.remove(&entity) {
            self.probes.remove(&slot);
        }
    }

    fn next_generation(&mut self) -> u32 {
        self.generation = self.generation.wrapping_add(1);
        self.generation
    }

    // Records extraction, not GPU completion. Only the shader echoes a
    // generation into results, and only after all dispatch gates pass.
    fn submit(&mut self, slot: u32, generation: u32) {
        let Some(probe) = self.probes.get_mut(&slot) else {
            return;
        };
        if !probe.accepting {
            probe.accepting = true;
            probe.first_generation = generation;
            probe.latest_applied = None;
        }
        probe.latest_submitted = generation;
    }

    fn invalidate(&mut self, entity: Entity) {
        let Some(slot) = self.slots.get(&entity) else {
            return;
        };
        if let Some(probe) = self.probes.get_mut(slot) {
            probe.accepting = false;
        }
    }

    fn accept(&mut self, slot: u32, generation: u32) -> Option<Entity> {
        let probe = self.probes.get_mut(&slot)?;
        let fresh = match probe.latest_applied {
            Some(applied) => generation_after(generation, applied),
            None => {
                generation == probe.first_generation
                    || generation_after(generation, probe.first_generation)
            }
        };
        let submitted = generation == probe.latest_submitted
            || generation_after(probe.latest_submitted, generation);
        if !probe.accepting || !fresh || !submitted {
            return None;
        }
        probe.latest_applied = Some(generation);
        Some(probe.entity)
    }
}

fn generation_after(candidate: u32, reference: u32) -> bool {
    let distance = candidate.wrapping_sub(reference);
    distance != 0 && distance < GENERATION_HALF_RANGE
}

#[derive(Resource, Clone, ExtractResource)]
struct Buffers {
    requests: Handle<ShaderBuffer>,
    results: Handle<ShaderBuffer>,
}

#[derive(Resource, Default)]
struct Batch {
    bytes: Vec<u8>,
    count: usize,
    generation: u32,
}

/// Registers extraction, GPU sampling, and async readback for [`WaveQuery`].
#[derive(Debug, Default, Clone, Copy)]
pub struct AquaQueryPlugin;

impl Plugin for AquaQueryPlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "wave_query.wgsl");
        app.init_resource::<Registry>()
            .add_systems(Startup, init_buffers)
            .add_systems(Update, (reclaim_slots, assign_slots).chain())
            .add_plugins(ExtractResourcePlugin::<Buffers>::default());
        if let Some(render) = app.get_sub_app_mut(RenderApp) {
            render
                .add_systems(RenderStartup, init_pipeline)
                .add_systems(ExtractSchedule, extract_wave_queries)
                .add_systems(
                    Render,
                    prepare_bind_groups.in_set(RenderSystems::PrepareBindGroups),
                )
                .add_systems(
                    Core3d,
                    dispatch_wave_query
                        .after(bevy_aqua_core::AnimWavesWritten)
                        .before(Core3dSystems::MainPass),
                );
        }
    }
}

fn init_buffers(mut commands: Commands, mut buffers: ResMut<Assets<ShaderBuffer>>) {
    let requests = buffers.add(ShaderBuffer::with_size(
        MAX_QUERIES as usize * QueryRequest::SHADER_SIZE.get() as usize,
        RenderAssetUsages::default(),
    ));
    let results = buffers.add(ShaderBuffer::with_size(
        MAX_QUERIES as usize * QueryResult::SHADER_SIZE.get() as usize,
        RenderAssetUsages::default(),
    ));
    commands.insert_resource(Buffers {
        requests: requests.clone(),
        results: results.clone(),
    });
    commands
        .spawn(Readback::buffer(results))
        .observe(apply_results);
}

fn assign_slots(
    mut spawned: Query<(Entity, &mut WaveSurface), Added<WaveQuery>>,
    mut registry: ResMut<Registry>,
) {
    for (entity, mut surface) in &mut spawned {
        registry.assign(entity);
        *surface = WaveSurface::default();
    }
}

fn reclaim_slots(mut removed: RemovedComponents<WaveQuery>, mut registry: ResMut<Registry>) {
    for entity in removed.read() {
        registry.reclaim(entity);
    }
}

fn pack_requests(submissions: &[(u32, Vec2, u32, Vec4)]) -> (Vec<u8>, usize) {
    let count = submissions.len().min(MAX_QUERIES as usize);
    if count == 0 {
        return (Vec::new(), 0);
    }
    let mut requests = Vec::with_capacity(count);
    for &(slot, world_xz, kind, flow) in submissions.iter().take(count) {
        requests.push(QueryRequest {
            world_xz,
            slot,
            kind,
            flow,
        });
    }
    let mut bytes = Vec::with_capacity(count * QueryRequest::SHADER_SIZE.get() as usize);
    let mut wrapper = bevy::render::render_resource::encase::StorageBuffer::new(&mut bytes);
    wrapper.write(&requests).expect("packed requests write");
    (bytes, count)
}

fn decode_results(data: &[u8], registry: &mut Registry) -> Vec<(Entity, Vec3, Vec3, f32)> {
    let wrapper = bevy::render::render_resource::encase::StorageBuffer::new(data);
    let results: Vec<QueryResult> = wrapper
        .create()
        .expect("wave-query result buffer must match QueryResult");
    let mut samples = Vec::new();
    for result in results.into_iter().take(MAX_QUERIES as usize) {
        if result.metadata.z != 1 {
            continue;
        }
        let Some(entity) = registry.accept(result.metadata.x, result.metadata.y) else {
            continue;
        };
        samples.push((
            entity,
            result.displacement.truncate(),
            result.normal_crest.truncate(),
            result.normal_crest.w.clamp(0.0, 1.0),
        ));
    }
    samples
}

fn apply_results(
    event: On<ReadbackComplete>,
    mut registry: ResMut<Registry>,
    mut surfaces: Query<&mut WaveSurface, With<WaveQuery>>,
) {
    let merged = decode_results(&event.data, &mut registry);
    for (entity, displacement, normal, crest) in merged {
        if let Ok(mut surface) = surfaces.get_mut(entity) {
            surface.displacement = displacement;
            surface.normal = normal.normalize_or_zero();
            surface.valid = true;
            surface.crest = crest;
        }
    }
}

#[derive(Debug)]
enum ProbeResolution {
    Ocean,
    Body,
    River { flowed: FlowSample },
}

fn probe_resolution(
    bodies: &[ResolvedWaterBody],
    ocean_level: Option<f32>,
    world_xz: Vec2,
) -> Option<ProbeResolution> {
    for body in bodies {
        if let Some(flowed) = body.flow_at(world_xz)
            && flowed.margin >= 0.0
        {
            return Some(ProbeResolution::River { flowed });
        }
        if body.contains(world_xz) {
            return Some(ProbeResolution::Body);
        }
    }
    ocean_level.map(|_| ProbeResolution::Ocean)
}

fn extract_wave_queries(mut main_world: ResMut<MainWorld>, mut commands: Commands) {
    let bodies = main_world.resource::<ResolvedWaterBodies>().0.clone();
    let ocean_level = main_world.get_resource::<Ocean>().map(|ocean| ocean.level);
    let slots = main_world.resource::<Registry>().slots.clone();
    let generation = main_world.resource_mut::<Registry>().next_generation();
    let mut submissions: Vec<(u32, Vec2, u32, Vec4)> = Vec::new();
    let mut invalid = Vec::new();
    let mut probes = main_world.query::<(Entity, Option<&GlobalTransform>, &WaveQuery)>();
    for (entity, transform, _) in probes.iter(&main_world) {
        let Some(slot) = slots.get(&entity) else {
            continue;
        };
        let Some(world_xz) = transform
            .map(|transform| transform.translation().xz())
            .filter(|position| position.is_finite())
        else {
            invalid.push(entity);
            continue;
        };
        let Some(resolution) = probe_resolution(&bodies, ocean_level, world_xz) else {
            invalid.push(entity);
            continue;
        };
        // Over-budget probes still in water keep their previous result.
        // Exits and invalid positions must close their old readback epoch.
        if submissions.len() >= MAX_QUERIES as usize {
            continue;
        }
        match resolution {
            ProbeResolution::River { flowed, .. } => {
                submissions.push((
                    *slot,
                    world_xz,
                    2,
                    Vec4::new(
                        flowed.flow.x,
                        flowed.flow.y,
                        flowed.margin,
                        flowed.half_width,
                    ),
                ));
            }
            ProbeResolution::Body => {
                submissions.push((*slot, world_xz, 1, Vec4::ZERO));
            }
            ProbeResolution::Ocean => {
                submissions.push((*slot, world_xz, 0, Vec4::ZERO));
            }
        }
    }
    // Sort only the admitted prefix. Admission follows ECS iteration and
    // can change after archetype moves; over-budget probes keep their sample.
    submissions.sort_by_key(|(slot, ..)| *slot);
    {
        let mut registry = main_world.resource_mut::<Registry>();
        for &(slot, ..) in &submissions {
            registry.submit(slot, generation);
        }
        for &entity in &invalid {
            registry.invalidate(entity);
        }
    }
    for entity in invalid {
        if let Some(mut surface) = main_world.get_mut::<WaveSurface>(entity) {
            *surface = WaveSurface::default();
        }
    }
    let (bytes, count) = pack_requests(&submissions);
    commands.insert_resource(Batch {
        bytes,
        count,
        generation,
    });
}

const QUERY: &str = "Wave query";
const SAMPLE: &str = "sample";

fn pass_table() -> Vec<pass::PassSpec> {
    vec![pass::PassSpec {
        key: QUERY,
        shader: pass::ShaderSource::Path(SHADER_PATH),
        entry_points: &[SAMPLE],
        shader_defs: &[],
        wgsl_entry: None,
        layout: BindGroupLayoutDescriptor::new(
            QUERY,
            &BindGroupLayoutEntries::sequential(
                ShaderStages::COMPUTE,
                (
                    texture_2d_array(TextureSampleType::Float { filterable: true }),
                    sampler(SamplerBindingType::Filtering),
                    uniform_buffer::<bevy_aqua_core::AnimWavesUniform>(false),
                    uniform_buffer::<QueryBatch>(false),
                    storage_buffer_read_only::<QueryRequest>(false),
                    storage_buffer::<QueryResult>(false),
                ),
            ),
        ),
    }]
}

// Asset handles can survive replacement of their underlying GPU objects.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct BindingKey {
    texture: TextureViewId,
    sampler: SamplerId,
    waves: BufferId,
    batch: BufferId,
    requests: BufferId,
    results: BufferId,
}

#[derive(Resource)]
struct Prepared {
    passes: pass::Passes,
    sampler: Sampler,
    batch: Option<UniformBuffer<QueryBatch>>,
    groups: pass::Groups,
    bound: Option<BindingKey>,
    ready: bool,
}

fn init_pipeline(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    cache: Res<PipelineCache>,
    device: Res<RenderDevice>,
) {
    let sampler = device.create_sampler(&SamplerDescriptor {
        label: Some("aqua wave query"),
        ..default()
    });
    commands.insert_resource(Prepared {
        passes: pass::Passes::new(&asset_server, &cache, pass_table()),
        sampler,
        batch: None,
        groups: pass::Groups::default(),
        bound: None,
        ready: false,
    });
}

fn prepare_bind_groups(
    resources: (
        Res<Data>,
        Res<AnimWavesUniformSlot>,
        Option<Res<Buffers>>,
        Res<Batch>,
    ),
    assets: (
        Res<RenderAssets<GpuImage>>,
        Res<RenderAssets<GpuShaderBuffer>>,
    ),
    device: (Res<RenderDevice>, Res<RenderQueue>, Res<PipelineCache>),
    mut prepared: ResMut<Prepared>,
) {
    let (data, slot, buffers, batch) = resources;
    prepared.ready = false;
    pass::write_uniform(
        &mut prepared.batch,
        QueryBatch {
            count: batch.count as u32,
            generation: batch.generation,
            reserved: UVec2::ZERO,
        },
        &device.0,
        &device.1,
    );
    let (images, ssbos) = assets;
    let anim_waves = data.texture();
    let (Some(buffers), Some(output), Some(uniform), Some(batch_uniform)) = (
        buffers.as_ref(),
        images.get(&anim_waves),
        slot.0.as_ref(),
        prepared.batch.as_ref(),
    ) else {
        return;
    };
    let (Some(requests), Some(results)) =
        (ssbos.get(&buffers.requests), ssbos.get(&buffers.results))
    else {
        return;
    };
    let (Some(wave_buffer), Some(batch_buffer)) = (uniform.buffer(), batch_uniform.buffer()) else {
        return;
    };
    let key = BindingKey {
        texture: output.texture_view.id(),
        sampler: prepared.sampler.id(),
        waves: wave_buffer.id(),
        batch: batch_buffer.id(),
        requests: requests.buffer.id(),
        results: results.buffer.id(),
    };
    if prepared.bound == Some(key) {
        prepared.ready = true;
        return;
    }
    let group = pass::bind_group(
        &device.0,
        &device.2,
        &prepared.passes,
        QUERY,
        QUERY,
        &BindGroupEntries::sequential((
            &output.texture_view,
            &prepared.sampler,
            uniform,
            batch_uniform,
            BufferBinding {
                buffer: &requests.buffer,
                offset: 0,
                size: None,
            },
            BufferBinding {
                buffer: &results.buffer,
                offset: 0,
                size: None,
            },
        )),
    );
    prepared.groups.register("query", group);
    prepared.bound = Some(key);
    prepared.ready = true;
}

fn dispatch_wave_query(
    view: ViewQuery<Option<&OceanView>>,
    resources: (
        Res<Batch>,
        Res<Prepared>,
        Option<Res<Buffers>>,
        Res<AnimWavesStatus>,
    ),
    assets: (Res<RenderAssets<GpuShaderBuffer>>, Res<RenderQueue>),
    cache: Res<PipelineCache>,
    mut context: RenderContext,
) {
    if view.into_inner().is_none() {
        return;
    }
    let (batch, prepared, buffers, waves) = resources;
    let (ssbos, queue) = assets;
    if batch.count == 0 || !waves.written || !prepared.ready {
        return;
    }
    let Some(group) = prepared.groups.get("query") else {
        return;
    };
    let Some(ready) = prepared.passes.ready_all(&cache, &[(QUERY, SAMPLE)]) else {
        return;
    };
    let Some(buffers) = buffers.as_ref() else {
        return;
    };
    let Some(gpu_requests) = ssbos.get(&buffers.requests) else {
        return;
    };
    // Queue writes land before this frame's encoder submission, so the
    // dispatch always samples the positions extracted this frame.
    queue.write_buffer(&gpu_requests.buffer, 0, &batch.bytes);
    let steps = vec![pass::Step::Dispatch {
        pipeline: ready.get(QUERY, SAMPLE),
        group,
        workgroups: [batch.count.div_ceil(WORKGROUP_SIZE as usize) as u32, 1, 1],
    }];
    pass::run_spans(&mut context, &[pass::Span::new("aqua_wave_query", steps)]);
}

#[cfg(test)]
#[path = "query_tests.rs"]
mod tests;

#[cfg(all(test, not(target_arch = "wasm32")))]
#[path = "gpu_tests.rs"]
mod gpu_tests;
