//! Native GPU regression; run explicitly with an external capture lock.

use super::*;
use bevy::{
    app::PluginsState,
    camera::{CameraPlugin, RenderTarget},
    core_pipeline::CorePipelinePlugin,
    light::LightPlugin,
    mesh::MeshPlugin,
    pbr::PbrPlugin,
    render::RenderPlugin,
    time::TimeUpdateStrategy,
    window::ExitCondition,
};
use bevy_aqua_core::{AnimWavesWritten, CascadeMaterial, OceanWaves, cascade};
use bevy_aqua_waves::AquaWavesPlugin;
use std::{
    collections::VecDeque,
    time::{Duration, Instant},
};

#[path = "gpu_protocol_tests.rs"]
mod protocol;

// The shader ABI is independently specified, not inferred from Rust packing.
const ROW_BYTES: usize = 48;
const RESULT_BYTES: usize = 256 * ROW_BYTES;
const POISON: u8 = 0xcd;

#[derive(Resource, Default)]
struct Observed(VecDeque<Vec<u8>>);

fn observe(event: On<ReadbackComplete>, mut observed: ResMut<Observed>) {
    assert_eq!(event.data.len(), RESULT_BYTES);
    if observed.0.len() == 16 {
        observed.0.pop_front();
    }
    observed.0.push_back(event.data.clone());
}

#[derive(Resource, Default)]
struct Fault {
    seed: Option<Vec<u8>>,
    no_waves: bool,
}

fn seed_results(
    mut fault: ResMut<Fault>,
    buffers: Res<Buffers>,
    gpu: Res<RenderAssets<GpuShaderBuffer>>,
    queue: Res<RenderQueue>,
) {
    if let Some(bytes) = fault.seed.take() {
        let buffer = gpu.get(&buffers.results).expect("result buffer prepared");
        queue.write_buffer(&buffer.buffer, 0, &bytes);
    }
}

fn suppress_waves(fault: Res<Fault>, mut status: ResMut<bevy_aqua_core::AnimWavesStatus>) {
    if fault.no_waves {
        status.written = false;
    }
}

struct Rig {
    app: App,
    start: Instant,
    frames: usize,
    readback: Entity,
}

impl Rig {
    fn new() -> Self {
        let start = Instant::now();
        let mut app = App::new();
        // Explicit plugins avoid a window, a winit event loop, and a pipelined
        // render thread, even when workspace feature unification enables them.
        app.add_plugins((
            MinimalPlugins,
            TransformPlugin,
            WindowPlugin {
                primary_window: None,
                exit_condition: ExitCondition::DontExit,
                ..default()
            },
            AssetPlugin::default(),
            RenderPlugin::default(),
            ImagePlugin::default(),
            MeshPlugin,
            CameraPlugin,
            LightPlugin,
            CorePipelinePlugin,
            PbrPlugin::default(),
        ));
        app.insert_resource(TimeUpdateStrategy::ManualDuration(Duration::ZERO));
        app.init_asset::<CascadeMaterial>();
        app.init_resource::<ResolvedWaterBodies>();
        app.init_resource::<Observed>();
        app.insert_resource(Ocean::default());
        app.insert_resource(OceanWaves::default());
        let mut images = app.world_mut().resource_mut::<Assets<Image>>();
        let texture = images.add(cascade::make_texture());
        let surface = images.add(cascade::make_fft_surface_texture());
        let target = images.add(Image::new_target_texture(
            32,
            32,
            TextureFormat::Rgba8UnormSrgb,
            None,
        ));
        app.insert_resource(Data::new(
            Handle::default(),
            texture,
            surface,
            cascade::GpuLayout::new(&cascade::layout(Vec2::ZERO), Vec2::ZERO, 0.0),
        ));
        bevy_aqua_core::add_shader(&mut app);
        bevy_aqua_core::bed::add(&mut app);
        app.add_plugins((
            ExtractResourcePlugin::<Data>::default(),
            AquaWavesPlugin,
            AquaQueryPlugin,
        ));
        app.world_mut().spawn((
            Camera3d::default(),
            RenderTarget::Image(target.into()),
            Transform::from_xyz(0.0, 10.0, 10.0).looking_at(Vec3::ZERO, Vec3::Y),
            Msaa::Off,
            OceanView,
        ));
        app.sub_app_mut(RenderApp)
            .init_resource::<Fault>()
            .add_systems(
                Render,
                seed_results
                    .after(RenderSystems::PrepareBindGroups)
                    .before(RenderSystems::Render),
            )
            .add_systems(
                Core3d,
                suppress_waves
                    .after(AnimWavesWritten)
                    .before(dispatch_wave_query),
            );
        while app.plugins_state() != PluginsState::Ready {
            assert!(
                start.elapsed() < Duration::from_secs(30),
                "renderer startup timed out"
            );
            std::thread::yield_now();
        }
        app.finish();
        app.cleanup();
        app.update();
        let adapter = app
            .sub_app(RenderApp)
            .world()
            .resource::<bevy::render::renderer::RenderAdapterInfo>();
        eprintln!(
            "GPU adapter: {} ({:?}, {:?})",
            adapter.name, adapter.backend, adapter.device_type
        );
        let buffers = app.world().resource::<Buffers>().clone();
        let readback = app
            .world_mut()
            .query::<(Entity, &Readback)>()
            .iter(app.world())
            .find_map(|(entity, readback)| match readback {
                Readback::Buffer { buffer, .. } if *buffer == buffers.results => Some(entity),
                _ => None,
            })
            .expect("AquaQueryPlugin created its readback entity");
        app.world_mut().entity_mut(readback).observe(observe);
        Self {
            app,
            start,
            frames: 1,
            readback,
        }
    }

    fn render(&self) -> &World {
        self.app.sub_app(RenderApp).world()
    }

    fn render_mut(&mut self) -> &mut World {
        self.app.sub_app_mut(RenderApp).world_mut()
    }

    fn step(&mut self) {
        assert!(
            self.frames < 4096 && self.start.elapsed() < Duration::from_secs(180),
            "GPU fixture deadline exceeded at frame {}",
            self.frames
        );
        self.app.update();
        self.frames += 1;
        // Complete submitted work, but deliver it only through Bevy's real
        // async ReadbackComplete observer on the following extraction.
        if let Some(device) = self.render().get_resource::<RenderDevice>() {
            device
                .poll(PollType::Wait {
                    submission_index: None,
                    timeout: Some(Duration::from_secs(5)),
                })
                .expect("GPU completion within five seconds");
        }
    }

    fn spawn(&mut self, count: usize) -> Vec<Entity> {
        (0..count)
            .map(|index| {
                self.app
                    .world_mut()
                    .spawn((
                        WaveQuery,
                        Transform::from_xyz(index as f32 * 0.37, 0.0, index as f32 * -0.19),
                    ))
                    .id()
            })
            .collect()
    }

    fn warm(&mut self, probes: &[Entity]) {
        for _ in 0..1024 {
            self.step();
            if probes.iter().all(|entity| self.sample(*entity).valid) {
                assert!(
                    self.render()
                        .resource::<bevy_aqua_core::AnimWavesStatus>()
                        .written
                );
                eprintln!("GPU production waves/query ready: frame {}", self.frames);
                return;
            }
        }
        panic!("no production query completion after 1024 frames");
    }

    fn sample(&self, entity: Entity) -> WaveSurface {
        *self.app.world().get::<WaveSurface>(entity).unwrap()
    }

    fn position(&mut self, entity: Entity, position: Vec3) {
        self.app
            .world_mut()
            .get_mut::<Transform>(entity)
            .unwrap()
            .translation = position;
    }

    fn clear_observed(&mut self) {
        self.app.world_mut().resource_mut::<Observed>().0.clear();
    }

    fn snapshot(&mut self, label: &str, matches: impl Fn(&[u8]) -> bool) -> Vec<u8> {
        for _ in 0..512 {
            self.step();
            let found = self
                .app
                .world()
                .resource::<Observed>()
                .0
                .iter()
                .find(|bytes| matches(bytes))
                .cloned();
            if let Some(bytes) = found {
                eprintln!(
                    "GPU checkpoint {label}: frame {}, {} bytes",
                    self.frames,
                    bytes.len()
                );
                if let Some(directory) = std::env::var_os("AQUA_QUERY_GPU_ARTIFACTS") {
                    let directory = std::path::PathBuf::from(directory);
                    assert!(directory.is_absolute(), "artifact path must be absolute");
                    std::fs::create_dir_all(&directory).unwrap();
                    std::fs::write(directory.join(format!("{label}.bin")), &bytes).unwrap();
                }
                return bytes;
            }
        }
        panic!("no matching real readback for {label} within 512 frames");
    }

    fn seed(&mut self, bytes: Vec<u8>) {
        assert_eq!(bytes.len(), RESULT_BYTES);
        self.clear_observed();
        self.render_mut().resource_mut::<Fault>().seed = Some(bytes);
        self.step();
        // Discard completions copied before the seed was submitted.
        self.clear_observed();
    }
}

#[test]
#[ignore = "requires a native GPU; use flock /tmp/apophany-capture.lock and timeout"]
fn gpu_query_protocol() {
    let mut rig = Rig::new();
    let probes = rig.spawn(65);
    rig.warm(&probes);
    // Keep slot 64, whose old row lies inside the first workgroup. The old
    // shader rewrites rows 1..63 after this shrink, including that duplicate.
    let survivor = probes[63];
    for entity in probes.into_iter().filter(|entity| *entity != survivor) {
        rig.app.world_mut().despawn(entity);
    }
    rig.seed(vec![POISON; RESULT_BYTES]);
    let shrink = rig.snapshot("shrink-to-one", |bytes| {
        bytes[..ROW_BYTES] != [POISON; ROW_BYTES]
            && bytes[128 * ROW_BYTES..].iter().all(|byte| *byte == POISON)
    });
    assert!(
        shrink[ROW_BYTES..].iter().all(|byte| *byte == POISON),
        "inactive result rows were rewritten after 65 -> 1 queries"
    );

    let retained = rig.sample(survivor);
    rig.render_mut().resource_mut::<Fault>().no_waves = true;
    rig.seed(vec![POISON; RESULT_BYTES]);
    let skipped = rig.snapshot("waves-not-written", |bytes| {
        bytes[ROW_BYTES..].iter().all(|byte| *byte == POISON)
    });
    assert_eq!(
        skipped,
        vec![POISON; RESULT_BYTES],
        "query wrote results while AnimWavesStatus.written was false"
    );
    assert_eq!(rig.sample(survivor), retained);
    rig.render_mut().resource_mut::<Fault>().no_waves = false;
    rig.clear_observed();
    rig.snapshot("waves-recovered", |bytes| {
        bytes[0..ROW_BYTES] != [POISON; ROW_BYTES]
    });

    protocol::verify(&mut rig, survivor);

    rig.app
        .world_mut()
        .entity_mut(survivor)
        .remove::<WaveQuery>();
    rig.seed(vec![POISON; RESULT_BYTES]);
    let empty = rig.snapshot("zero-queries", |bytes| {
        bytes.iter().all(|byte| *byte == POISON)
    });
    assert_eq!(empty, vec![POISON; RESULT_BYTES]);
    assert_eq!(rig.render().resource::<Batch>().count, 0);
    eprintln!(
        "GPU query protocol passed in {} frames, {:?}",
        rig.frames,
        rig.start.elapsed()
    );
}
