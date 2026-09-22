//! Real wave -> foam GPU lifecycle test. No public fixture or substitute WGSL.

use super::*;
use crate::AquaFoamPlugin;
use bevy::{ecs::system::RunSystemOnce, time::TimeUpdateStrategy};
use bevy_aqua_core::{AnimWavesStatus, AnimWavesWritten, BedHeightMap};
use bevy_aqua_waves::AquaWavesPlugin;
use std::time::{Duration, Instant};

#[path = "gpu_fixture.rs"]
mod fixture;

#[derive(Resource, Default)]
struct Fault {
    stale_groups: Option<pass::Groups>,
    no_waves: bool,
    cold: bool,
    withhold_bed: bool,
    held_bed: Option<(AssetId<Image>, GpuImage)>,
}

fn stale_groups(mut fault: ResMut<Fault>, mut prepared: ResMut<Prepared>) {
    if let Some(groups) = fault.stale_groups.take() {
        prepared.groups = groups;
    }
}

fn withhold_bed(world: &mut World) {
    if !std::mem::take(&mut world.resource_mut::<Fault>().withhold_bed) {
        return;
    }
    let id = world.resource::<BedHeightMap>().image.id();
    let image = world
        .resource_mut::<RenderAssets<GpuImage>>()
        .remove(id)
        .unwrap();
    world.resource_mut::<Fault>().held_bed = Some((id, image));
    // The producer's groups already hold valid texture references. Re-run only
    // the real foam preparation with this input unavailable. Thus written=true
    // cannot mask a broken foam readiness gate in the downstream writer.
    world.run_system_once(prepare_bind_groups).unwrap();
}

fn skip_faults(
    mut fault: ResMut<Fault>,
    mut prepared: ResMut<Prepared>,
    mut waves: ResMut<AnimWavesStatus>,
    server: Res<AssetServer>,
    cache: Res<PipelineCache>,
) {
    if fault.no_waves {
        waves.written = false;
    }
    if std::mem::take(&mut fault.cold) {
        // Queue real production pipelines after cache processing, not a dummy
        // pipeline that could pass while the actual shader remains broken.
        prepared.passes = pass::Passes::new(&server, &cache, pass_table());
        assert!(
            prepared
                .passes
                .ready_all(
                    &cache,
                    &[
                        (UPDATE, PREVIOUS),
                        (UPDATE, PREVIOUS_ZERO),
                        (UPDATE, CURRENT)
                    ]
                )
                .is_none()
        );
    }
}

#[derive(Debug, PartialEq)]
struct History {
    state_is_a: bool,
    completed_tick: u32,
    layout: Option<Vec<u8>>,
}

fn layout_bytes(layout: &bevy_aqua_core::GpuLayout) -> Vec<u8> {
    let mut bytes = Vec::<u8>::new();
    encase::UniformBuffer::new(&mut bytes)
        .write(layout)
        .unwrap();
    bytes
}

#[derive(PartialEq)]
struct Published {
    surface: Vec<u8>,
    a: Vec<u8>,
    b: Vec<u8>,
}

struct Rig {
    app: App,
    a: BedHeightMap,
    b: BedHeightMap,
    frames: usize,
    start: Instant,
}

impl Rig {
    fn new() -> Self {
        let mut app = fixture::app();
        let (a, b) = fixture::beds(&mut app);
        app.insert_resource(a.clone());
        app.add_plugins((AquaWavesPlugin, AquaFoamPlugin));
        app.sub_app_mut(RenderApp)
            .init_resource::<Fault>()
            .add_systems(
                Render,
                stale_groups
                    .after(prepare_bind_groups)
                    .in_set(RenderSystems::PrepareBindGroups),
            )
            .add_systems(
                Render,
                withhold_bed
                    .after(RenderSystems::PrepareBindGroups)
                    .before(RenderSystems::Render),
            )
            .add_systems(
                Core3d,
                skip_faults.after(AnimWavesWritten).before(write_foam),
            );
        fixture::finish(&mut app);
        Self {
            app,
            a,
            b,
            frames: 0,
            start: Instant::now(),
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
            self.frames < 4096 && self.start.elapsed() < Duration::from_secs(240),
            "GPU fixture deadline"
        );
        fixture::step(&mut self.app);
        self.frames += 1;
    }

    fn tick(&mut self) {
        // Just above 1/30 s so nanosecond representation does not lose a tick.
        self.app
            .insert_resource(TimeUpdateStrategy::ManualDuration(Duration::from_nanos(
                33_333_334,
            )));
        self.step();
        self.app
            .insert_resource(TimeUpdateStrategy::ManualDuration(Duration::ZERO));
    }

    fn warm(&mut self) {
        for _ in 0..1024 {
            self.step();
            let prepared = self.render().resource::<Prepared>();
            let frame = self.render().resource::<Frame>();
            if self.render().resource::<AnimWavesStatus>().written
                && prepared.ready
                && prepared.state_layout.is_some()
                && prepared.completed_tick == frame.uniform.step.x
                && prepared
                    .passes
                    .ready_all(
                        self.render().resource::<PipelineCache>(),
                        &[
                            (UPDATE, PREVIOUS),
                            (UPDATE, PREVIOUS_ZERO),
                            (UPDATE, CURRENT),
                        ],
                    )
                    .is_some()
            {
                self.current();
                return;
            }
        }
        panic!("production wave/foam pipelines/assets did not become ready");
    }

    fn history(&self) -> History {
        let prepared = self.render().resource::<Prepared>();
        History {
            state_is_a: prepared.state_is_a,
            completed_tick: prepared.completed_tick,
            layout: prepared.state_layout.as_ref().map(layout_bytes),
        }
    }

    fn current(&self) {
        let prepared = self.render().resource::<Prepared>();
        let frame = self.render().resource::<Frame>();
        assert!(prepared.ready && prepared.groups.created());
        assert!(self.render().resource::<AnimWavesStatus>().written);
        assert_eq!(prepared.completed_tick, frame.uniform.step.x);
        assert_eq!(
            self.history().layout,
            Some(layout_bytes(&frame.uniform.target_layout))
        );
        let selected = self.render().get_resource::<BedHeightMap>().map_or_else(
            || self.render().resource::<bed::GpuFallback>().0.id(),
            |bed| bed.image.id(),
        );
        let view = self
            .render()
            .resource::<RenderAssets<GpuImage>>()
            .get(selected)
            .unwrap()
            .texture_view
            .id();
        assert_eq!(prepared.bed_binding, Some((selected, view)));
        assert_eq!(
            prepared
                .uniform
                .as_ref()
                .unwrap()
                .get()
                .target_layout
                .bed_range,
            frame.uniform.target_layout.bed_range
        );
    }

    fn snapshot(&self, label: &str) -> Published {
        let frame = self.render().resource::<Frame>();
        let prepared = self.render().resource::<Prepared>();
        eprintln!(
            "{label}: frame={} target_tick={} committed_tick={} state_a={} ready={} written={} bed={:?}",
            self.frames,
            frame.uniform.step.x,
            prepared.completed_tick,
            prepared.state_is_a,
            prepared.ready,
            self.render().resource::<AnimWavesStatus>().written,
            prepared.bed_binding
        );
        let result = Published {
            surface: fixture::pixels(
                self.render(),
                &frame.surface,
                LOD_COUNT as u32,
                &format!("{label}-surface"),
            ),
            a: fixture::pixels(
                self.render(),
                &frame.state_a,
                LOD_COUNT as u32,
                &format!("{label}-state-a"),
            ),
            b: fixture::pixels(
                self.render(),
                &frame.state_b,
                LOD_COUNT as u32,
                &format!("{label}-state-b"),
            ),
        };
        for bytes in [&result.surface, &result.a, &result.b] {
            fixture::finite(bytes);
            assert!(
                bytes.chunks_exact(8).all(|pixel| {
                    let density = u16::from_le_bytes([pixel[0], pixel[1]]);
                    density <= 0x3c00 && pixel[2..].iter().all(|byte| *byte == 0)
                }),
                "production foam must contain density in [0,1] and zero unused channels"
            );
        }
        result
    }

    fn waves(&self, label: &str) -> Vec<u8> {
        let frame = self.render().resource::<Frame>();
        let bytes = fixture::pixels(self.render(), &frame.waves, LOD_COUNT as u32, label);
        fixture::finite(&bytes);
        assert!(
            bytes.iter().any(|byte| *byte != 0),
            "real wave producer emitted no signal"
        );
        bytes
    }

    fn clear_history_for_control(&mut self) {
        // Identical zero initial data for two production foam dispatches. This
        // does not implement advection, shoreline sampling, decay, or FFT math.
        let frame = self.render().resource::<Frame>().clone();
        let images = self.render().resource::<RenderAssets<GpuImage>>();
        let queue = self.render().resource::<RenderQueue>();
        for handle in [&frame.state_a, &frame.state_b, &frame.surface] {
            let image = images.get(handle).unwrap();
            let size = image.texture_descriptor.size;
            let zero = vec![
                0;
                size.width as usize
                    * size.height as usize
                    * size.depth_or_array_layers as usize
                    * 8
            ];
            queue.write_texture(
                TexelCopyTextureInfo {
                    texture: &image.texture,
                    mip_level: 0,
                    origin: Origin3d::ZERO,
                    aspect: TextureAspect::All,
                },
                &zero,
                TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(size.width * 8),
                    rows_per_image: Some(size.height),
                },
                size,
            );
        }
        let mut prepared = self.render_mut().resource_mut::<Prepared>();
        prepared.state_is_a = true;
        prepared.completed_tick = frame
            .uniform
            .step
            .x
            .checked_sub(1)
            .expect("control needs one pending tick");
        prepared.state_layout = None;
    }
}

#[test]
#[ignore = "native GPU required; host owns flock /tmp/apophany-capture.lock"]
fn gpu_bed_foam_history() {
    let mut rig = Rig::new();
    rig.warm();
    fixture::adapter(&rig.app);
    for _ in 0..3 {
        rig.tick();
        rig.current();
    }
    let a = rig.snapshot("foam-a");
    assert!(
        a.surface
            .chunks_exact(8)
            .any(|pixel| pixel[0] != 0 || pixel[1] != 0),
        "no production foam signal"
    );
    let waves_a = rig.waves("foam-input-waves-a");
    let history_a = rig.history();

    let image = rig
        .app
        .world()
        .resource::<Assets<Image>>()
        .get(&rig.b.image)
        .unwrap()
        .clone();
    let handle = rig.app.world().resource::<Assets<Image>>().reserve_handle();
    let mut waiting = rig.b.clone();
    waiting.image = handle.clone();
    rig.app.insert_resource(waiting);
    for index in 0..2 {
        rig.tick();
        // Archive the actual frame before a state assertion can fail in a
        // separately built behavioral-negative variant.
        let held = rig.snapshot(&format!("foam-unfilled-{index}"));
        let held_waves = rig.waves(&format!("foam-unfilled-{index}-waves"));
        assert_eq!(rig.render().resource::<BedHeightMap>().image, handle);
        assert!(
            rig.render()
                .resource::<RenderAssets<GpuImage>>()
                .get(&handle)
                .is_none()
        );
        let prepared = rig.render().resource::<Prepared>();
        assert!(!prepared.ready && !prepared.groups.created());
        assert_eq!(prepared.bed_binding, None);
        assert!(!rig.render().resource::<AnimWavesStatus>().written);
        assert_eq!(
            rig.history(),
            history_a,
            "unready bed advanced foam history"
        );
        assert!(held == a, "unready bed changed foam GPU state/publication");
        assert!(
            held_waves == waves_a,
            "unready bed changed real wave output"
        );
    }
    rig.app
        .world_mut()
        .resource_mut::<Assets<Image>>()
        .insert(handle.id(), image)
        .unwrap();
    rig.warm();
    let recovered = rig.snapshot("foam-bed-recovered");
    assert!(
        recovered.surface != a.surface,
        "recovered foam did not dispatch pending ticks"
    );
    assert!(rig.history().completed_tick > history_a.completed_tick);
    assert!(
        rig.waves("foam-input-waves-b") != waves_a,
        "recovered waves did not change"
    );

    let retained = rig.history();
    rig.render_mut().resource_mut::<Fault>().cold = true;
    rig.tick();
    assert!(
        rig.render().resource::<AnimWavesStatus>().written,
        "cold foam must not suppress the real wave producer"
    );
    assert_eq!(rig.history(), retained, "cold pipeline advanced history");
    assert!(
        rig.snapshot("foam-cold-pipelines") == recovered,
        "cold pipeline changed GPU history/publication"
    );
    rig.warm();
    let after_cold = rig.snapshot("foam-pipelines-recovered");
    assert!(after_cold.surface != recovered.surface);

    let retained = rig.history();
    rig.render_mut().resource_mut::<Fault>().withhold_bed = true;
    rig.tick();
    let input_unavailable = rig.snapshot("foam-input-unavailable");
    assert!(rig.render().resource::<AnimWavesStatus>().written);
    assert!(!rig.render().resource::<Prepared>().ready);
    assert_eq!(
        rig.history(),
        retained,
        "unready foam with written waves advanced history"
    );
    assert!(input_unavailable == after_cold);
    let (id, image) = rig
        .render_mut()
        .resource_mut::<Fault>()
        .held_bed
        .take()
        .unwrap();
    rig.render_mut()
        .resource_mut::<RenderAssets<GpuImage>>()
        .insert(id, image);
    rig.warm();
    let after_input = rig.snapshot("foam-input-recovered");
    assert!(after_input.surface != after_cold.surface);

    let retained = rig.history();
    rig.render_mut().resource_mut::<Fault>().no_waves = true;
    rig.tick();
    assert!(!rig.render().resource::<AnimWavesStatus>().written);
    assert_eq!(
        rig.history(),
        retained,
        "unwritten waves advanced foam history"
    );
    assert!(rig.snapshot("foam-waves-unwritten") == after_input);
    rig.render_mut().resource_mut::<Fault>().no_waves = false;
    rig.warm();
    assert!(rig.snapshot("foam-waves-recovered").surface != after_input.surface);

    rig.app.world_mut().remove_resource::<BedHeightMap>();
    rig.tick();
    rig.current();
    assert!(rig.render().get_resource::<BedHeightMap>().is_none());
    assert!(
        rig.render()
            .resource::<Frame>()
            .uniform
            .target_layout
            .bed_range
            .y
            < 0.0
    );
    rig.snapshot("foam-remove-bed");

    // Observable negative: A's real bind groups with B's new decode metadata.
    // Both trials start from identical zero history and execute one real tick.
    rig.app.insert_resource(rig.a.clone());
    rig.tick();
    rig.current();
    let old_groups = std::mem::take(&mut rig.render_mut().resource_mut::<Prepared>().groups);
    rig.render_mut().resource_mut::<Fault>().stale_groups = Some(old_groups);
    rig.app.insert_resource(rig.b.clone());
    rig.clear_history_for_control();
    rig.step();
    let wrong = rig.snapshot("foam-negative-stale-bed");
    {
        let mut prepared = rig.render_mut().resource_mut::<Prepared>();
        prepared.groups = pass::Groups::default();
        prepared.bed_binding = None;
    }
    rig.clear_history_for_control();
    rig.step();
    rig.current();
    let correct = rig.snapshot("foam-b-correct");
    assert!(
        wrong.surface != correct.surface,
        "stale-bed negative control was not detectable in production foam"
    );
    // Repeat a view-only replacement at one asset ID. The alternate authored
    // handle supplies a real-production reference with identical decode values.
    let mut b_decode_a_texels = rig.b.clone();
    b_decode_a_texels.image = rig.a.image.clone();
    rig.app.insert_resource(b_decode_a_texels);
    rig.clear_history_for_control();
    rig.step();
    let view_reference = rig.snapshot("foam-view-reference");
    rig.app.insert_resource(rig.b.clone());
    rig.clear_history_for_control();
    rig.step();
    rig.current();
    let before_view = rig.snapshot("foam-before-view-replacement");
    let b_id = rig.b.image.id();
    let replacement = rig
        .render()
        .resource::<RenderAssets<GpuImage>>()
        .get(&rig.a.image)
        .unwrap()
        .clone();
    let original = rig
        .render_mut()
        .resource_mut::<RenderAssets<GpuImage>>()
        .insert(b_id, replacement)
        .unwrap();
    let old_view = original.texture_view.id();
    rig.clear_history_for_control();
    rig.step();
    rig.current();
    assert_ne!(
        rig.render().resource::<Prepared>().bed_binding.unwrap().1,
        old_view
    );
    assert!(
        rig.snapshot("foam-same-handle-new-view") == view_reference,
        "same-handle GPU view replacement did not bind the new texels"
    );
    rig.render_mut()
        .resource_mut::<RenderAssets<GpuImage>>()
        .insert(b_id, original);
    rig.clear_history_for_control();
    rig.step();
    rig.current();
    assert!(rig.snapshot("foam-view-restored") == before_view);
    eprintln!(
        "production bed/foam history passed in {} frames",
        rig.frames
    );
}
