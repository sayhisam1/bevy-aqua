//! Production FFT/bed regression. Explicitly run under the host GPU lock.

use super::*;
use crate::AquaWavesPlugin;
use bevy_aqua_core::{BedHeightMap, OceanWaves};
use std::time::{Duration, Instant};

#[path = "gpu_fixture.rs"]
mod fixture;

#[derive(Resource, Default)]
struct Fault {
    fft_bins: Option<f32>,
    stale_groups: Option<pass::Groups>,
    cold: bool,
}

fn wrong_bins(mut fault: ResMut<Fault>, mut frame: ResMut<Frame>) {
    if let Some(bins) = fault.fft_bins.take() {
        // The old failure: dispatch plans use this frame's count, but the real
        // production shader is given the previous count. No shader is replaced.
        frame.fft_uniform.mode.x = bins;
    }
}

fn stale_groups(mut fault: ResMut<Fault>, mut prepared: ResMut<Prepared>) {
    if let Some(groups) = fault.stale_groups.take() {
        prepared.groups = groups;
    }
}

fn cold_pipeline(
    mut fault: ResMut<Fault>,
    mut prepared: ResMut<Prepared>,
    server: Res<AssetServer>,
    stockham: Res<StockhamShader>,
    cache: Res<PipelineCache>,
) {
    if std::mem::take(&mut fault.cold) {
        // After the pipeline-cache stage, queue the actual production pipelines.
        // They cannot compile until a later render frame.
        prepared.passes = pass::Passes::new(&server, &cache, pass_table(&stockham));
        assert!(prepared.passes.ready(&cache, EVOLVE, "evolve").is_none());
    }
}

struct Rig {
    app: App,
    a: BedHeightMap,
    b: BedHeightMap,
    start: Instant,
    frames: usize,
}

#[derive(PartialEq)]
struct Published {
    waves: Vec<u8>,
    surface: Vec<u8>,
}

impl Rig {
    fn new(start_with_bed: bool) -> Self {
        let mut app = fixture::app();
        let (a, b) = fixture::beds(&mut app);
        if start_with_bed {
            app.insert_resource(a.clone());
        }
        app.add_plugins(AquaWavesPlugin);
        app.sub_app_mut(RenderApp)
            .init_resource::<Fault>()
            .add_systems(
                Render,
                wrong_bins
                    .before(prepare_bind_groups)
                    .in_set(RenderSystems::PrepareBindGroups),
            )
            .add_systems(
                Render,
                stale_groups
                    .after(prepare_bind_groups)
                    .in_set(RenderSystems::PrepareBindGroups),
            )
            .add_systems(Core3d, cold_pipeline.before(write_anim_waves));
        fixture::finish(&mut app);
        Self {
            app,
            a,
            b,
            start: Instant::now(),
            frames: 0,
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

    fn warm(&mut self, bins: u32) {
        let mut first_write = false;
        for _ in 0..1024 {
            self.step();
            if !self.render().resource::<AnimWavesStatus>().written {
                continue;
            }
            if !first_write {
                fixture::adapter(&self.app);
                self.snapshot(&format!("waves-first-write-{}-bins-{bins}", self.frames));
                self.current(bins);
                first_write = true;
            }
            // Warm all four-bin entry points before the directed transition;
            // otherwise a cold pipeline can hide the one-frame bin mismatch.
            let mut all = self.render().resource::<Frame>().clone();
            all.fft_bins = 4;
            if fft_spans(
                &all,
                self.render().resource::<Prepared>(),
                self.render().resource::<PipelineCache>(),
            )
            .is_some()
                && self
                    .render()
                    .resource::<RenderAssets<GpuImage>>()
                    .get(&self.b.image)
                    .is_some()
            {
                return;
            }
        }
        panic!("production FFT pipelines/assets did not become ready");
    }

    fn current(&self, bins: u32) {
        let frame = self.render().resource::<Frame>();
        let prepared = self.render().resource::<Prepared>();
        assert!(prepared.ready && prepared.groups.created());
        assert!(self.render().resource::<AnimWavesStatus>().written);
        assert_eq!(frame.fft_bins, bins, "host dispatch count");
        assert_eq!(
            frame.fft_uniform.mode.x, bins as f32,
            "current extracted uniform count"
        );
        assert_eq!(
            prepared.fft_uniform.as_ref().unwrap().get().mode.x,
            bins as f32,
            "actual upload count"
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
                .fft_uniform
                .as_ref()
                .unwrap()
                .get()
                .layout
                .bed_range,
            frame.fft_uniform.layout.bed_range
        );
    }

    fn snapshot(&self, label: &str) -> Published {
        let result = self.readback(label);
        fixture::finite(&result.waves);
        fixture::finite(&result.surface);
        assert!(
            result
                .waves
                .chunks_exact(2)
                .any(|word| u16::from_le_bytes([word[0], word[1]]) & 0x7fff != 0),
            "no wave signal"
        );
        result
    }

    fn readback(&self, label: &str) -> Published {
        let frame = self.render().resource::<Frame>();
        let prepared = self.render().resource::<Prepared>();
        eprintln!(
            "{label}: frame={} host_bins={} uniform_bins={} written={} ready={} bed={:?}",
            self.frames,
            frame.fft_bins,
            frame.fft_uniform.mode.x,
            self.render().resource::<AnimWavesStatus>().written,
            prepared.ready,
            prepared.bed_binding
        );
        let result = Published {
            waves: fixture::pixels(
                self.render(),
                &frame.output,
                LOD_COUNT as u32,
                &format!("{label}-displacement"),
            ),
            surface: fixture::pixels(
                self.render(),
                &frame.surface,
                if frame.fft_bins == 1 { 15 } else { 25 },
                &format!("{label}-surface"),
            ),
        };
        result
    }

    fn transition(&mut self, label: &str, bins: u32) -> Published {
        self.step();
        let first = self.snapshot(&format!("{label}-event"));
        self.current(bins);
        self.step();
        let settled = self.snapshot(&format!("{label}-next"));
        self.current(bins);
        assert!(
            first == settled,
            "{label}: transition GPU output differs from the next frame at identical time/input"
        );
        settled
    }

    fn rebuild(&mut self) {
        let mut prepared = self.render_mut().resource_mut::<Prepared>();
        prepared.groups = pass::Groups::default();
        prepared.bed_binding = None;
    }
}

#[test]
#[ignore = "native GPU required; host owns flock /tmp/apophany-capture.lock"]
fn gpu_bed_fft_startup_with_bed() {
    let mut rig = Rig::new(true);
    rig.warm(4);
    rig.transition("waves-startup-bed", 4);
}

#[test]
#[ignore = "native GPU required; host owns flock /tmp/apophany-capture.lock"]
fn gpu_bed_fft_transitions() {
    let mut rig = Rig::new(false);
    rig.warm(1);
    let deep = rig.transition("waves-no-bed", 1);
    rig.app.insert_resource(rig.a.clone());
    let a = rig.transition("waves-insert-a", 4);
    assert!(
        a.waves != deep.waves,
        "bed did not affect actual wave output"
    );
    rig.app
        .world_mut()
        .resource_mut::<OceanWaves>()
        .shallow_water_attenuation = 0.0;
    rig.transition("waves-attenuation-off", 1);
    rig.app
        .world_mut()
        .resource_mut::<OceanWaves>()
        .shallow_water_attenuation = 0.95;
    let a_again = rig.transition("waves-attenuation-on", 4);
    assert!(
        a == a_again,
        "same A input did not reproduce the same GPU output"
    );

    // Negative sensitivity control: run the production shader with the old lag
    // and prove its output is distinguishable from the correct four-bin output.
    rig.render_mut().resource_mut::<Fault>().fft_bins = Some(1.0);
    rig.step();
    let wrong = rig.readback("waves-negative-stale-bin");
    assert!(
        wrong.waves != a.waves || wrong.surface != a.surface,
        "bin-lag negative control was not detectable"
    );
    rig.step();
    rig.current(4);
    assert!(rig.snapshot("waves-bin-recovered") == a);

    // View identity can change without changing the authored asset handle.
    // A separate normal-handle run supplies the oracle; no sampling arithmetic
    // is duplicated. This probes defensive GPU rebinding, not terrain editing.
    let mut a_decode_b_texels = rig.a.clone();
    a_decode_b_texels.image = rig.b.image.clone();
    rig.app.insert_resource(a_decode_b_texels);
    let view_reference = rig.transition("waves-view-reference", 4);
    rig.app.insert_resource(rig.a.clone());
    assert!(rig.transition("waves-before-view-replacement", 4) == a);
    let a_id = rig.a.image.id();
    let replacement = rig
        .render()
        .resource::<RenderAssets<GpuImage>>()
        .get(&rig.b.image)
        .unwrap()
        .clone();
    let original = rig
        .render_mut()
        .resource_mut::<RenderAssets<GpuImage>>()
        .insert(a_id, replacement)
        .unwrap();
    let old_view = original.texture_view.id();
    let rebound = rig.transition("waves-same-handle-new-view", 4);
    assert_ne!(
        rig.render().resource::<Prepared>().bed_binding.unwrap().1,
        old_view
    );
    assert!(
        rebound == view_reference,
        "same-handle GPU view replacement did not bind the new texels"
    );
    rig.render_mut()
        .resource_mut::<RenderAssets<GpuImage>>()
        .insert(a_id, original);
    assert!(rig.transition("waves-view-restored", 4) == a);

    // Stale A groups paired with current B metadata are a separate failure.
    let old_groups = std::mem::take(&mut rig.render_mut().resource_mut::<Prepared>().groups);
    rig.render_mut().resource_mut::<Fault>().stale_groups = Some(old_groups);
    rig.app.insert_resource(rig.b.clone());
    rig.step();
    let wrong_bed = rig.readback("waves-negative-stale-bed");
    rig.rebuild();
    let b = rig.transition("waves-b-correct", 4);
    assert!(
        wrong_bed.waves != b.waves || wrong_bed.surface != b.surface,
        "stale-bed negative control was not detectable"
    );
    assert!(a.waves != b.waves, "replacement B did not affect waves");

    // Reserve a handle without creating an Image. No timing assumption about
    // upload latency can accidentally turn this into an already-ready case.
    let image = rig
        .app
        .world()
        .resource::<Assets<Image>>()
        .get(&rig.a.image)
        .unwrap()
        .clone();
    let handle = rig.app.world().resource::<Assets<Image>>().reserve_handle();
    let mut waiting = rig.a.clone();
    waiting.image = handle.clone();
    rig.app.insert_resource(waiting);
    for index in 0..2 {
        rig.step();
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
        assert!(
            rig.snapshot(&format!("waves-unfilled-{index}")) == b,
            "unready frame changed published wave output"
        );
    }
    rig.app
        .world_mut()
        .resource_mut::<Assets<Image>>()
        .insert(handle.id(), image)
        .unwrap();
    rig.warm(4);
    assert!(rig.transition("waves-filled-recovered", 4) == a);

    rig.render_mut().resource_mut::<Fault>().cold = true;
    rig.step();
    assert!(!rig.render().resource::<AnimWavesStatus>().written);
    assert!(rig.snapshot("waves-cold-pipelines") == a);
    rig.warm(4);
    assert!(rig.transition("waves-pipelines-recovered", 4) == a);

    rig.app.world_mut().remove_resource::<BedHeightMap>();
    let removed = rig.transition("waves-remove-bed", 1);
    assert!(rig.render().get_resource::<BedHeightMap>().is_none());
    assert!(
        rig.render()
            .resource::<Frame>()
            .fft_uniform
            .layout
            .bed_range
            .y
            < 0.0
    );
    assert!(
        removed == deep,
        "removal did not recover the no-bed GPU output"
    );
    eprintln!(
        "production bed/FFT transitions passed in {} frames",
        rig.frames
    );
}
