use super::*;
use bevy_aqua_core::AnimWavesUniform;

#[derive(Clone, Copy, Debug)]
enum Object {
    Requests,
    Results,
    Texture,
    Waves,
    Batch,
    Sampler,
}

#[derive(Resource, Default)]
struct Injection {
    replace: Option<Object>,
    missing: Option<Object>,
    cold: bool,
    zero_weights: bool,
    gradient: bool,
    linear: bool,
}

// Run after the producer's preparation, then rerun the *production* query
// preparation. Only test code creates faults; production APIs stay unchanged.
fn inject(world: &mut World) {
    let replace = world.resource_mut::<Injection>().replace.take();
    let missing = world.resource_mut::<Injection>().missing.take();
    let device = world.resource::<RenderDevice>().clone();
    let queue = world.resource::<RenderQueue>().clone();
    let buffers = world.resource::<Buffers>().clone();
    let image = world.resource::<Data>().texture();
    if let Some(object) = missing {
        match object {
            Object::Requests => {
                world
                    .resource_mut::<RenderAssets<GpuShaderBuffer>>()
                    .remove(&buffers.requests);
            }
            Object::Results => {
                world
                    .resource_mut::<RenderAssets<GpuShaderBuffer>>()
                    .remove(&buffers.results);
            }
            Object::Texture => {
                world
                    .resource_mut::<RenderAssets<GpuImage>>()
                    .remove(&image);
            }
            Object::Waves => {
                world.resource_mut::<AnimWavesUniformSlot>().0 = None;
            }
            Object::Batch | Object::Sampler => unreachable!(),
        }
    }
    if let Some(object) = replace {
        match object {
            Object::Requests | Object::Results => {
                let (handle, size) = match object {
                    Object::Requests => (&buffers.requests, 256 * 32),
                    _ => (&buffers.results, RESULT_BYTES),
                };
                let mut assets = world.resource_mut::<RenderAssets<GpuShaderBuffer>>();
                // A stale request binding must not accidentally sample the same
                // content as the replacement. Poison the still-live old object.
                if matches!(object, Object::Requests)
                    && let Some(old) = assets.get(handle)
                {
                    queue.write_buffer(&old.buffer, 0, &vec![0; size]);
                }
                let descriptor = BufferDescriptor {
                    label: Some("query regression replacement"),
                    size: size as u64,
                    usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                };
                assets.insert(
                    handle.id(),
                    GpuShaderBuffer {
                        buffer: device.create_buffer(&descriptor),
                        buffer_descriptor: descriptor,
                        had_data: false,
                    },
                );
            }
            Object::Texture => {
                // Exact half-float texels: (1, 2, 3, 0). This is data, not a
                // replacement query shader or a CPU wave implementation.
                let descriptor = TextureDescriptor {
                    label: Some("query regression replacement"),
                    size: Extent3d {
                        width: if world.resource::<Injection>().gradient {
                            2
                        } else {
                            1
                        },
                        height: 1,
                        depth_or_array_layers: 5,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: TextureDimension::D2,
                    format: TextureFormat::Rgba16Float,
                    usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
                    view_formats: &[],
                };
                let texture = device.create_texture(&descriptor);
                let mut pixel = [0x3c00_u16, 0x4000, 0x4200, 0]
                    .map(u16::to_le_bytes)
                    .concat();
                if world.resource::<Injection>().gradient {
                    pixel.extend(
                        [0x4200_u16, 0x4400, 0x4500, 0]
                            .map(u16::to_le_bytes)
                            .concat(),
                    );
                }
                queue.write_texture(
                    TexelCopyTextureInfo {
                        texture: &texture,
                        mip_level: 0,
                        origin: Origin3d::ZERO,
                        aspect: TextureAspect::All,
                    },
                    &pixel.repeat(5),
                    TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(8 * descriptor.size.width),
                        rows_per_image: Some(1),
                    },
                    descriptor.size,
                );
                let view_descriptor = TextureViewDescriptor {
                    dimension: Some(TextureViewDimension::D2Array),
                    ..default()
                };
                let texture_view = texture.create_view(&view_descriptor);
                world.resource_mut::<RenderAssets<GpuImage>>().insert(
                    image.id(),
                    GpuImage {
                        texture,
                        texture_view,
                        sampler: device.create_sampler(&SamplerDescriptor::default()),
                        texture_descriptor: descriptor,
                        texture_view_descriptor: Some(view_descriptor),
                        had_data: true,
                    },
                );
            }
            Object::Waves => {
                let value = world.resource::<bevy_aqua_waves::Frame>().uniform().clone();
                let mut replacement = UniformBuffer::from(value);
                replacement.write_buffer(&device, &queue);
                world.resource_mut::<AnimWavesUniformSlot>().0 = Some(replacement);
            }
            Object::Batch => {
                let mut prepared = world.resource_mut::<Prepared>();
                let old = prepared.batch.as_mut().unwrap();
                old.set(QueryBatch::default());
                old.write_buffer(&device, &queue);
                prepared.batch = None;
            }
            Object::Sampler => {
                let filter = if world.resource::<Injection>().linear {
                    FilterMode::Linear
                } else {
                    FilterMode::Nearest
                };
                world.resource_mut::<Prepared>().sampler =
                    device.create_sampler(&SamplerDescriptor {
                        mag_filter: filter,
                        min_filter: filter,
                        ..default()
                    });
            }
        }
    }
    if world.resource::<Injection>().zero_weights {
        let mut slot = world.resource_mut::<AnimWavesUniformSlot>();
        if let Some(uniform) = slot.0.as_mut() {
            let mut value: AnimWavesUniform = uniform.get().clone();
            for cascade in &mut value.layout.cascades {
                cascade.weight = 0.0;
            }
            uniform.set(value);
            uniform.write_buffer(&device, &queue);
        }
    }
}

fn cold_pipeline(
    mut injection: ResMut<Injection>,
    mut prepared: ResMut<Prepared>,
    server: Res<AssetServer>,
    cache: Res<PipelineCache>,
) {
    if std::mem::take(&mut injection.cold) {
        // This executes after PipelineCache processing for the frame. The real
        // shader is queued, but cannot be compiled until a later render frame.
        prepared.passes = pass::Passes::new(&server, &cache, pass_table());
        assert!(
            prepared
                .passes
                .ready_all(&cache, &[(QUERY, SAMPLE)])
                .is_none()
        );
    }
}

fn word(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
}

fn displacement(bytes: &[u8]) -> Vec3 {
    Vec3::new(
        f32::from_bits(word(bytes, 0)),
        f32::from_bits(word(bytes, 4)),
        f32::from_bits(word(bytes, 8)),
    )
}

fn fresh(rig: &mut Rig, label: &str) -> Vec<u8> {
    // Two frames also cover late replacement of a result object: Bevy had
    // already selected the old readback source in PrepareResources that frame.
    let target = rig
        .app
        .world()
        .resource::<Registry>()
        .generation
        .wrapping_add(2);
    rig.clear_observed();
    rig.snapshot(label, |bytes| {
        let generation = word(bytes, 36);
        (generation == target || generation_after(generation, target)) && word(bytes, 40) == 1
    })
}

fn replay(rig: &mut Rig, bytes: &[u8]) {
    // Replay a captured GPU packet through the production observer. This is
    // registry/lifecycle proof, separate from the real dispatch/readback proof.
    rig.app.world_mut().trigger(ReadbackComplete {
        entity: rig.readback,
        data: bytes.to_vec(),
    });
}

fn assert_sample(bytes: &[u8], expected: Vec3) {
    assert!(
        (displacement(bytes) - expected).length() < 0.0001,
        "replacement resource was not sampled: {:?}, expected {expected:?}",
        displacement(bytes)
    );
}

pub(super) fn verify(rig: &mut Rig, survivor: Entity) {
    rig.app
        .sub_app_mut(RenderApp)
        .init_resource::<Injection>()
        .add_systems(
            Render,
            (inject, prepare_bind_groups)
                .chain()
                .after(seed_results)
                .after(RenderSystems::PrepareBindGroups)
                .before(RenderSystems::Render),
        )
        .add_systems(
            Core3d,
            cold_pipeline
                .after(suppress_waves)
                .before(dispatch_wave_query),
        );

    // Production displacement at two positions must differ with time frozen.
    let a = Vec3::new(1.0, 0.0, 2.0);
    let b = Vec3::new(17.0, 0.0, -11.0);
    rig.position(survivor, a);
    let at_a = fresh(rig, "position-a");
    rig.position(survivor, b);
    let at_b = fresh(rig, "position-b");
    assert!(
        (displacement(&at_a) - displacement(&at_b)).length() > 0.001,
        "test positions did not produce distinct production-wave samples"
    );
    for (label, position, expected) in [
        ("position-a-again", a, displacement(&at_a)),
        ("position-b-again", b, displacement(&at_b)),
    ] {
        rig.position(survivor, position);
        let bytes = fresh(rig, label);
        assert_sample(&bytes, expected);
        assert!((rig.sample(survivor).displacement - expected).length() < 0.0001);
    }

    // Adjacent integer identities beyond f32 precision and serial wrap go
    // through real GPU storage, not an encase-only round trip.
    rig.render_mut().resource_mut::<Fault>().no_waves = true;
    rig.app
        .world_mut()
        .entity_mut(survivor)
        .remove::<WaveQuery>();
    {
        let mut registry = rig.app.world_mut().resource_mut::<Registry>();
        registry.next_slot = 0x0100_0000;
        registry.generation = u32::MAX - 2;
    }
    rig.app.world_mut().entity_mut(survivor).insert(WaveQuery);
    let neighbor = rig.spawn(1)[0];
    rig.step();
    assert!(!rig.sample(survivor).valid);
    replay(rig, &at_b);
    assert!(
        !rig.sample(survivor).valid,
        "removed slot accepted an old GPU packet"
    );
    rig.render_mut().resource_mut::<Fault>().no_waves = false;
    rig.clear_observed();
    let before_wrap = rig.snapshot("generation-max", |bytes| {
        word(bytes, 36) == u32::MAX && word(bytes, 40) == 1
    });
    assert_eq!(word(&before_wrap, 32), 0x0100_0001);
    assert_eq!(word(&before_wrap, ROW_BYTES + 32), 0x0100_0002);
    let after_wrap = rig.snapshot("generation-zero", |bytes| {
        word(bytes, 36) == 0 && word(bytes, 40) == 1
    });
    assert_eq!(word(&after_wrap, ROW_BYTES + 36), 0);
    let applied = rig.app.world().resource::<Registry>().probes[&0x0100_0001]
        .latest_applied
        .unwrap();
    assert!(
        applied < 16,
        "wrap result was not accepted by the production observer"
    );
    rig.app.world_mut().despawn(neighbor);

    // GPU work from the preceding frame is complete but its async event has
    // not yet been delivered. Reclaim/re-entry must close that old epoch.
    rig.render_mut().resource_mut::<Fault>().no_waves = true;
    rig.app
        .world_mut()
        .entity_mut(survivor)
        .remove::<WaveQuery>();
    rig.app.world_mut().entity_mut(survivor).insert(WaveQuery);
    rig.step();
    replay(rig, &after_wrap);
    assert!(!rig.sample(survivor).valid);
    let doomed = rig.spawn(1)[0];
    rig.render_mut().resource_mut::<Fault>().no_waves = false;
    let before_despawn = fresh(rig, "before-despawn");
    rig.render_mut().resource_mut::<Fault>().no_waves = true;
    rig.app.world_mut().despawn(doomed);
    let respawned = rig.spawn(1)[0];
    rig.step();
    replay(rig, &before_despawn);
    assert!(!rig.sample(respawned).valid);
    rig.app.world_mut().despawn(respawned);

    let body = ResolvedWaterBody::resolve(
        Entity::from_bits(1),
        &bevy_aqua_core::WaterShape::Circle { radius: 5.0 },
        None,
        &GlobalTransform::IDENTITY,
    )
    .unwrap();
    rig.app.world_mut().remove_resource::<Ocean>();
    rig.app.world_mut().resource_mut::<ResolvedWaterBodies>().0 = vec![body];
    rig.position(survivor, Vec3::ZERO);
    rig.render_mut().resource_mut::<Fault>().no_waves = false;
    let before_exit = fresh(rig, "bounded-inside");
    rig.render_mut().resource_mut::<Fault>().no_waves = true;
    rig.position(survivor, Vec3::splat(100.0));
    rig.step();
    assert!(!rig.sample(survivor).valid);
    rig.position(survivor, Vec3::ZERO);
    rig.step();
    replay(rig, &before_exit);
    assert!(
        !rig.sample(survivor).valid,
        "pre-exit packet reopened the bounded-body epoch"
    );
    rig.render_mut().resource_mut::<Fault>().no_waves = false;
    assert_sample(&fresh(rig, "bounded-reentered"), Vec3::ZERO);
    assert!(rig.sample(survivor).valid);
    rig.app
        .world_mut()
        .resource_mut::<ResolvedWaterBodies>()
        .0
        .clear();
    rig.app.world_mut().insert_resource(Ocean::default());
    rig.position(survivor, a);

    rig.render_mut().resource_mut::<Injection>().cold = true;
    rig.seed(vec![POISON; RESULT_BYTES]);
    let cold = rig.snapshot("cold-pipeline", |bytes| {
        bytes[ROW_BYTES..].iter().all(|byte| *byte == POISON)
    });
    assert_eq!(
        cold,
        vec![POISON; RESULT_BYTES],
        "cold query pipeline wrote results"
    );
    fresh(rig, "pipeline-recovered");

    for object in [
        Object::Requests,
        Object::Results,
        Object::Texture,
        Object::Waves,
        Object::Batch,
        Object::Sampler,
    ] {
        let before = rig.render().resource::<Prepared>().bound.unwrap();
        rig.render_mut().resource_mut::<Injection>().replace = Some(object);
        let bytes = fresh(rig, &format!("replace-{object:?}"));
        let after = rig.render().resource::<Prepared>().bound.unwrap();
        assert_ne!(before, after, "test did not replace {object:?}");
        assert_eq!(
            word(&bytes, 32),
            rig.app.world().resource::<Registry>().slots[&survivor]
        );
        if matches!(
            object,
            Object::Texture | Object::Waves | Object::Batch | Object::Sampler
        ) {
            assert_sample(&bytes, Vec3::new(1.0, 2.0, 3.0));
        }
    }
    // Distinguish sampler identity by a real filtering change at a two-texel seam.
    rig.position(survivor, Vec3::ZERO);
    rig.render_mut().resource_mut::<Injection>().gradient = true;
    rig.render_mut().resource_mut::<Injection>().replace = Some(Object::Texture);
    assert_sample(&fresh(rig, "nearest-sampler"), Vec3::new(3.0, 4.0, 5.0));
    rig.render_mut().resource_mut::<Injection>().linear = true;
    rig.render_mut().resource_mut::<Injection>().replace = Some(Object::Sampler);
    assert_sample(
        &fresh(rig, "replacement-linear-sampler"),
        Vec3::new(2.0, 3.0, 4.0),
    );
    rig.render_mut().resource_mut::<Injection>().gradient = false;
    rig.render_mut().resource_mut::<Injection>().replace = Some(Object::Texture);
    assert_sample(
        &fresh(rig, "constant-texture-restored"),
        Vec3::new(1.0, 2.0, 3.0),
    );
    rig.position(survivor, a);

    // Distinguish the new uniform by its actual sampling effect, not just ID.
    rig.render_mut().resource_mut::<Injection>().replace = Some(Object::Waves);
    rig.render_mut().resource_mut::<Injection>().zero_weights = true;
    assert_sample(&fresh(rig, "new-wave-uniform-zero-weights"), Vec3::ZERO);
    rig.render_mut().resource_mut::<Injection>().zero_weights = false;
    assert_sample(
        &fresh(rig, "wave-uniform-restored"),
        Vec3::new(1.0, 2.0, 3.0),
    );

    for object in [
        Object::Requests,
        Object::Results,
        Object::Texture,
        Object::Waves,
    ] {
        rig.render_mut().resource_mut::<Injection>().missing = Some(object);
        rig.seed(vec![POISON; RESULT_BYTES]);
        assert!(
            !rig.render().resource::<Prepared>().ready,
            "missing {object:?} did not close readiness"
        );
        // Restore next frame. The pending readback still owns the old result
        // object, so it can prove the missing-resource frame did not write.
        rig.render_mut().resource_mut::<Injection>().replace = Some(object);
        let missing = rig.snapshot(&format!("missing-{object:?}"), |bytes| {
            bytes.iter().all(|byte| *byte == POISON)
        });
        assert_eq!(missing, vec![POISON; RESULT_BYTES]);
        assert_sample(
            &fresh(rig, &format!("recovered-{object:?}")),
            Vec3::new(1.0, 2.0, 3.0),
        );
    }

    // The single unsubmitted valid probe keeps its actual prior component.
    let extra = rig.spawn(256);
    rig.render_mut().resource_mut::<Fault>().no_waves = true;
    rig.step();
    let previous = WaveSurface {
        displacement: Vec3::splat(91.0),
        normal: Vec3::Y,
        crest: 0.25,
        valid: true,
    };
    for entity in extra.iter().copied().chain([survivor]) {
        *rig.app.world_mut().get_mut::<WaveSurface>(entity).unwrap() = previous;
    }
    rig.step();
    let registry = rig.app.world().resource::<Registry>();
    let omitted: Vec<_> = registry
        .probes
        .values()
        .filter(|probe| probe.latest_submitted != registry.generation)
        .map(|probe| probe.entity)
        .collect();
    assert_eq!(omitted.len(), 1);
    rig.render_mut().resource_mut::<Fault>().no_waves = false;
    fresh(rig, "over-budget");
    assert_eq!(rig.render().resource::<Batch>().count, 256);
    assert_eq!(rig.sample(omitted[0]), previous);
    for entity in extra {
        rig.app.world_mut().despawn(entity);
    }
}
