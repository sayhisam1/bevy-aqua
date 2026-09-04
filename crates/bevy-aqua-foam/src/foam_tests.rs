use super::*;

#[test]
fn fixed_cadence_is_independent_of_render_rate() {
    for fps in [30_u32, 60, 120] {
        let elapsed = (0..fps).fold(0.0, |time, _| time + 1.0 / f64::from(fps));
        assert_eq!(target_tick(elapsed), 30, "{fps} Hz");
    }
    assert_eq!(target_tick(0.0), 0);
    assert_eq!(target_tick(STEP_SECONDS as f64), 1);
}

#[test]
fn state_texture_matches_the_generated_shader_contract() {
    let image = make_state_texture();
    let size = image.texture_descriptor.size;
    assert_eq!(size.width, RESOLUTION);
    assert_eq!(size.height, RESOLUTION);
    assert_eq!(size.depth_or_array_layers, FOAM_LOD_COUNT);
    assert_eq!(image.texture_descriptor.dimension, TextureDimension::D2);
    assert_eq!(image.texture_descriptor.format, TextureFormat::Rgba16Float);
    assert!(
        image
            .texture_descriptor
            .usage
            .contains(TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING)
    );
}

#[test]
fn foam_current_uses_distinct_source_and_history_coordinates() {
    let shader = include_str!("foam.wgsl");
    assert!(shader.contains("world_xz - foam.advection.xy * foam.advection.z"));
    assert!(shader.contains("world_xz - foam.advection.xy * dt"));
    assert_eq!(
        shader
            .matches("world_to_uv(wave_sample_xz(world_xz), cascade)")
            .count(),
        2
    );
    assert!(shader.contains("reproject(history_xz, slice, foam.source_layout)"));
    assert!(shader.contains("reproject(history_xz, slice, foam.target_layout)"));
    // Density lookup is world anchored: do not add a second material advection.
    let shade = include_str!("shade.wgsl");
    let density = shade
        .split("fn sample_foam_density(")
        .nth(1)
        .unwrap()
        .split("fn surface_foam_mask(")
        .next()
        .unwrap();
    assert!(!density.contains("advected_world"));
    let world = Vec2::new(11.0, -4.0);
    let flow = Vec2::new(2.0, -1.0);
    let wave_time = 3.0;
    assert_eq!(world - flow * wave_time, Vec2::new(5.0, -1.0));
    assert_eq!(world - flow * 0.0, world); // layout-only dispatch
    let dt = STEP_SECONDS;
    assert!(((world - flow * dt) + flow * dt - world).length() < 1e-5);
}

#[test]
fn foam_advection_uniform_follows_the_two_layouts_and_source_parameters() {
    let layout = lod::GpuLayout::new(&lod::layout(Vec2::ZERO), Vec2::ZERO, 0.0);
    let mut uniform = Uniform::new(layout);
    uniform.advection = Vec4::new(2.0, 0.6, 6.0, 0.0);
    let mut bytes = Vec::new();
    bevy::render::render_resource::encase::UniformBuffer::new(&mut bytes)
        .write(&uniform)
        .expect("foam uniform write");
    let expected_offset = 2 * lod::GpuLayout::min_size().get() as usize + 3 * 16;
    assert_eq!(bytes.len(), expected_offset + 16);
    let values: [f32; 4] = std::array::from_fn(|i| {
        let start = expected_offset + i * 4;
        f32::from_le_bytes(bytes[start..start + 4].try_into().unwrap())
    });
    assert_eq!(values, [2.0, 0.6, 6.0, 0.0]);
}
