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
fn pattern_texture_puts_caustics_in_green() {
    let image = make_pattern_texture();
    let data = image.data.as_ref().expect("pattern pixels");
    let x = 20u32;
    let y = 10u32;
    let texel = 4 * (y * PATTERN_SIZE + x) as usize;
    assert_eq!(
        data[texel + 1],
        bevy_aqua_core::caustic_green_srgb(x, y, PATTERN_SIZE)
    );
}
