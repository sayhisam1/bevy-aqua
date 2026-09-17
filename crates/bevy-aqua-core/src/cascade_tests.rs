use super::*;
use crate::*;

const UV_CENTER: f32 = 0.5;

/// World metres covered by one cascade slice edge to edge.
fn coverage(cascade: Cascade) -> f32 {
    cascade.texel_width * RESOLUTION as f32
}

/// Maps world XZ into one cascade's normalized texture coordinates.
///
/// Reimplementation of the approach in Crest `Shaders/OceanHelpersNew.hlsl` (`WorldToUV`).
fn world_to_uv(world: Vec2, cascade: Cascade) -> Vec2 {
    (world - cascade.center) / coverage(cascade) + Vec2::splat(UV_CENTER)
}

/// Maps normalized cascade coordinates back into world XZ.
///
/// Reimplementation of the approach in Crest `Shaders/OceanHelpersNew.hlsl` (`UVToWorld`).
fn uv_to_world(uv: Vec2, cascade: Cascade) -> Vec2 {
    coverage(cascade) * (uv - Vec2::splat(UV_CENTER)) + cascade.center
}

#[test]
fn body_params_abi_is_seven_full_vec4s() {
    // Full vec4 fields only: naga and encase disagree on smaller members
    // (the flow-advection uniform bug). flags, extent, aabb_min,
    // aabb_size, optics_a, optics_b, and optics_c occupy bytes 0..112 in that
    // order.
    assert_eq!(std::mem::size_of::<BodyParams>(), 112);
    let optics = crate::cascade::BodyOptics {
        extinction: Vec3::new(0.28, 0.16, 0.12),
        scatter_scale: 0.18,
        scatter_tint: Vec3::new(0.85, 1.0, 1.22),
        scattering_asymmetry: 0.8,
        sun_roughness: 0.1,
    };
    let mut bytes = Vec::new();
    bevy::render::render_resource::encase::UniformBuffer::new(&mut bytes)
        .write(&BodyParams::bounded(
            Vec2::new(-40.0, 20.0),
            25.0,
            Vec2::new(-70.0, -5.0),
            Vec2::new(60.0, 50.0),
            true,
            Some(optics),
        ))
        .expect("body params write");
    assert_eq!(bytes.len(), 112);
    let words: [f32; 28] = std::array::from_fn(|index| {
        f32::from_le_bytes(bytes[index * 4..index * 4 + 4].try_into().unwrap())
    });
    assert_eq!(
        words,
        [
            1.0, 1.0, 0.0, 0.0, //
            -40.0, 20.0, 0.0, 25.0, //
            -70.0, -5.0, 0.0, 0.0, //
            60.0, 50.0, 0.0, 0.0, //
            0.28, 0.16, 0.12, 1.0, //
            0.18, 0.1, 1.0, 0.8, //
            0.85, 1.0, 1.22, 0.0,
        ]
    );
}

#[test]
fn gpu_cascade_sentinel_and_bed_echo() {
    assert_eq!(std::mem::size_of::<GpuCascade>(), 32);
    let mut gpu = GpuLayout::new(&layout(Vec2::ZERO), Vec2::ZERO, 1.25);
    assert_eq!(gpu.center.z, 1.25);
    let mut expected = gpu.cascades[LOD_COUNT - 1];
    expected.weight = 0.0;
    assert_eq!(gpu.cascades[LOD_COUNT], expected);

    // No bed map: negative decode span marks every sample as deep default.
    gpu.set_bed(None, 2.5);
    assert_eq!(gpu.bed_range.y, crate::bed::NO_BED_SPAN);
    assert_eq!(gpu.bed_range.z, 2.5);

    // A supplied map echoes its world bounds and height range.
    let mut images = bevy::asset::Assets::<bevy::image::Image>::default();
    let map = crate::bed::BedHeightMap::from_height_fn(
        &mut images,
        |x, _z| x,
        4,
        Vec2::new(-1.5, -3.0),
        1.0,
    );
    gpu.set_bed(Some(&map), -0.5);
    assert_eq!(gpu.bed_transform.xy(), Vec2::new(-1.5, -3.0));
    // size = step * (resolution - 1) = 3 m between first and last texel
    // centres, so the uniform carries the reciprocal.
    assert_eq!(gpu.bed_transform.zw(), Vec2::splat(1.0 / 3.0));
    assert_eq!(gpu.bed_range.x, map.height_range[0]);
    assert_eq!(gpu.bed_range.y, map.height_range[1] - map.height_range[0]);
    assert_eq!(gpu.bed_range.z, -0.5);
}

#[test]
fn scale_and_coverage_double_per_lod() {
    let cascades = layout(Vec2::new(17.0, -9.0));
    for pair in cascades.windows(2) {
        assert_eq!(pair[1].scale, pair[0].scale * 2.0);
        assert_eq!(coverage(pair[1]), coverage(pair[0]) * 2.0);
        assert_eq!(pair[1].texel_width, pair[0].texel_width * 2.0);
    }
}

#[test]
fn centres_snap_down_for_negative_positions() {
    for cascade in layout(Vec2::new(-0.01, -31.7)) {
        let texels = cascade.center / cascade.texel_width;
        assert_eq!(texels, texels.round());
        assert!(cascade.center.x <= -0.01);
        assert!(cascade.center.y <= -31.7);
    }
}

#[test]
fn first_cascade_corners_map_to_unit_uv() {
    let cascade = layout(Vec2::ZERO)[0];
    let half_coverage = Vec2::splat(48.0);

    assert_eq!(coverage(cascade), 96.0);
    assert_eq!(world_to_uv(-half_coverage, cascade), Vec2::ZERO);
    assert_eq!(world_to_uv(half_coverage, cascade), Vec2::ONE);
    assert_eq!(uv_to_world(Vec2::ZERO, cascade), -half_coverage);
    assert_eq!(uv_to_world(Vec2::ONE, cascade), half_coverage);
}

#[test]
fn depth_gated_transmission_limits_residual_to_two_to_the_minus_ten() {
    const MAXIMUM_RESIDUAL: f32 = 1.0 / 1024.0;
    for (_, optics) in WaterOptics::PRESETS {
        let mut surface = SurfaceParams::default();
        surface.apply_optics(&optics);
        let density = surface.fog_density.truncate();
        assert_eq!(surface.scatter_tint.truncate(), optics.scatter_tint);
        let minimum_extinction = density.min_element();
        let cutoff = 1024.0_f32.ln() / minimum_extinction;
        assert!(minimum_extinction.is_finite() && minimum_extinction > 0.0);
        assert!(
            (-density * cutoff)
                .exp()
                .cmple(Vec3::splat(MAXIMUM_RESIDUAL * (1.0 + 1e-5)))
                .all()
        );
    }

    let crest_cutoff = 1024.0_f32.ln() / 0.3;
    assert!((crest_cutoff - 23.104_906).abs() < 0.000_01);
}

#[test]
fn detail_mips_preserve_filtered_slope_variance() {
    let mut source = Vec::new();
    for slope in [Vec2::X, -Vec2::X, Vec2::X, -Vec2::X] {
        source.extend_from_slice(&encode_detail_normal(slope, slope.length_squared()));
    }
    let filtered = downsample_detail_normals(&source, 2);
    let mean = Vec2::new(
        filtered[0] as f32 / 127.5 - 1.0,
        filtered[1] as f32 / 127.5 - 1.0,
    );
    let second_moment = 2.0 * filtered[2] as f32 / 255.0;
    assert!(mean.length() < 0.01);
    assert!((second_moment - 1.0).abs() < 0.01);
    assert!((second_moment - mean.length_squared() - 1.0).abs() < 0.01);
}

#[test]
fn detail_mips_leave_constant_slopes_without_variance() {
    let slope = Vec2::new(0.25, -0.5);
    let pixel = encode_detail_normal(slope, slope.length_squared());
    let filtered = downsample_detail_normals(&pixel.repeat(4), 2);
    let mean = Vec2::new(
        filtered[0] as f32 / 127.5 - 1.0,
        filtered[1] as f32 / 127.5 - 1.0,
    );
    let second_moment = 2.0 * filtered[2] as f32 / 255.0;
    assert!((second_moment - mean.length_squared()).abs() < 0.015);
}

#[test]
fn make_texture_includes_displacement_and_surface_layers() {
    let image = make_texture();
    assert_eq!(
        image.texture_descriptor.size.depth_or_array_layers,
        CASCADE_LAYER_COUNT
    );
    let scratch = make_lod_scratch();
    assert_eq!(
        scratch.texture_descriptor.size.depth_or_array_layers,
        LOD_COUNT as u32
    );
}
