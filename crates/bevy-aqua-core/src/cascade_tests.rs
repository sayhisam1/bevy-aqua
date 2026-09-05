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
fn body_params_abi_is_six_full_vec4s() {
    // Full vec4 fields only: naga and encase disagree on smaller members
    // (the flow-advection uniform bug). flags, extent, aabb_min,
    // aabb_size, optics_a, and optics_b occupy bytes 0..96 in that order.
    assert_eq!(std::mem::size_of::<BodyParams>(), 96);
    let optics = crate::cascade::BodyOptics {
        extinction: Vec3::new(0.28, 0.16, 0.12),
        scatter_scale: 0.18,
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
    assert_eq!(bytes.len(), 96);
    let words: [f32; 24] = std::array::from_fn(|index| {
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
            0.18, 0.1, 1.0, 0.0,
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
fn default_sun_floor_preserves_wave_roughness_and_body_inheritance() {
    let surface = SurfaceParams::default();
    assert_eq!(surface.sun, Vec4::new(1.0, 0.04, 0.0, 1.0));
    assert_eq!(surface.reflection.y, 1.0);
    assert_eq!(surface.reflection.w, 0.28);
    assert_eq!(WaterOptics::CLEAR_FRESH.sun_roughness, 0.1);
    assert!(WaterOptics::DEEP_OCEAN.sun_roughness < 0.0);
}

// boundary43: reference arithmetic plus actual resolved bounds and ABI wiring.
// GPU execution of the composed shader is covered by the temporary native fixture.
fn boundary43_distance(eye: Vec2, params: BodyParams) -> f32 {
    ((eye - params.extent.xy()).abs() - Vec2::splat(params.extent.w))
        .max(Vec2::ZERO)
        .length()
}

#[test]
fn boundary43_square_distance_and_shader_contract() {
    let shader = include_str!("cascade/material.wgsl");
    assert!(shader.contains("let distance_to_extent = length(max(\n            abs(view.world_position.xz - params.extent.xy) - vec2(params.extent.w),\n            vec2(0.0),\n        ));"));
    assert!(shader.contains("if distance_to_extent > surface.far_tier.y {\n            discard;"));
    let begin = shader.find("let bounded = slot > 0u;").unwrap();
    let owner = shader[begin..]
        .find("let params = owning_body(slot);")
        .unwrap()
        + begin;
    let gate = shader[owner..].find("if bounded {").unwrap() + owner;
    let distance = shader.find("let distance_to_extent =").unwrap();
    assert!(gate < distance);
    for (center, half, eye, expected, cull) in [
        (Vec2::splat(450.0), 100.0, Vec2::ZERO, 494.97475, false),
        (Vec2::splat(450.0), 102.0, Vec2::ZERO, 492.14633, false),
        (Vec2::splat(600.0), 102.0, Vec2::ZERO, 704.2783, true),
        (Vec2::new(612.0, 0.0), 100.0, Vec2::ZERO, 512.0, false),
        (Vec2::new(613.0, 0.0), 100.0, Vec2::ZERO, 513.0, true),
        (Vec2::splat(450.0), 100.0, Vec2::splat(450.0), 0.0, false),
        (Vec2::splat(-450.0), 100.0, Vec2::ZERO, 494.97475, false),
    ] {
        let params = BodyParams::bounded(center, half, Vec2::ZERO, Vec2::ONE, false, None);
        let distance = boundary43_distance(eye, params);
        assert!((distance - expected).abs() < 0.001);
        assert_eq!(distance > 512.0, cull);
    }
}

#[test]
fn boundary43_resolved_rectangles_yaw_nonuniform_and_shear_enclose_vertices() {
    let points = vec![
        Vec2::new(-100.0, -20.0),
        Vec2::new(100.0, -20.0),
        Vec2::new(100.0, 20.0),
        Vec2::new(-100.0, 20.0),
    ];
    let shape = WaterShape::Polygon {
        points: points.clone(),
    };
    let transforms = [
        GlobalTransform::from(Transform::from_xyz(450.0, 0.0, 450.0)),
        GlobalTransform::from(
            Transform::from_xyz(450.0, 0.0, 450.0)
                .with_rotation(Quat::from_rotation_y(0.71))
                .with_scale(Vec3::new(-2.0, 1.0, 0.4)),
        ),
        GlobalTransform::from(bevy::math::Affine3A::from_mat3_translation(
            Mat3::from_cols(Vec3::new(2.0, 0.0, 0.5), Vec3::Y, Vec3::new(0.7, 0.0, 0.4)),
            Vec3::new(450.0, 0.0, 450.0),
        )),
    ];
    for transform in transforms {
        let body =
            ResolvedWaterBody::resolve(Entity::from_bits(1), &shape, None, &transform).unwrap();
        let (minimum, maximum) = body.aabb();
        let (center, half) = body.extent();
        let params = BodyParams::bounded(center, half, minimum, maximum - minimum, false, None);
        assert_eq!(params.extent, Vec4::new(center.x, center.y, 0.0, half));
        for point in &points {
            let world = body.world_point(*point);
            assert!(world.cmpge(minimum).all() && world.cmple(maximum).all());
            assert!((world - center).abs().cmple(Vec2::splat(half)).all());
            for eye in [Vec2::ZERO, Vec2::new(-120.0, 650.0), center] {
                assert!(boundary43_distance(eye, params) <= eye.distance(world) + 0.001);
            }
        }
    }
}
