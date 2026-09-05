use super::*;

fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = if bits & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exponent = ((bits & 0x7C00) >> 10) as i32;
    let fraction = bits & 0x03FF;
    if exponent == 0 {
        return sign * (fraction as f32) * 2.0_f32.powi(-24);
    }
    if exponent == 0x1F {
        return sign * f32::INFINITY;
    }
    sign * (1.0 + fraction as f32 / 1024.0) * 2.0_f32.powf(exponent as f32 - 15.0)
}

fn read_texel(
    image: &Image,
    column: u32,
    row: u32,
    width: u32,
    height: u32,
    layer: u32,
    channels: usize,
) -> Vec<f32> {
    let layer_stride = (width * height * channels as u32 * 2) as usize;
    let base =
        layer_stride * layer as usize + ((row * width + column) * channels as u32 * 2) as usize;
    let data = image.data.as_ref().unwrap();
    (0..channels)
        .map(|channel| {
            let bytes = &data[base + channel * 2..base + channel * 2 + 2];
            f16_bits_to_f32(u16::from_le_bytes([bytes[0], bytes[1]]))
        })
        .collect()
}

#[test]
fn bake_claims_only_texels_inside_body_shapes() {
    let river = WaterShape::River {
        path: bevy_aqua_sdf::RiverPath {
            points: vec![
                bevy_aqua_sdf::RiverPoint::new(Vec2::new(0.0, 0.0), 12.0, 2.0),
                bevy_aqua_sdf::RiverPoint::new(Vec2::new(100.0, 0.0), 12.0, 2.0),
            ],
        },
    };
    let pond = WaterShape::Circle { radius: 30.0 };
    let bodies = vec![
        ResolvedWaterBody::resolve(
            Entity::from_bits(1),
            &river,
            None,
            &GlobalTransform::IDENTITY,
        )
        .unwrap(),
        ResolvedWaterBody::resolve(
            Entity::from_bits(2),
            &pond,
            None,
            &GlobalTransform::from(Transform::from_xyz(160.0, 3.0, 60.0)),
        )
        .unwrap(),
    ];
    let (params, maps) = bake(&bodies, false);
    assert_eq!(params.meta.x, 2.0, "two bounded bodies");
    assert_eq!(params.meta.y, 0.0, "no Ocean resource");

    let size = params.region.zw();
    let texel = params.meta.z;
    let width = (((size.x / texel).ceil()) as u32).clamp(4, MAX_FIELD_SIDE);
    let height = (((size.y / texel).ceil()) as u32).clamp(4, MAX_FIELD_SIDE);

    let sample = |world_xz: Vec2| {
        let uv = (world_xz - params.region.xy()) / size;
        let column = (uv.x * width as f32) as u32;
        let row = (uv.y * height as f32) as u32;
        (
            read_texel(&maps, column, row, width, height, 0, 4)[0..2].to_vec(),
            read_texel(&maps, column, row, width, height, 1, 4),
        )
    };

    let (level, flow_sample) = sample(Vec2::new(50.0, 0.0));
    assert_eq!(level[1], 1.0, "centreline must claim river slot 1");
    assert!((level[0] - 0.0).abs() < 1e-3, "river level");
    assert!(flow_sample[2] >= 0.0, "inside the channel banks");

    let (level, _) = sample(Vec2::new(160.0, 60.0));
    assert_eq!(level[1], 2.0, "pond interior must claim slot 2");
    assert!((level[0] - 3.0).abs() < 1e-3, "pond level");

    for point in [Vec2::new(130.0, 30.0), Vec2::new(50.0, 45.0)] {
        let (level, flow_sample) = sample(point);
        assert_eq!(level[1], 0.0, "{point} must be unclaimed");
        assert!(flow_sample[2] < 0.0, "{point} must sit outside the banks");
    }
}

#[test]
fn f32_to_f16_bits_matches_known_values() {
    // Reference half-precision decoder for round-trip checks.
    fn f16_bits_to_f32(bits: u16) -> f32 {
        let sign = if bits & 0x8000 != 0 { -1.0 } else { 1.0 };
        let exponent = ((bits & 0x7C00) >> 10) as i32;
        let fraction = (bits & 0x03FF) as u32;
        if exponent == 0 {
            return sign * (fraction as f32) * 2.0_f32.powi(-24);
        }
        if exponent == 0x1F {
            return sign * f32::INFINITY;
        }
        sign * (1.0 + fraction as f32 / 1024.0) * 2.0_f32.powi(exponent - 15)
    }

    assert_eq!(f32_to_f16_bits(0.0), 0x0000);
    assert_eq!(f32_to_f16_bits(1.0), 0x3C00);
    assert_eq!(f32_to_f16_bits(-1.0), 0xBC00);
    assert_eq!(f32_to_f16_bits(2.0), 0x4000);
    // Flow magnitudes survive the packing well below visual thresholds.
    for value in [-8.0, -1.5, -0.25, 0.25, 1.5, 8.0] {
        let decoded = f16_bits_to_f32(f32_to_f16_bits(value));
        assert!(
            (decoded - value).abs() < 0.002 * value.abs().max(0.001),
            "decoded {decoded} != {value}"
        );
    }
    // Overflow clamps to half infinity rather than wrapping.
    assert_eq!(f32_to_f16_bits(1.0e6), 0x7C00);
    // Tiny magnitudes flush toward zero.
    assert_eq!(f32_to_f16_bits(1.0e-9), 0x0000);
}

#[test]
fn boundary43_real_bake_preserves_square_extent_and_corner_ownership() {
    let shape = WaterShape::Polygon {
        points: vec![
            Vec2::new(-100.0, -100.0),
            Vec2::new(100.0, -100.0),
            Vec2::new(100.0, 100.0),
            Vec2::new(-100.0, 100.0),
        ],
    };
    let body = ResolvedWaterBody::resolve(
        Entity::from_bits(1),
        &shape,
        None,
        &GlobalTransform::from(Transform::from_xyz(450.0, 0.0, 450.0)),
    )
    .unwrap();
    assert_eq!(body.aabb(), (Vec2::splat(348.0), Vec2::splat(552.0)));
    assert_eq!(body.extent(), (Vec2::splat(450.0), 102.0));
    let (params, maps) = bake(&[body], false);
    assert_eq!(params.meta.x, 1.0);
    assert_eq!(params.meta.y, 0.0);
    // Read the actual uploaded BodyParams extent, not a separately built value.
    let mut bytes = Vec::new();
    bevy::render::render_resource::encase::UniformBuffer::new(&mut bytes)
        .write(&params.bodies[0])
        .unwrap();
    let read = |offset| f32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
    assert_eq!(
        [read(16), read(20), read(24), read(28)],
        [450.0, 450.0, 0.0, 102.0]
    );
    let size = params.region.zw();
    let width = maps.texture_descriptor.size.width;
    let height = maps.texture_descriptor.size.height;
    for point in [Vec2::splat(353.0), Vec2::splat(360.0), Vec2::splat(450.0)] {
        let uv = (point - params.region.xy()) / size;
        let texel = read_texel(
            &maps,
            (uv.x * width as f32) as u32,
            (uv.y * height as f32) as u32,
            width,
            height,
            0,
            4,
        );
        assert_eq!(texel[0], 0.0, "explicit water level");
        assert_eq!(texel[1], 1.0, "corner interior has bounded owner");
    }
    assert!(Vec2::splat(450.0).length() - read(28) > 512.0);
    assert!(Vec2::splat(353.0).length() < 512.0);
}
