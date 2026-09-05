//! CPU image and source-wiring tests. These do not execute WGSL on a GPU.
use super::*;

#[test]
fn caustic_image_preserves_base_and_packs_exact_area_means() {
    let image = make_caustics_texture();
    let data = image.data.as_ref().expect("CPU upload data");
    let size = CAUSTIC_TEXTURE_SIZE as usize;
    assert_eq!(image.texture_descriptor.size.width, 512);
    assert_eq!(image.texture_descriptor.size.height, 512);
    assert_eq!(image.texture_descriptor.size.depth_or_array_layers, 1);
    assert_eq!(image.texture_descriptor.dimension, TextureDimension::D2);
    assert_eq!(image.texture_descriptor.format, TextureFormat::R8Unorm);
    assert_eq!(image.texture_descriptor.mip_level_count, 10);
    assert_eq!(data.len(), 349_525);
    // Reconstruct the shipped base formula, independently of mip packing.
    let mut base = Vec::with_capacity(size * size);
    for y in 0..CAUSTIC_TEXTURE_SIZE {
        for x in 0..CAUSTIC_TEXTURE_SIZE {
            let point = Vec2::new(x as f32, y as f32) * CAUSTIC_CELL_COUNT as f32
                / CAUSTIC_TEXTURE_SIZE as f32;
            let ridge = caustic_ridge(point);
            base.push((ridge * ridge * 255.0).round() as u8);
        }
    }
    assert_eq!(&data[..size * size], base.as_slice());
    let base_mean = base.iter().map(|&v| u64::from(v)).sum::<u64>() as f64 / base.len() as f64;
    let mut offset = 0;
    for level in 0..image.texture_descriptor.mip_level_count {
        let width = size >> level;
        let area_width = 1usize << level;
        let area = (area_width * area_width) as u64;
        let bytes = &data[offset..offset + width * width];
        for y in 0..width {
            for x in 0..width {
                // Direct integer integration of base pixels, not recursive means.
                let mut sum = 0u64;
                for by in y * area_width..(y + 1) * area_width {
                    for bx in x * area_width..(x + 1) * area_width {
                        sum += u64::from(base[by * size + bx]);
                    }
                }
                assert_eq!(bytes[y * width + x], ((sum + area / 2) / area) as u8);
            }
        }
        let mean = bytes.iter().map(|&v| u64::from(v)).sum::<u64>() as f64 / bytes.len() as f64;
        // Upload rounding can change a mean by at most half an R8 code.
        assert!((mean - base_mean).abs() <= 0.5);
        offset += width * width;
    }
    assert_eq!(offset, data.len());
    let ImageSampler::Descriptor(sampler) = &image.sampler else {
        panic!("explicit trilinear sampler required");
    };
    assert_eq!(sampler.address_mode_u, ImageAddressMode::Repeat);
    assert_eq!(sampler.address_mode_v, ImageAddressMode::Repeat);
    assert_eq!(sampler.mag_filter, ImageFilterMode::Linear);
    assert_eq!(sampler.min_filter, ImageFilterMode::Linear);
    assert_eq!(sampler.mipmap_filter, ImageFilterMode::Linear);
    assert!(sampler.lod_min_clamp <= 0.0);
    assert!(sampler.lod_max_clamp >= 9.0);
}

#[test]
fn caustic_mips_never_feed_rounded_upload_values_into_next_level() {
    let mut base = [0; 16];
    base[0] = 2;
    base[2] = 2;
    base[8] = 2;
    let data = caustic_mip_chain(&base, 4);
    assert_eq!(&data[..16], &base);
    assert_eq!(&data[16..20], &[1, 1, 1, 0]);
    // True mean is 6/16, not the rounded previous level's 3/4.
    assert_eq!(data[20], 0);
    for value in [0, 1, 127, 255] {
        assert_eq!(caustic_mip_chain(&[value; 16], 4), vec![value; 21]);
    }
}

#[test]
fn caustic_lod_uses_actual_uv_coordinates_and_both_layer_scales() {
    let shore = include_str!("water.wgsl");
    let material = include_str!("../../bevy-aqua-core/src/cascade/material.wgsl");
    let optics = include_str!("../../bevy-aqua-optics/src/optics.wgsl");
    let cascade = include_str!("../../bevy-aqua-core/src/cascade.rs");
    let fragment = material.split("fn fragment(").nth(1).unwrap();
    let cached = "set_caustic_xz_footprint(max(\n        length(dpdx(in.undisplaced_xz)),\n        length(dpdy(in.undisplaced_xz)),\n    ));";
    assert!(fragment.contains("set_xz_footprint(max(\n        length(dpdx(in.world_position.xz)),\n        length(dpdy(in.world_position.xz)),\n    ));"));
    assert!(fragment.contains(cached));
    assert!(fragment.find(cached).unwrap() < fragment.find("if ").unwrap());
    assert!(optics.contains(
        "return caustic_bed_radiance(\n        scene_colour,\n        in.undisplaced_xz,"
    ));
    assert!(shore.contains("let scale = surface.caustics.y * CAUSTIC_CELLS_PER_TILE;"));
    assert!(shore.contains("let uv_a = world_xz / scale"));
    assert!(shore.contains("let uv_b = world_xz * 1.37 / scale"));
    assert!(shore.contains("let texels_per_pixel = caustic_screen_xz_footprint()\n        * f32(textureDimensions(caustics_texture, 0).x) / scale;"));
    assert!(shore.contains("let maximum_lod = f32(textureNumLevels(caustics_texture) - 1u);"));
    assert!(
        shore.contains("let lod_a = clamp(log2(max(texels_per_pixel, 1.0)), 0.0, maximum_lod);")
    );
    assert!(
        shore.contains(
            "let lod_b = clamp(log2(max(texels_per_pixel * 1.37, 1.0)), 0.0, maximum_lod);"
        )
    );
    assert!(
        shore.contains("textureSampleLevel(caustics_texture, caustics_sampler, uv_a, lod_a).r")
    );
    assert!(
        shore.contains("textureSampleLevel(caustics_texture, caustics_sampler, uv_b, lod_b).r")
    );
    assert!(shore.contains("let pattern = CAUSTIC_FOCUS_GAIN * a * b;"));
    assert!(!shore.contains("dpdx(") && !shore.contains("dpdy("));
    assert!(cascade.contains("caustics.scale.max(0.01)"));
}
