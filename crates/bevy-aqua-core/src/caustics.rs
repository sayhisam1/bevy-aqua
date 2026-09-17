//! Periodic Voronoi ridge field packed into the foam-pattern green channel.
//!
//! Shore sampling (`aqua::shore::water`) reads that green channel. Foam
//! owns the atlas because the cascade material keeps foam breakup and
//! bed caustics in one sampled texture.

use bevy::math::{IVec2, Vec2};

/// Cells across one tile; must match `CAUSTIC_CELLS_PER_TILE` in water.wgsl.
pub const PATTERN_CAUSTIC_CELLS: i32 = 16;
// Keeps cell borders narrow without losing them under bilinear filtering.
const CAUSTIC_RIDGE_WIDTH: f32 = 0.16;

fn linear_to_srgb(value: f32) -> u8 {
    let encoded = if value <= 0.003_130_8 {
        12.92 * value
    } else {
        1.055 * value.powf(1.0 / 2.4) - 0.055
    };
    (255.0 * encoded.clamp(0.0, 1.0)).round() as u8
}

fn caustic_feature(cell: IVec2) -> Vec2 {
    let x = cell.x.rem_euclid(PATTERN_CAUSTIC_CELLS) as u32;
    let y = cell.y.rem_euclid(PATTERN_CAUSTIC_CELLS) as u32;
    // Fixed avalanche factors make the periodic texture deterministic.
    let hash = x.wrapping_mul(0x9e37_79b9) ^ y.wrapping_mul(0x85eb_ca6b);
    Vec2::new(
        (hash.wrapping_mul(0x27d4_eb2d) & 0xffff) as f32 / 65_535.0,
        (hash.rotate_left(13).wrapping_mul(0x1656_67b1) & 0xffff) as f32 / 65_535.0,
    )
}

fn caustic_ridge(point: Vec2) -> f32 {
    let cell = point.floor().as_ivec2();
    let mut nearest = [f32::MAX; 2];
    for oy in -1..=1 {
        for ox in -1..=1 {
            let neighbour = cell + IVec2::new(ox, oy);
            let distance = point.distance(neighbour.as_vec2() + caustic_feature(neighbour));
            if distance < nearest[0] {
                nearest = [distance, nearest[0]];
            } else if distance < nearest[1] {
                nearest[1] = distance;
            }
        }
    }
    (1.0 - (nearest[1] - nearest[0]) / CAUSTIC_RIDGE_WIDTH).clamp(0.0, 1.0)
}

/// sRGB green byte for one texel of the packed caustic ridge field.
pub fn caustic_green_srgb(x: u32, y: u32, size: u32) -> u8 {
    let point = Vec2::new(x as f32, y as f32) * PATTERN_CAUSTIC_CELLS as f32 / size as f32;
    let ridge = caustic_ridge(point);
    linear_to_srgb(ridge * ridge)
}

/// Writes the ridge field into the green channel of an RGBA8 sRGB image.
pub fn write_pattern_caustics(pixels: &mut [u8], size: u32) {
    for y in 0..size {
        for x in 0..size {
            pixels[4 * (y * size + x) as usize + 1] = caustic_green_srgb(x, y, size);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ridge_is_periodic_over_the_cell_tile() {
        let a = caustic_ridge(Vec2::new(1.25, 3.5));
        let b = caustic_ridge(Vec2::new(
            1.25 + PATTERN_CAUSTIC_CELLS as f32,
            3.5 + PATTERN_CAUSTIC_CELLS as f32,
        ));
        assert!((a - b).abs() < 1e-5);
    }

    #[test]
    fn write_pattern_caustics_fills_green() {
        let size = 8u32;
        let mut pixels = vec![0u8; (4 * size * size) as usize];
        write_pattern_caustics(&mut pixels, size);
        assert_eq!(
            pixels[4 * (2 * size + 3) as usize + 1],
            caustic_green_srgb(3, 2, size)
        );
    }
}
