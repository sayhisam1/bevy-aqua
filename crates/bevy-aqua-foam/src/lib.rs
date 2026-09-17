//! Persistent foam for Aqua: a Crest-style simulation cascade (advect,
//! fade, accumulate) plus the shoreline/bank foam terms the material
//! samples through `aqua::foam::shade`.
//!
//! The sim consumes whatever the wave producers wrote this frame (checked
//! through [`bevy_aqua_core::AnimWavesStatus`]) and publishes one density array
//! per LOD that the cascade material binds as slots 7/8.

#![warn(unreachable_pub)]

use bevy::{
    asset::{RenderAssetUsages, embedded_asset},
    image::{
        CompressedImageFormats, ImageAddressMode, ImageFilterMode, ImageSampler,
        ImageSamplerDescriptor, ImageType,
    },
    prelude::*,
    render::{
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        render_resource::{
            Extent3d, ShaderType, TextureDimension, TextureFormat, TextureUsages,
            TextureViewDescriptor, TextureViewDimension,
        },
    },
};

use bevy_aqua_core::cascade as lod;
use bevy_aqua_core::{CascadeDataReady, CascadeMaterialsUpdated};
use bevy_aqua_core::{LOD_COUNT, OceanWaves, WaveModel};

mod render;

#[derive(Resource)]
struct ShaderLibraries {
    _handles: Vec<Handle<Shader>>,
}

/// Adds persistent foam simulation after the animated-wave render pass.
#[derive(Debug, Default, Clone, Copy)]
pub struct AquaFoamPlugin;

impl Plugin for AquaFoamPlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "foam.wgsl");
        // Surface module consumed by the composed cascade material.
        embedded_asset!(app, "shade.wgsl");
        // One generated import module keeps Rust allocation/dispatch sizes and
        // both foam shaders on the same compile-time contract.
        let contract = app
            .world_mut()
            .resource_mut::<Assets<Shader>>()
            .add(Shader::from_wgsl(
                FOAM_CONTRACT_WGSL,
                "bevy_aqua_foam/contract.wgsl",
            ));
        let shade = app
            .world()
            .resource::<AssetServer>()
            .load("embedded://bevy_aqua_foam/shade.wgsl");
        app.insert_resource(ShaderLibraries {
            _handles: vec![contract, shade],
        });
        app.init_resource::<Textures>()
            .add_plugins(ExtractResourcePlugin::<Frame>::default())
            .add_systems(Startup, init.after(CascadeDataReady))
            .add_systems(
                PostUpdate,
                update
                    .after(CascadeMaterialsUpdated)
                    .after(bevy_aqua_core::WaterBodiesResolved),
            );
        render::add(app);
    }
}

const SHADER_PATH: &str = "embedded://bevy_aqua_foam/foam.wgsl";
const UPDATE_FREQUENCY: f32 = 30.0;
const STEP_SECONDS: f32 = 1.0 / UPDATE_FREQUENCY;
const MAX_CATCH_UP_STEPS: u32 = 8;
const FADE_RATE: f32 = 0.8;
const WAVE_STRENGTH: f32 = 1.0;
const WAVE_COVERAGE: f32 = 0.55;
const SHORE_OUTER_DEPTH: f32 = 3.2;
const SHORE_STRENGTH: f32 = 1.6;
const SHORE_WET_EDGE_DEPTH: f32 = 0.35;
const SHORE_BREAKER_PEAK_DEPTH: f32 = 1.15;
const WORKGROUP_SIZE: u32 = 8;

macro_rules! foam_contract {
    ($resolution:literal, $pattern_resolution:literal, $lod_count:literal) => {
        pub(crate) const RESOLUTION: u32 = $resolution;
        const PATTERN_SIZE: u32 = $pattern_resolution;
        const FOAM_LOD_COUNT: u32 = $lod_count;
        const FOAM_CONTRACT_WGSL: &str = concat!(
            "#define_import_path aqua::foam::contract\n",
            "const FOAM_TEXTURE_RESOLUTION: f32 = ",
            stringify!($resolution),
            ".0;\n",
            "const FOAM_TEXTURE_RESOLUTION_U32: u32 = ",
            stringify!($resolution),
            "u;\n",
            "const FOAM_PATTERN_RESOLUTION: f32 = ",
            stringify!($pattern_resolution),
            ".0;\n",
            "const FOAM_LOD_COUNT: u32 = ",
            stringify!($lod_count),
            "u;\n",
        );
    };
}

// This invocation is the single source for Rust allocation/dispatch sizes and
// the WGSL simulation/shading constants generated above.
foam_contract!(512, 512, 5);

const _: () = assert!(RESOLUTION.is_multiple_of(WORKGROUP_SIZE));
const _: () = assert!(FOAM_LOD_COUNT as usize == LOD_COUNT);

#[doc(hidden)]
#[derive(Resource, Clone, Debug)]
pub struct Textures {
    pub state_a: Handle<Image>,
    pub state_b: Handle<Image>,
    pub surface: Handle<Image>,
    pub pattern: Handle<Image>,
}

impl FromWorld for Textures {
    fn from_world(world: &mut World) -> Self {
        let mut images = world.resource_mut::<Assets<Image>>();
        Self {
            state_a: images.add(make_state_texture()),
            state_b: images.add(make_state_texture()),
            surface: images.add(make_state_texture()),
            pattern: images.add(make_pattern_texture()),
        }
    }
}

/// Seeds the foam frame resources from the cascade layout and the shared
/// texture pool. Runs once at startup, after [`CascadeDataReady`].
#[doc(hidden)]
pub fn init(mut commands: Commands, data: Res<lod::Data>, textures: Res<Textures>) {
    let layout = data.layout().clone();
    commands.insert_resource(Frame {
        state_a: textures.state_a.clone(),
        state_b: textures.state_b.clone(),
        surface: textures.surface.clone(),
        waves: data.texture(),
        uniform: Uniform::new(layout.clone()),
        simulated_layout: layout,
    });
}

/// Refreshes per-frame sim uniforms (tick count, model gate, layouts).
#[doc(hidden)]
pub fn update(
    time: Res<Time>,
    data: Res<lod::Data>,
    waves: Res<OceanWaves>,
    mut frame: ResMut<Frame>,
) {
    let target_layout = data.layout().clone();
    frame.uniform.source_layout = frame.simulated_layout.clone();
    frame.uniform.target_layout = target_layout.clone();
    frame.uniform.step.x = target_tick(time.elapsed_secs_f64());
    frame.uniform.step.z = u32::from(waves.model == WaveModel::Spectral);
    frame.simulated_layout = target_layout;
}

fn target_tick(elapsed_seconds: f64) -> u32 {
    // The small epsilon prevents exact fixed-step boundaries represented just
    // below an integer from losing a tick.
    (elapsed_seconds
        .mul_add(UPDATE_FREQUENCY as f64, 1e-9)
        .floor() as u64)
        .min(u64::from(u32::MAX)) as u32
}

#[doc(hidden)]
#[derive(Resource, Clone, ExtractResource, Debug)]
pub struct Frame {
    state_a: Handle<Image>,
    state_b: Handle<Image>,
    surface: Handle<Image>,
    waves: Handle<Image>,
    uniform: Uniform,
    simulated_layout: lod::GpuLayout,
}

#[derive(ShaderType, Clone, Debug)]
struct Uniform {
    source_layout: lod::GpuLayout,
    target_layout: lod::GpuLayout,
    // Number of fixed steps, reserved, FFT source flag, reserved.
    step: UVec4,
    // dt, fade rate, wave strength, wave coverage.
    wave: Vec4,
    // shoreline outer depth, strength, wet-edge depth, breaker peak depth.
    shore: Vec4,
}

impl Uniform {
    fn new(layout: lod::GpuLayout) -> Self {
        Self {
            source_layout: layout.clone(),
            target_layout: layout,
            step: UVec4::new(0, 1, 0, 0),
            wave: Vec4::new(STEP_SECONDS, FADE_RATE, WAVE_STRENGTH, WAVE_COVERAGE),
            shore: Vec4::new(
                SHORE_OUTER_DEPTH,
                SHORE_STRENGTH,
                SHORE_WET_EDGE_DEPTH,
                SHORE_BREAKER_PEAK_DEPTH,
            ),
        }
    }
}

pub(crate) fn make_state_texture() -> Image {
    let bytes_per_pixel = TextureFormat::Rgba16Float
        .block_copy_size(None)
        .expect("Rgba16Float must have a fixed block size") as usize;
    let pixel_count = RESOLUTION as usize * RESOLUTION as usize * LOD_COUNT;
    let mut image = Image::new(
        Extent3d {
            width: RESOLUTION,
            height: RESOLUTION,
            depth_or_array_layers: LOD_COUNT as u32,
        },
        TextureDimension::D2,
        vec![0; pixel_count * bytes_per_pixel],
        TextureFormat::Rgba16Float,
        RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
    );
    image.texture_descriptor.usage = TextureUsages::COPY_DST
        | TextureUsages::COPY_SRC
        | TextureUsages::STORAGE_BINDING
        | TextureUsages::TEXTURE_BINDING;
    image.texture_view_descriptor = Some(TextureViewDescriptor {
        dimension: Some(TextureViewDimension::D2Array),
        ..default()
    });
    image.sampler = ImageSampler::Descriptor(ImageSamplerDescriptor {
        address_mode_u: ImageAddressMode::ClampToEdge,
        address_mode_v: ImageAddressMode::ClampToEdge,
        address_mode_w: ImageAddressMode::ClampToEdge,
        mag_filter: ImageFilterMode::Linear,
        min_filter: ImageFilterMode::Linear,
        mipmap_filter: ImageFilterMode::Nearest,
        ..default()
    });
    image
}

const SOURCE_PATTERN_SIZE: u32 = 450;
// Unity's `nPOTScale: ToNearest` imports Crest's 450px source at PATTERN_SIZE.
const PATTERN_BYTES: &[u8] = include_bytes!("../assets/Foam2.png");

fn srgb_to_linear(value: u8) -> f32 {
    let encoded = f32::from(value) / 255.0;
    if encoded <= 0.040_45 {
        encoded / 12.92
    } else {
        ((encoded + 0.055) / 1.055).powf(2.4)
    }
}

fn linear_to_srgb(value: f32) -> u8 {
    let encoded = if value <= 0.003_130_8 {
        12.92 * value
    } else {
        1.055 * value.powf(1.0 / 2.4) - 0.055
    };
    (255.0 * encoded.clamp(0.0, 1.0)).round() as u8
}

// Resample repeatable sRGB texels at pixel centres and filter RGB in linear light.
fn resize_srgb(source: &[u8], source_size: u32, target_size: u32) -> Vec<u8> {
    let scale = source_size as f32 / target_size as f32;
    (0..target_size * target_size)
        .flat_map(|index| {
            let target_x = (index % target_size) as f32;
            let target_y = (index / target_size) as f32;
            let source_x = (target_x + 0.5) * scale - 0.5;
            let source_y = (target_y + 0.5) * scale - 0.5;
            let x0 = source_x.floor() as i32;
            let y0 = source_y.floor() as i32;
            let tx = source_x - source_x.floor();
            let ty = source_y - source_y.floor();
            std::array::from_fn::<_, 4, _>(|channel| {
                let sample = |x: i32, y: i32| {
                    let wrapped_x = x.rem_euclid(source_size as i32) as u32;
                    let wrapped_y = y.rem_euclid(source_size as i32) as u32;
                    let offset = 4 * (wrapped_y * source_size + wrapped_x) as usize + channel;
                    if channel == 3 {
                        f32::from(source[offset]) / 255.0
                    } else {
                        srgb_to_linear(source[offset])
                    }
                };
                let top = sample(x0, y0).lerp(sample(x0 + 1, y0), tx);
                let bottom = sample(x0, y0 + 1).lerp(sample(x0 + 1, y0 + 1), tx);
                let filtered = top.lerp(bottom, ty);
                if channel == 3 {
                    (255.0 * filtered.clamp(0.0, 1.0)).round() as u8
                } else {
                    linear_to_srgb(filtered)
                }
            })
        })
        .collect()
}

fn make_pattern_texture() -> Image {
    let sampler = ImageSampler::Descriptor(ImageSamplerDescriptor {
        address_mode_u: ImageAddressMode::Repeat,
        address_mode_v: ImageAddressMode::Repeat,
        mag_filter: ImageFilterMode::Linear,
        min_filter: ImageFilterMode::Linear,
        mipmap_filter: ImageFilterMode::Linear,
        ..default()
    });
    let mut decoded = Image::from_buffer(
        PATTERN_BYTES,
        ImageType::Extension("png"),
        CompressedImageFormats::NONE,
        true,
        sampler,
        RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
    )
    .expect("bundled Crest Foam2.png must decode");
    assert_eq!(decoded.texture_descriptor.size.width, SOURCE_PATTERN_SIZE);
    assert_eq!(decoded.texture_descriptor.size.height, SOURCE_PATTERN_SIZE);
    assert_eq!(
        decoded.texture_descriptor.format,
        TextureFormat::Rgba8UnormSrgb
    );

    let source = decoded
        .data
        .take()
        .expect("decoded Crest foam pattern must have CPU pixels");
    let mut levels = Vec::new();
    let mut size = PATTERN_SIZE;
    let mut level = resize_srgb(&source, SOURCE_PATTERN_SIZE, PATTERN_SIZE);
    bevy_aqua_core::write_pattern_caustics(&mut level, PATTERN_SIZE);
    loop {
        levels.extend_from_slice(&level);
        if size == 1 {
            break;
        }
        let next_size = size / 2;
        level = resize_srgb(&level, size, next_size);
        size = next_size;
    }

    let mut image = Image::new_uninit(
        Extent3d {
            width: PATTERN_SIZE,
            height: PATTERN_SIZE,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
    );
    image.data = Some(levels);
    image.texture_descriptor.mip_level_count = PATTERN_SIZE.ilog2() + 1;
    image.texture_descriptor.usage = TextureUsages::COPY_DST | TextureUsages::TEXTURE_BINDING;
    image.sampler = ImageSampler::Descriptor(ImageSamplerDescriptor {
        address_mode_u: ImageAddressMode::Repeat,
        address_mode_v: ImageAddressMode::Repeat,
        mag_filter: ImageFilterMode::Linear,
        min_filter: ImageFilterMode::Linear,
        mipmap_filter: ImageFilterMode::Linear,
        ..default()
    });
    image
}

#[cfg(test)]
#[path = "foam_tests.rs"]
mod tests;
