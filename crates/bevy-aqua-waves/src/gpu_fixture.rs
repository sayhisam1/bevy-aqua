//! Private headless setup and byte copies. No replacement shader or sampling math.

use bevy::{
    app::PluginsState,
    camera::{CameraPlugin, RenderTarget},
    core_pipeline::CorePipelinePlugin,
    light::LightPlugin,
    mesh::MeshPlugin,
    pbr::PbrPlugin,
    prelude::*,
    render::{
        RenderApp, RenderPlugin,
        extract_resource::ExtractResourcePlugin,
        render_asset::RenderAssets,
        render_resource::*,
        renderer::{RenderDevice, RenderQueue},
        texture::GpuImage,
    },
    time::TimeUpdateStrategy,
    window::ExitCondition,
};
use bevy_aqua_core::{
    BedHeightMap, CascadeMaterial, CascadeMaterialsUpdated, Ocean, OceanView, OceanWaves,
    ResolvedWaterBodies, WaveModel, bed, cascade,
};
use std::{
    sync::mpsc,
    time::{Duration, Instant},
};

pub(super) fn app() -> App {
    let mut app = App::new();
    // No window, winit runner, or pipelined render thread, even if workspace
    // feature unification enables them. Each update completes one render frame.
    app.add_plugins((
        MinimalPlugins,
        TransformPlugin,
        WindowPlugin {
            primary_window: None,
            exit_condition: ExitCondition::DontExit,
            ..default()
        },
        AssetPlugin::default(),
        RenderPlugin::default(),
        ImagePlugin::default(),
        MeshPlugin,
        CameraPlugin,
        LightPlugin,
        CorePipelinePlugin,
        PbrPlugin::default(),
    ));
    app.insert_resource(TimeUpdateStrategy::ManualDuration(Duration::ZERO));
    app.init_asset::<CascadeMaterial>();
    app.init_resource::<ResolvedWaterBodies>();
    app.insert_resource(Ocean::default());
    app.insert_resource(OceanWaves {
        model: WaveModel::Spectral,
        ..default()
    });
    let mut images = app.world_mut().resource_mut::<Assets<Image>>();
    let waves = images.add(cascade::make_texture());
    let surface = images.add(cascade::make_fft_surface_texture());
    app.insert_resource(cascade::Data::new(
        Handle::default(),
        waves,
        surface,
        cascade::GpuLayout::new(&cascade::layout(Vec2::ZERO), Vec2::ZERO, 0.0),
    ));
    bevy_aqua_core::add_shader(&mut app);
    bed::add(&mut app);
    app.add_plugins(ExtractResourcePlugin::<cascade::Data>::default());
    app.add_systems(PostUpdate, publish_layout.in_set(CascadeMaterialsUpdated));
    app
}

fn publish_layout(mut data: ResMut<cascade::Data>, bed: Option<Res<BedHeightMap>>) {
    // The actual production constructor/decode packing supplies the fixture's
    // input. Root material synchronization/culling is deliberately outside scope.
    let mut layout = data.layout().clone();
    layout.set_bed(bed.as_deref(), 0.0);
    *data = cascade::Data::new(data.material(), data.texture(), data.fft_surface(), layout);
}

pub(super) fn finish(app: &mut App) {
    // The wave plugin registers OceanView's required extraction components.
    // Spawn only after those plugins were added, before the first update.
    let target = app
        .world_mut()
        .resource_mut::<Assets<Image>>()
        .add(Image::new_target_texture(
            32,
            32,
            TextureFormat::Rgba8UnormSrgb,
            None,
        ));
    app.world_mut().spawn((
        Camera3d::default(),
        RenderTarget::Image(target.into()),
        Transform::from_xyz(0.0, 10.0, 10.0).looking_at(Vec3::ZERO, Vec3::Y),
        Msaa::Off,
        OceanView,
    ));
    let start = Instant::now();
    while app.plugins_state() != PluginsState::Ready {
        assert!(
            start.elapsed() < Duration::from_secs(30),
            "renderer startup timed out"
        );
        std::thread::yield_now();
    }
    app.finish();
    app.cleanup();
}

pub(super) fn step(app: &mut App) {
    app.update();
    let world = app.sub_app(RenderApp).world();
    world
        .resource::<RenderDevice>()
        .poll(PollType::Wait {
            submission_index: None,
            timeout: Some(Duration::from_secs(5)),
        })
        .expect("GPU frame did not complete within five seconds");
}

pub(super) fn beds(app: &mut App) -> (BedHeightMap, BedHeightMap) {
    let mut images = app.world_mut().resource_mut::<Assets<Image>>();
    let a = BedHeightMap::from_height_fn(
        &mut images,
        |x, _| if x < 0.0 { -0.6 } else { -5.0 },
        64,
        Vec2::splat(-512.0),
        1024.0 / 63.0,
    );
    let b = BedHeightMap::from_height_fn(
        &mut images,
        |x, _| if x < 0.0 { -8.0 } else { -1.2 },
        64,
        Vec2::splat(-512.0),
        1024.0 / 63.0,
    );
    (a, b)
}

pub(super) fn pixels(world: &World, image: &Handle<Image>, layers: u32, label: &str) -> Vec<u8> {
    let gpu = world
        .resource::<RenderAssets<GpuImage>>()
        .get(image)
        .expect("readback image prepared");
    assert_eq!(gpu.texture_descriptor.format, TextureFormat::Rgba16Float);
    let size = gpu.texture_descriptor.size;
    assert!(layers <= size.depth_or_array_layers);
    let row = size.width * 8;
    let stride = row.div_ceil(256) * 256;
    let buffer_size = u64::from(stride) * u64::from(size.height) * u64::from(layers);
    let device = world.resource::<RenderDevice>();
    let queue = world.resource::<RenderQueue>();
    let buffer = device.create_buffer(&BufferDescriptor {
        label: Some("Aqua GPU regression readback"),
        size: buffer_size,
        usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor::default());
    encoder.copy_texture_to_buffer(
        TexelCopyTextureInfo {
            texture: &gpu.texture,
            mip_level: 0,
            origin: Origin3d::ZERO,
            aspect: TextureAspect::All,
        },
        TexelCopyBufferInfo {
            buffer: &buffer,
            layout: TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(stride),
                rows_per_image: Some(size.height),
            },
        },
        Extent3d {
            depth_or_array_layers: layers,
            ..size
        },
    );
    queue.submit([encoder.finish()]);
    let (tx, rx) = mpsc::channel();
    buffer.slice(..).map_async(MapMode::Read, move |result| {
        tx.send(result).unwrap();
    });
    device
        .poll(PollType::Wait {
            submission_index: None,
            timeout: Some(Duration::from_secs(5)),
        })
        .expect("GPU readback did not complete within five seconds");
    rx.recv_timeout(Duration::from_secs(5))
        .expect("map callback")
        .expect("map success");
    let mapped = buffer.slice(..).get_mapped_range();
    let bytes: Vec<u8> = mapped
        .chunks_exact(stride as usize)
        .flat_map(|padded| padded[..row as usize].iter().copied())
        .collect();
    drop(mapped);
    buffer.unmap();
    eprintln!(
        "{label}: {}x{}x{layers} rgba16float mip0, {} packed bytes",
        size.width,
        size.height,
        bytes.len()
    );
    if let Some(directory) = std::env::var_os("AQUA_BED_GPU_ARTIFACTS") {
        let directory = std::path::PathBuf::from(directory);
        assert!(
            directory.is_absolute(),
            "artifact directory must be absolute"
        );
        std::fs::create_dir_all(&directory).unwrap();
        std::fs::write(directory.join(format!("{label}.rgba16float")), &bytes).unwrap();
    }
    bytes
}

pub(super) fn finite(bytes: &[u8]) {
    assert!(
        bytes
            .chunks_exact(2)
            .all(|word| { u16::from_le_bytes([word[0], word[1]]) & 0x7c00 != 0x7c00 }),
        "non-finite half-float in production GPU output"
    );
}

pub(super) fn adapter(app: &App) {
    let adapter = app
        .sub_app(RenderApp)
        .world()
        .resource::<bevy::render::renderer::RenderAdapterInfo>();
    eprintln!(
        "GPU adapter: {} ({:?}, {:?})",
        adapter.name, adapter.backend, adapter.device_type
    );
}
