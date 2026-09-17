//! LodData cascade layout, texture storage, and visualization.

use bevy::{
    asset::{AssetPath, RenderAssetUsages, embedded_asset, embedded_path},
    ecs::system::SystemParam,
    image::{
        CompressedImageFormats, ImageAddressMode, ImageFilterMode, ImageSampler,
        ImageSamplerDescriptor, ImageType,
    },
    mesh::MeshVertexBufferLayoutRef,
    pbr::{MaterialPipeline, MaterialPipelineKey},
    prelude::*,
    render::render_resource::{
        AsBindGroup, Extent3d, RenderPipelineDescriptor, ShaderType, SpecializedMeshPipelineError,
        TextureDimension, TextureFormat, TextureUsages, TextureViewDescriptor,
        TextureViewDimension,
    },
    shader::ShaderRef,
};

use crate::fields::FieldParams;
use crate::{
    AquaDebug, AquaSettings, OceanWaves, ViewDetail, ViewPos, ViewSeaLevel, WaterOptics, bed,
};

pub use bevy_aqua_geom::{LOD_COUNT, TILE_RESOLUTION};

/// World-space scale of the finest ring in metres.
pub const BASE_SCALE: f32 = 24.0;
const LOD_SCALE_MULTIPLIER: f32 = 2.0;

/// Cascade texture side length in texels; AnimWaves, FFT, and the packed
/// surface array share this width and height. Foam keeps its own resolution.
pub const RESOLUTION: u32 = 256;
/// First array layer of the cached FFT normal-cross / Jacobian in the packed
/// cascade texture. Displacement occupies `[0, LOD_COUNT)`.
pub const SURFACE_LAYER_BASE: u32 = LOD_COUNT as u32;
/// Displacement plus FFT surface, one layer per LOD each.
pub const CASCADE_LAYER_COUNT: u32 = 2 * LOD_COUNT as u32;
const COVERAGE_MULTIPLIER: f32 = 4.0;
const CASCADE_COUNT: usize = LOD_COUNT + 1;
const MAX_WAVELENGTH_TEXELS: f32 = 4.0;
// Shader ABI: keep these names and integer values aligned with cascade.wgsl.
const DEBUG_MODE_WATER_PATH: f32 = 1.0;
const DEBUG_MODE_REFRACTION_VALIDITY: f32 = 2.0;
const DEBUG_MODE_TRANSMISSION: f32 = 3.0;
const DEBUG_MODE_UNREFRACTED: f32 = 4.0;
const DEBUG_MODE_BEER_LAMBERT: f32 = 5.0;
const DEBUG_MODE_SEA_FLOOR: f32 = 6.0;
const DEBUG_MODE_BEAUTY: f32 = 7.0;
const DEBUG_MODE_REFLECTION: f32 = 8.0;
const DEBUG_MODE_FOAM: f32 = 9.0;
const DEBUG_MODE_WAVE_HEIGHT: f32 = 10.0;
const DEBUG_MODE_LIGHT_RADIANCE: f32 = 11.0;
const DEBUG_MODE_REFLECTION_FRACTION: f32 = 12.0;
const DEBUG_MODE_FAR_TIER: f32 = 13.0;

/// One cascade ring's transform: world centre, edge-to-edge scale, and
/// metres per texture texel.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Cascade {
    /// World-XZ centre of this cascade this frame.
    pub center: Vec2,
    /// World-space side length of the covered square in metres.
    pub scale: f32,
    /// World metres per texel.
    pub texel_width: f32,
}

/// The shared cascade resources one participant inserts at startup and
/// everyone else reads: the material handle, the packed cascade texture,
/// and the live layout. Assembled by umbrella glue because it pulls foam
/// textures from a feature resource. Extracted into the render app so
/// render-side consumers see the same handles.
#[derive(Resource, Debug, Clone, bevy::render::extract_resource::ExtractResource)]
pub struct Data {
    material: Handle<CascadeMaterial>,
    texture: Handle<Image>,
    layout: GpuLayout,
}

impl Data {
    /// Assembles the shared data from its parts.
    pub fn new(
        material: Handle<CascadeMaterial>,
        texture: Handle<Image>,
        layout: GpuLayout,
    ) -> Self {
        Self {
            material,
            texture,
            layout,
        }
    }

    pub fn material(&self) -> Handle<CascadeMaterial> {
        self.material.clone()
    }

    pub fn texture(&self) -> Handle<Image> {
        self.texture.clone()
    }

    pub fn layout(&self) -> &GpuLayout {
        &self.layout
    }
}

/// The ocean surface material: every cascade, wave, foam, field, and bed
/// binding in one fixed ABI. Slot numbers are the contract; the WGSL side
/// (`cascade/common.wgsl`) mirrors them exactly.
#[derive(Asset, TypePath, AsBindGroup, Debug, Clone)]
pub struct CascadeMaterial {
    #[texture(0, dimension = "2d_array")]
    #[sampler(1)]
    pub texture: Handle<Image>,
    #[uniform(2)]
    pub layout: GpuLayout,
    #[uniform(3)]
    pub surface: SurfaceParams,
    #[texture(4, dimension = "2d")]
    pub sea_floor: Handle<Image>,
    #[texture(5)]
    #[sampler(6)]
    pub detail_normal: Handle<Image>,
    #[texture(7, dimension = "2d_array")]
    #[sampler(8)]
    pub foam: Handle<Image>,
    /// Foam breakup in red, bed caustics in green. One 2D texture so the
    /// cascade material stays inside WebGPU's fragment sampled-texture limit.
    #[texture(9)]
    #[sampler(10)]
    pub foam_pattern: Handle<Image>,
    /// Global water fields: region uniform, per-slot parameters,
    /// level+slot map, and per-texel flow.
    #[uniform(15)]
    pub fields: FieldParams,
    /// Packed field maps: layer 0 is level+slot, layer 1 is river flow.
    #[texture(16, dimension = "2d_array")]
    #[sampler(17)]
    pub field_maps: Handle<Image>,
    #[texture(19)]
    #[sampler(20)]
    pub reflection_a: Handle<Image>,
    /// Second mirrored scene target for the next-nearest visible level.
    #[texture(21)]
    pub reflection_b: Handle<Image>,
    #[uniform(22)]
    pub reflections: PlanarReflectionParams,
}

/// One horizontal water level's mirrored view transform.
#[derive(ShaderType, Debug, Clone, Copy, PartialEq)]
pub struct PlanarReflectionView {
    pub view_projection: Mat4,
    pub level: f32,
}

#[derive(ShaderType, Debug, Clone, Copy, PartialEq)]
pub struct PlanarReflectionParams {
    pub views: [PlanarReflectionView; 2],
    pub view_count: u32,
    pub distortion: f32,
}

impl Default for PlanarReflectionParams {
    fn default() -> Self {
        Self {
            views: [PlanarReflectionView {
                view_projection: Mat4::IDENTITY,
                level: 0.0,
            }; 2],
            view_count: 0,
            distortion: 0.0,
        }
    }
}

impl Material for CascadeMaterial {
    fn vertex_shader() -> ShaderRef {
        shader_ref()
    }

    fn fragment_shader() -> ShaderRef {
        shader_ref()
    }

    fn reads_view_transmission_texture(&self) -> bool {
        true
    }

    fn specialize(
        _pipeline: &MaterialPipeline,
        descriptor: &mut RenderPipelineDescriptor,
        _layout: &MeshVertexBufferLayoutRef,
        _key: MaterialPipelineKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        descriptor.primitive.cull_mode = None;
        Ok(())
    }
}

/// Per-body extent parameters for localized water. Ocean tiles carry the
/// default (bounded off).
#[derive(ShaderType, Debug, Clone, Copy, PartialEq)]
pub struct BodyParams {
    /// x: 1.0 when the mesh is a bounded body (no camera snap/morph;
    /// fragment discard active), 0.0 for ocean tiles. y: 1.0 when the body
    /// binds a flow texture; zw reserved.
    pub(crate) flags: Vec4,
    /// xy: world-XZ centre; z: reserved; w: conservative radius in metres.
    extent: Vec4,
    /// xy: world-XZ AABB minimum of the flow-texture domain; zw reserved.
    aabb_min: Vec4,
    /// xy: world-XZ AABB size of the flow-texture domain; zw reserved.
    aabb_size: Vec4,
    /// rgb: per-channel Beer-Lambert extinction (1/m) replacing the ocean
    /// profile; w: optics enable flag. Fresh-water bodies author low
    /// extinction so the bed shows through.
    optics_a: Vec4,
    /// x: scatter-scale for particle σs; y: sun roughness; z: plain Schlick
    /// flag; w: Henyey-Greenstein `g`.
    optics_b: Vec4,
    /// rgb: particle scatter chromaticity; w reserved.
    optics_c: Vec4,
}

impl BodyParams {
    /// The inactive ocean default used by unclaimed slots.
    pub const fn ocean() -> Self {
        Self {
            flags: Vec4::ZERO,
            extent: Vec4::ZERO,
            aabb_min: Vec4::ZERO,
            aabb_size: Vec4::ZERO,
            optics_a: Vec4::ZERO,
            optics_b: Vec4::ZERO,
            optics_c: Vec4::ZERO,
        }
    }

    /// The bounded-body parameters for one extent.
    pub const fn bounded(
        center: Vec2,
        radius: f32,
        aabb_min: Vec2,
        aabb_size: Vec2,
        has_flow: bool,
        optics: Option<BodyOptics>,
    ) -> Self {
        // Body Fresnel is plain Schlick (no roughness damping); the ocean
        // preset keeps its damped curve.
        let (extinction, scale, tint, roughness, schlick, g, enabled) = match optics {
            Some(optics) => (
                optics.extinction,
                optics.scatter_scale,
                optics.scatter_tint,
                optics.sun_roughness,
                1.0,
                optics.scattering_asymmetry,
                1.0,
            ),
            None => (Vec3::ZERO, 1.0, Vec3::ZERO, -1.0, 0.0, 0.0, 0.0),
        };
        Self {
            flags: Vec4::new(1.0, if has_flow { 1.0 } else { 0.0 }, 0.0, 0.0),
            extent: Vec4::new(center.x, center.y, 0.0, radius),
            aabb_min: Vec4::new(aabb_min.x, aabb_min.y, 0.0, 0.0),
            aabb_size: Vec4::new(aabb_size.x, aabb_size.y, 0.0, 0.0),
            optics_a: Vec4::new(extinction.x, extinction.y, extinction.z, enabled),
            optics_b: Vec4::new(scale, roughness, schlick, g),
            optics_c: Vec4::new(tint.x, tint.y, tint.z, 0.0),
        }
    }
}

/// Per-body water optics: low extinction keeps shallow fresh water clear
/// over visible beds; scatter scale is particle load for the shared medium.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BodyOptics {
    /// Per-channel Beer-Lambert extinction in inverse metres.
    pub extinction: Vec3,
    /// Multiplier on particle scatter for the shared water medium.
    pub scatter_scale: f32,
    /// Per-channel particle scatter chromaticity for the shared water medium.
    pub scatter_tint: Vec3,
    /// Henyey-Greenstein `g` for the shared water medium.
    pub scattering_asymmetry: f32,
    /// Surface roughness driving the Fresnel response; negative inherits
    /// the ocean value.
    pub sun_roughness: f32,
}

/// Surface shading parameters uploaded with the material: Fresnel,
/// reflection, sun, debug routing, and advection. Mirrors `SurfaceParams`
/// in cascade/common.wgsl field for field.
#[derive(ShaderType, Debug, Clone, Copy, PartialEq)]
pub struct SurfaceParams {
    /// x: water F0, y: Godot Fresnel power, z: specular strength, w reserved.
    pub fresnel: Vec4,
    /// x: FFT flag, y: micro-roughness strength, z: daylight lux,
    /// w: maximum roughness.
    pub reflection: Vec4,
    /// x: direct-sun strength, y: GGX roughness floor,
    /// z: filter raw sunlight through the atmosphere; w reserved.
    pub sun: Vec4,
    /// x: mode, y: shader-property refraction strength, z: debug range,
    /// w: bilinear-foam diagnostic flag.
    pub debug: Vec4,
    /// rgb: ocean Beer-Lambert extinction per channel; w: particle scatter scale.
    pub fog_density: Vec4,
    /// rgb: particle scatter chromaticity; w reserved.
    pub scatter_tint: Vec4,
    /// x: maximum sampled depth; y: debug range; z: waterline fade depth;
    /// w: direct-sun visibility. Depths are metres.
    pub sea_floor: Vec4,
    /// Sunlit subsurface scattering tint (rgb); w reserved.
    pub sss_tint: Vec4,
    /// SSS pedestal, strength, and range; w reserved.
    pub sss: Vec4,
    /// Detail normals: scale, strength, and overall strength; w reserved.
    pub detail: Vec4,
    /// Capillary ripples: frequency ratio, slope strength, resolved-fade
    /// start/end in metres.
    pub capillary: Vec4,
    /// Foam scale, feather, and lighting strength; w reserved.
    pub foam: Vec4,
    /// xy: world-space current in m/s; zw reserved. The shader advects wave
    /// sampling by `flow * globals.time`.
    pub advection: Vec4,
    /// x: far-tier transition start in metres, y: end;
    /// z reserved; w: Henyey-Greenstein `g`.
    pub far_tier: Vec4,
    /// Strength, metres per cell, metres per second, and maximum depth in metres.
    pub caustics: Vec4,
}

impl SurfaceParams {
    /// Applies one optics preset's extinction, particle scatter, and crest SSS
    /// tint to the uniform.
    pub fn apply_optics(&mut self, optics: &WaterOptics) {
        self.fog_density = optics.extinction.extend(optics.scatter_scale.max(0.0));
        self.scatter_tint = optics.scatter_tint.max(Vec3::ZERO).extend(0.0);
        self.sss_tint = optics.sss_tint.extend(0.0);
    }
}

impl Default for SurfaceParams {
    fn default() -> Self {
        Self {
            // Water F0, Godot Fresnel power, shipped Crest specular strength, reserved.
            fresnel: Vec4::new(0.020_373_19, 5.0, 1.0, 0.0),
            // FFT flag, micro-roughness strength, daylight lux, maximum roughness.
            reflection: Vec4::new(0.0, 1.0, 10_000.0, 0.28),
            // Godot direct-sun strength/GGX floor, atmospheric sunlight filter, reserved.
            sun: Vec4::new(1.0, 0.4, 0.0, 1.0),
            // Mode, shader-property refraction (`Ocean.shader:148`), debug range, reserved.
            // Shipped `Ocean.mat:145` uses strength 1.0; Aqua retains 0.5 for the accepted view.
            debug: Vec4::new(0.0, 0.5, 32.0, 0.0),
            // Shader-property extinction (`Ocean.shader:146`); Ocean.mat:185 differs.
            fog_density: Vec4::new(0.9, 0.3, 0.35, 0.2),
            scatter_tint: Vec4::new(0.85, 1.0, 1.22, 0.0),
            // Maximum depth, debug range, waterline fade, direct-sun visibility.
            sea_floor: Vec4::new(32.0, 10.0, 1.0, 0.0),
            // Shader-property SSS (`Ocean.shader:48,50-54`); Ocean.mat:156,164-165,195 differs.
            sss_tint: Vec4::new(0.088_506_84, 0.497, 0.456_150_74, 0.0),
            sss: Vec4::new(0.0, 1.7, 5.0, 1.0),
            // Crest normal-map scale, strength, overall strength, and reserved.
            detail: Vec4::new(40.0, 0.08, 1.0, 1.0),
            // Frequency ratio, slope strength, resolved-fade start/end in metres.
            capillary: Vec4::new(16.0, 0.08, 30.0, 50.0),
            // Shader-default foam scale (`Ocean.shader:114`), shipped feather/light, reserved.
            // `Ocean.mat:128,176-177` instead stores 5.0, 0.4, and 1.353.
            foam: Vec4::new(10.0, 0.4, 1.35, 1.0),
            // No current by default; the accepted goldens stay world-anchored.
            advection: Vec4::ZERO,
            far_tier: Vec4::new(320.0, 512.0, 0.0, 0.8),
            caustics: Vec4::ZERO,
        }
    }
}

#[derive(ShaderType, Debug, Default, Clone, Copy, PartialEq)]
/// The per-cascade uniform block: transform plus derived constants, as
/// uploaded to the shader (`CascadeParams` in waves_sample.wgsl).
pub struct GpuCascade {
    /// World-XZ centre of this cascade this frame.
    pub center: Vec2,
    /// World-space side length of the covered square in metres.
    pub scale: f32,
    /// Texture side length in texels.
    pub texture_res: f32,
    /// Reciprocal of `texture_res`.
    pub inv_texture_res: f32,
    /// World metres per texel.
    pub texel_width: f32,
    /// Blend weight (0 weights out the sentinel last slot).
    pub weight: f32,
    /// Longest wavelength this cascade resolves in metres.
    pub max_wavelength: f32,
}

#[derive(ShaderType, Debug, Clone)]
/// The cascade layout uniform (`CascadeLayout` in waves_sample.wgsl): the ring
/// stack plus camera/bed mapping.
pub struct GpuLayout {
    /// Per-cascade parameters; the last slot is a zero-weighted sentinel
    /// copy so shader reads never index out of bounds.
    pub cascades: [GpuCascade; CASCADE_COUNT],
    /// XY camera centre, Z detail LOD, W number of sampled LOD slices.
    pub center: Vec4,
    /// XY bed-map first-texel world origin, ZW inverse world extent.
    pub bed_transform: Vec4,
    /// X height minimum, Y height span, Z sea level. A negative Y span marks
    /// "no bed map": every shader sample takes the deep-water path.
    pub bed_range: Vec4,
}

impl GpuLayout {
    /// Builds the layout for a camera position and detail LOD.
    pub fn new(cascades: &[Cascade; LOD_COUNT], center: Vec2, detail_lod: f32) -> Self {
        let mut gpu = [GpuCascade::default(); CASCADE_COUNT];
        for (target, source) in gpu.iter_mut().zip(cascades) {
            *target = GpuCascade {
                center: source.center,
                scale: source.scale,
                texture_res: RESOLUTION as f32,
                inv_texture_res: (RESOLUTION as f32).recip(),
                texel_width: source.texel_width,
                weight: 1.0,
                max_wavelength: MAX_WAVELENGTH_TEXELS * source.texel_width,
            };
        }
        gpu[LOD_COUNT] = gpu[LOD_COUNT - 1];
        gpu[LOD_COUNT].weight = 0.0;
        Self {
            cascades: gpu,
            center: center.extend(detail_lod).extend(LOD_COUNT as f32),
            bed_transform: Vec4::ZERO,
            bed_range: Vec4::new(0.0, bed::NO_BED_SPAN, 0.0, 0.0),
        }
    }

    /// Publishes the bed map (when the game supplied one) and the sea level
    /// into the shared shader uniform.
    pub fn set_bed(&mut self, bed: Option<&bed::BedHeightMap>, sea_level: f32) {
        match bed {
            Some(map) => {
                self.bed_transform = map
                    .origin
                    .extend(map.size.x.max(f32::MIN_POSITIVE).recip())
                    .extend(map.size.y.max(f32::MIN_POSITIVE).recip());
                self.bed_range = Vec4::new(
                    map.height_range[0],
                    (map.height_range[1] - map.height_range[0]).max(f32::MIN_POSITIVE),
                    sea_level,
                    0.0,
                );
            }
            None => {
                self.bed_transform = Vec4::ZERO;
                self.bed_range = Vec4::new(0.0, bed::NO_BED_SPAN, sea_level, 0.0);
            }
        }
    }
}

fn shader_ref() -> ShaderRef {
    ShaderRef::Path(
        AssetPath::from_path_buf(embedded_path!("cascade/material.wgsl")).with_source("embedded"),
    )
}

#[derive(Resource)]
struct ShaderLibraries {
    _handles: Vec<Handle<Shader>>,
}

/// Embeds the cascade WGSL modules and retains the import-only modules.
/// Call once from the umbrella plugin before any pipeline loads.
pub fn add_shader(app: &mut App) {
    embedded_asset!(app, "cascade/common.wgsl");
    embedded_asset!(app, "cascade/river.wgsl");
    embedded_asset!(app, "cascade/waves_sample.wgsl");
    embedded_asset!(app, "cascade/deform.wgsl");
    embedded_asset!(app, "cascade/types.wgsl");
    embedded_asset!(app, "cascade/material.wgsl");
    let server = app.world().resource::<AssetServer>();
    let handles = vec![
        server.load("embedded://bevy_aqua_core/cascade/common.wgsl"),
        server.load("embedded://bevy_aqua_core/cascade/river.wgsl"),
        server.load("embedded://bevy_aqua_core/cascade/waves_sample.wgsl"),
        server.load("embedded://bevy_aqua_core/cascade/deform.wgsl"),
        server.load("embedded://bevy_aqua_core/cascade/types.wgsl"),
    ];
    app.insert_resource(ShaderLibraries { _handles: handles });
    bevy_aqua_optics::add_shader(app);
}

/// Inputs used to refresh the material each frame.
#[derive(SystemParam, Debug)]
pub struct UpdateInputs<'w> {
    pub view: Res<'w, ViewPos>,
    pub detail: Res<'w, ViewDetail>,
    pub debug: Res<'w, AquaDebug>,
    pub settings: Res<'w, AquaSettings>,
    pub caustic_sun: Res<'w, crate::CausticsSunVisibility>,
    pub waves: Res<'w, OceanWaves>,
    pub sea_level: Res<'w, ViewSeaLevel>,
    pub bed: Option<Res<'w, crate::bed::BedHeightMap>>,
}

/// Refreshes the material's layout and surface uniforms from the shared
/// config resources when any of them change.
pub fn update(
    inputs: UpdateInputs,
    mut data: ResMut<Data>,
    mut materials: ResMut<Assets<CascadeMaterial>>,
) {
    let UpdateInputs {
        view,
        detail,
        debug,
        settings,
        caustic_sun,
        waves,
        sea_level,
        bed,
    } = inputs;
    if !view.is_changed()
        && !detail.is_changed()
        && !debug.is_changed()
        && !settings.is_changed()
        && !caustic_sun.is_changed()
        && !waves.is_changed()
        && !sea_level.is_changed()
        && !bed
            .as_ref()
            .map(bevy::ecs::change_detection::DetectChanges::is_changed)
            .unwrap_or(false)
    {
        return;
    }
    let mut layout = GpuLayout::new(&layout(view.0), view.0, detail.0);
    layout.set_bed(bed.as_deref(), sea_level.0);
    data.layout = layout.clone();
    let apply_globals = |material: &mut CascadeMaterial| {
        material.surface.apply_optics(&settings.water_optics);
        material.surface.debug.x = match *debug {
            AquaDebug::Shaded | AquaDebug::ShallowComposite => DEBUG_MODE_BEAUTY,
            AquaDebug::ReflectionSanity => DEBUG_MODE_REFLECTION,
            AquaDebug::FoamDensity | AquaDebug::FoamDensityBilinear => DEBUG_MODE_FOAM,
            AquaDebug::WaveHeight => DEBUG_MODE_WAVE_HEIGHT,
            AquaDebug::LightRadiance => DEBUG_MODE_LIGHT_RADIANCE,
            AquaDebug::ReflectionFraction => DEBUG_MODE_REFLECTION_FRACTION,
            AquaDebug::FarTier => DEBUG_MODE_FAR_TIER,
            AquaDebug::WaterPath => DEBUG_MODE_WATER_PATH,
            AquaDebug::RefractionValidity => DEBUG_MODE_REFRACTION_VALIDITY,
            AquaDebug::Transmission => DEBUG_MODE_TRANSMISSION,
            AquaDebug::TransmissionUnrefracted => DEBUG_MODE_UNREFRACTED,
            AquaDebug::BeerLambert => DEBUG_MODE_BEER_LAMBERT,
            AquaDebug::SeaFloorDepth => DEBUG_MODE_SEA_FLOOR,
        };
        material.surface.debug.w = if *debug == AquaDebug::FoamDensityBilinear {
            1.0
        } else {
            0.0
        };
        material.surface.reflection.x = if waves.model == crate::WaveModel::Spectral {
            1.0
        } else {
            0.0
        };
        let detail = settings.detail_strength.clamp(0.0, 2.0);
        material.surface.reflection.y = (detail * 12.5).clamp(0.0, 2.0);
        material.surface.sun.z = if settings.atmospheric_sunlight {
            1.0
        } else {
            0.0
        };
        material.surface.detail.y = detail;
        material.surface.capillary.y = detail.min(0.5);
        material.surface.advection = Vec4::new(waves.flow.x, waves.flow.y, 0.0, 0.0);
        let far_start = settings.far_tier_start.max(0.0);
        let far_end = settings.far_tier_end.max(far_start + 1.0);
        material.surface.far_tier = Vec4::new(
            far_start,
            far_end,
            0.0,
            settings.water_optics.scattering_asymmetry,
        );
        material.surface.sea_floor.w = caustic_sun.0.clamp(0.0, 1.0);
        material.surface.caustics = settings.caustics.map_or(Vec4::ZERO, |caustics| {
            Vec4::new(
                caustics.strength.max(0.0),
                caustics.scale.max(0.01),
                caustics.speed,
                caustics.depth_max.max(0.0),
            )
        });
    };
    {
        let mut material = materials
            .get_mut(&data.material)
            .expect("Aqua's cascade material must remain loaded");
        material.layout = layout.clone();
        if settings.caustics.is_some()
            && let Some(bed) = bed.as_deref()
        {
            material.sea_floor = bed.image.clone();
        }
        apply_globals(&mut material);
    }
}

/// Builds the camera-centred, power-of-two cascade stack.
///
/// Reimplementation of the approach in Crest `Scripts/LodData/LodTransform.cs`. Each slice covers four
/// times its LOD scale and its centre snaps down to whole texture texels.
pub fn layout(camera: Vec2) -> [Cascade; LOD_COUNT] {
    std::array::from_fn(|lod| {
        let scale = lod_scale(lod);
        let texel_width = COVERAGE_MULTIPLIER * scale / RESOLUTION as f32;
        let center = (camera / texel_width).floor() * texel_width;
        Cascade {
            center,
            scale,
            texel_width,
        }
    })
}

/// Creates the published cascade array: displacement in layers
/// `[0, LOD_COUNT)`, FFT surface in `[SURFACE_LAYER_BASE, CASCADE_LAYER_COUNT)`.
///
/// Reimplementation of the approach in `Scripts/LodData/LodDataMgrAnimWaves.cs`.
pub fn make_texture() -> Image {
    make_array_texture(CASCADE_LAYER_COUNT)
}

/// Creates a LOD-only scratch array for AnimWaves combine intermediates.
pub fn make_lod_scratch() -> Image {
    make_array_texture(LOD_COUNT as u32)
}

fn make_array_texture(layers: u32) -> Image {
    let bytes_per_pixel = TextureFormat::Rgba16Float
        .block_copy_size(None)
        .expect("Rgba16Float must have a fixed block size") as usize;
    let pixel_count = RESOLUTION as usize * RESOLUTION as usize * layers as usize;
    let mut image = Image::new(
        Extent3d {
            width: RESOLUTION,
            height: RESOLUTION,
            depth_or_array_layers: layers,
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

const DETAIL_NORMAL_SIZE: u32 = 1024;
const DETAIL_NORMAL_BYTES: &[u8] = include_bytes!("../assets/WaveNormals.png");

/// Loads Crest's shipped normal map and builds the mip chain Unity creates on import.
pub fn make_detail_normal_texture() -> Image {
    let sampler = ImageSampler::Descriptor(ImageSamplerDescriptor {
        address_mode_u: ImageAddressMode::Repeat,
        address_mode_v: ImageAddressMode::Repeat,
        mag_filter: ImageFilterMode::Linear,
        min_filter: ImageFilterMode::Linear,
        mipmap_filter: ImageFilterMode::Linear,
        ..default()
    });
    let mut image = Image::from_buffer(
        DETAIL_NORMAL_BYTES,
        ImageType::Extension("png"),
        CompressedImageFormats::NONE,
        false,
        sampler,
        RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
    )
    .expect("bundled Crest WaveNormals.png must decode");
    assert_eq!(image.texture_descriptor.size.width, DETAIL_NORMAL_SIZE);
    assert_eq!(image.texture_descriptor.size.height, DETAIL_NORMAL_SIZE);
    assert_eq!(image.texture_descriptor.format, TextureFormat::Rgba8Unorm);

    let mut levels = Vec::new();
    let mut size = DETAIL_NORMAL_SIZE;
    let mut previous = image
        .data
        .take()
        .expect("decoded Crest normal map must have CPU pixels");
    encode_detail_moments(&mut previous);
    loop {
        levels.extend_from_slice(&previous);
        if size == 1 {
            break;
        }
        previous = downsample_detail_normals(&previous, size);
        size /= 2;
    }
    image.data = Some(levels);
    image.texture_descriptor.mip_level_count = DETAIL_NORMAL_SIZE.ilog2() + 1;
    image.texture_descriptor.usage = TextureUsages::COPY_DST | TextureUsages::TEXTURE_BINDING;
    image
}

fn downsample_detail_normals(source: &[u8], source_size: u32) -> Vec<u8> {
    let target_size = source_size / 2;
    let mut target = Vec::with_capacity((target_size * target_size * 4) as usize);
    for y in 0..target_size {
        for x in 0..target_size {
            let mut slope = Vec2::ZERO;
            let mut second_moment = 0.0;
            for offset_y in 0..2 {
                for offset_x in 0..2 {
                    let source_x = 2 * x + offset_x;
                    let source_y = 2 * y + offset_y;
                    let index = ((source_y * source_size + source_x) * 4) as usize;
                    slope += Vec2::new(
                        source[index] as f32 / 127.5 - 1.0,
                        source[index + 1] as f32 / 127.5 - 1.0,
                    );
                    second_moment += 2.0 * source[index + 2] as f32 / 255.0;
                }
            }
            slope *= 0.25;
            second_moment *= 0.25;
            if slope.length_squared() > 1.0 {
                slope = slope.normalize();
            }
            target.extend_from_slice(&encode_detail_normal(slope, second_moment));
        }
    }
    target
}

fn encode_detail_moments(pixels: &mut [u8]) {
    let (pixels, remainder) = pixels.as_chunks_mut::<4>();
    debug_assert!(remainder.is_empty());
    for pixel in pixels {
        let slope = Vec2::new(pixel[0] as f32 / 127.5 - 1.0, pixel[1] as f32 / 127.5 - 1.0);
        pixel[2] = encode_detail_second_moment(slope.length_squared());
        pixel[3] = 255;
    }
}

fn encode_detail_normal(slope: Vec2, second_moment: f32) -> [u8; 4] {
    let encoded = ((slope.clamp(Vec2::splat(-1.0), Vec2::ONE) + Vec2::ONE) * 127.5).round();
    [
        encoded.x as u8,
        encoded.y as u8,
        encode_detail_second_moment(second_moment),
        255,
    ]
}

fn encode_detail_second_moment(value: f32) -> u8 {
    (value.clamp(0.0, 2.0) * 127.5).round() as u8
}

/// World-space scale of one cascade ring: the base scale doubled per LOD.
pub fn lod_scale(lod: usize) -> f32 {
    BASE_SCALE * LOD_SCALE_MULTIPLIER.powi(lod as i32)
}

#[cfg(test)]
#[path = "cascade_tests.rs"]
mod tests;
