# bevy-aqua

[![crates.io](https://img.shields.io/crates/v/bevy-aqua.svg)](https://crates.io/crates/bevy-aqua)

## [Try the live WebGPU demos →](https://sayhisam1.github.io/bevy-aqua/)

Camera-centred ocean rendering for Bevy 0.19 with analytic and FFT waves,
depth-aware transmission, reflections, persistent foam, localized water
bodies, and GPU surface queries.

![FFT ocean at sunset with planar buoy reflection](docs/images/sunset-fft.jpg)

## Features

- Five camera-centred displacement cascades with smooth LOD blending.
- Crest-style analytic waves or Tessendorf spectral waves.
- Beer-Lambert transmission, refraction, reflections, and scene lighting.
- Persistent whitecaps and shoreline foam.
- Static terrain heightfields for shoaling and shallow-water optics.
- Bounded ponds, lakes, and river corridors with per-body optics.
- GPU `WaveQuery` samples for buoyancy and gameplay.
- Optional budgeted Hanabi spray.
- Native desktop and browser WebGPU/Wasm rendering.

## Highlights

| FFT open ocean | Coastal foam and shallow-water optics |
|---|---|
| ![Close FFT wave detail](docs/images/fft-open-ocean.jpg) | ![Foam and coastal transmission at an island shore](docs/images/coastal-foam.jpg) |

| Bounded river corridor | Planar reflection |
|---|---|
| ![Localized river water following a curved corridor](docs/images/bounded-river.jpg) | ![Planar reflection of a buoy in calm water](docs/images/planar-reflection.jpg) |

## Compatibility

| bevy-aqua | Bevy | Rust | Verified target |
|---|---|---|---|
| 0.1 | 0.19 | 1.95+ | Desktop Vulkan and browser WebGPU (Wasm) |

Browser WebGPU/Wasm support was contributed by
[@wellscrosby](https://github.com/wellscrosby) in [#1](https://github.com/sayhisam1/bevy-aqua/pull/1).

The default `query` and `reflect` features enable GPU wave probes and planar
reflections. The optional `spray` feature adds `bevy-aqua-spray` and `bevy_hanabi`,
implies `query`, and defaults to `Off` at runtime. Mobile and other desktop APIs
are not yet verified.

## Quick start

```rust
use bevy::{
    camera::{Exposure, Hdr},
    core_pipeline::prepass::DepthPrepass,
    light::{Atmosphere, AtmosphereEnvironmentMapLight, atmosphere::ScatteringMedium},
    pbr::AtmosphereSettings,
    prelude::*,
};
use bevy_aqua::{AquaPlugin, Ocean};

fn main() {
    App::new()
        .insert_resource(Ocean::default())
        .add_plugins((DefaultPlugins, AquaPlugin))
        .add_systems(Startup, |mut commands: Commands,
                              mut media: ResMut<Assets<ScatteringMedium>>| {
            commands.spawn(Atmosphere::earth(media.add(ScatteringMedium::earth(256, 256))));
            commands.spawn((
                Camera3d::default(),
                Hdr,
                Exposure { ev100: 12.0 },
                AtmosphereSettings::default(),
                AtmosphereEnvironmentMapLight::default(),
                DepthPrepass,
                Transform::from_xyz(24.0, 12.0, 32.0)
                    .looking_at(Vec3::ZERO, Vec3::Y),
            ));
            commands.spawn((
                DirectionalLight { illuminance: 16_000.0, ..default() },
                Transform::from_rotation(Quat::from_euler(
                    EulerRot::XYZ, -0.8, -0.6, 0.0,
                )),
            ));
        })
        .run();
}
```

Insert one `Ocean` resource. `Ocean::level` sets the global sea level. Remove
the resource for bounded-water-only worlds. One active `Camera3d` is the
supported view path.

Use HDR and an explicit camera exposure when lighting water with daylight-level
illuminance. The examples use EV100 12; tune exposure for your scene rather than
changing the water's light response. A `ClearColor` only paints the background:
it does not light the water or supply a reflected sky. Provide an environment
map, or use `AtmosphereEnvironmentMapLight` with an atmosphere as above.

## Configuration

Insert `OceanWaves` and `AquaSettings` before `AquaPlugin` to replace their
defaults.

`OceanWaves` selects `WaveModel::Analytic` or `WaveModel::Spectral`. It also
sets sea state, shallow-water attenuation, wind direction and speed, fetch,
and world-XZ flow. `sea_state`, wind, and fetch determine startup spectrum
data; set them before the plugin starts.

`AquaSettings` selects a `WaterOptics` preset and a `detail_strength` in
`0..=2`. `WaterOptics::DEEP_OCEAN` is the default. Coastal, tropical, and
clear-fresh presets are also provided. `far_tier_start` and `far_tier_end`
bound the reduced-cost shading transition in metres. Far shading keeps sun
and reflections while omitting depth, foam, and sampled subsurface detail.
`reflections` selects the default planar mirror views or the cubemap-only path. Both use the same dielectric Fresnel response. Mark terrain or a
scene root with `ReflectedInWater` to include it and its descendants in planar
views.

**Bevy 0.19.1 limitation:** planar mirrors can light double-sided
`StandardMaterial` geometry with the wrong normal polarity. The mirrored camera
reverses winding, but upstream material specialization swaps culling without
changing the raster front-face convention. This has been reproduced with both
forward and deferred reflection rendering; the ordinary view remains correct.
Aqua does not currently patch that dependency. Disabling `double_sided` is not a
general workaround: physical back faces then lose their intended lighting.
A comprehensive repair must change front-face convention and culling together
in Bevy. A workspace dependency patch must also be configured by downstream
applications; it is not inherited from a library's manifest.

`caustics` controls the default procedural shallow-bed lighting; set it
to `None` to skip both texture samples. Hosts can update
`CausticsSunVisibility` to fold cloud-shadow coverage into the direct sun.

Caustic patterns are anchored to the opaque receiver selected by transmission,
including accepted refraction or its raw fallback. The receiver must be submerged
in the transmitting body; its depth uses that body's local water level. Missing,
exposed, and cross-body receivers omit the caustic contribution.

Mip selection uses four neighboring depth samples after caustic admission. These
hold the central refraction offset fixed, rather than replaying neighboring
normals and refraction acceptance. A neighbor view-depth jump above
`max(0.05 m, 1% of receiver view depth)` suppresses the contribution. This
heuristic can reject steep continuous surfaces and miss small discontinuities;
it is not a surface-identity test or exact refracted filtering. Sun/shadow
selection remains at the water fragment. The pattern and incoming attenuation
are still flat-interface surrogates, not traced sun-to-receiver caustics.
Disabling caustics also skips these four neighbor reads.

### Terrain bed

Insert a `BedHeightMap` before `AquaPlugin`. Its single-channel image stores
normalized height. `origin` is the world-XZ centre of texel `(0, 0)`, `size`
is the distance from the first to last texel centres, and `height_range`
decodes normalized values to metres.

```rust,ignore
commands.insert_resource(BedHeightMap {
    image: terrain_heightmap,
    origin: Vec2::splat(-10_000.0),
    size: Vec2::splat(20_000.0),
    height_range: [0.0, 1_808.0],
});
```

Without a bed map, Aqua uses deep-water attenuation everywhere.

### Localized water and queries

`WaterBody` marks a bounded surface. Add a sibling `WaterShape` for circles,
polygons, corridors, or rivers. Shape coordinates are local to the entity. Its
propagated `Transform` supplies world
XZ placement and surface Y, so parenting, yaw, reflection, nonuniform scale,
and planar shear work naturally. Tilted water is rejected because the renderer
uses one horizontal level per body. Add `WaterOptics` as a sibling component to
override the global ocean optics.

```rust,ignore
commands.spawn((
    WaterBody,
    WaterShape::Circle { radius: 24.0 },
    WaterOptics::CLEAR_FRESH,
    Transform::from_xyz(40.0, 3.0, -20.0),
));
```

Keep bodies inside the bed-map region when they need shoreline foam or
shallow-water attenuation. Moving a body or a transformed ancestor rebuilds
the shared shoreline fields, so body transforms are intended to change
infrequently.

With the default `query` feature enabled, add `WaveQuery` to an entity whose
transform sits on the owning surface's mean plane. Aqua refreshes its
`WaveSurface` with relative displacement. `WaveSurface::valid` is false when
no ocean or bounded body owns the point. GPU samples arrive with about one
frame of readback latency. Rivers use the matching analytic path; other bounded
shapes remain flat, matching their rendered geometry. The per-frame limit is
256 probes. `WaveSurface::crest` exposes the same
horizontal-compression source used to seed persistent whitecaps.

### Spatial consistency (unreleased)

These fixes make the renderer, queries, and baked fields agree about where
water is. In plain terms:

1. **Bounded-water distance uses its enclosing square.** The stored extent was
   a square half-width but far culling treated it as a circle radius, so pond
   corners could disappear too early. Forward and motion passes now measure
   distance from the square. There is no API change; distant edge pixels and
   motion coverage can change.
2. **Ocean query LOD stays fixed in world space.** Flow should move wave phase,
   not the LOD rings. Queries previously chose an LOD from the flow-shifted
   sample point, so a stationary probe could cross LODs as time passed. LOD now
   uses the requested world XZ while displacement still follows flow. There is
   no API change; query values can change near LOD boundaries.
3. **Shore fields publish their actual rounded texel region.** Image dimensions
   are rounded up to whole texels, but the old metadata kept the pre-rounding
   size. Sampling and baked texel centres could then disagree near the positive
   X/Z edges. The region now spans `width * texel` by `height * texel`. There is
   no API change; ownership, flow, and shore results can shift at edge texels.
4. **Bed-map `size` means centre-to-centre span.** The implementation already
   used `size` from the first texel centre to the last. Documentation had called
   it an edge-to-edge image size. The docs now state `(N - 1) * step`. Runtime
   behavior is unchanged, but maps authored from the old wording may need their
   metadata corrected.
5. **Pond-only worlds keep unused support vertices still.** Without an `Ocean`,
   vertices outside bounded-water ownership could still receive ocean
   displacement and motion history. They now remain flat while river-owned
   vertices keep their river wave path. There is no API change; stray moving
   pond-edge geometry disappears and motion vectors match.

See [`MIGRATION.md`](MIGRATION.md) for the migration handoff.

### Spray

Enable Cargo feature `spray`, then insert `SpraySettings` before `AquaPlugin`.
`SprayQuality::Off` does no wave-query or particle work. Low and High use fixed
probe, emitter, particle-rate, distance, and projected-screen-coverage budgets.
They reuse `WaveSurface::crest` and bed depth rather than adding a spray fluid
simulation.

### Spray consistency (unreleased)

In plain terms:

1. **A burst spends budget only when an emitter keeps it.** Candidates used to
   wrap around the fixed emitter pool and overwrite pending bursts after those
   bursts had already spent tokens and cooldown. Each emitter is now reserved
   once per dispatch, so crowded crests keep the spray density the budget paid
   for. There is no API change; busy-view burst selection and density can change.
2. **Probe direction stays stable when looking nearly vertical.** Projecting the
   camera's almost-vertical forward vector made tiny rounding errors steer the
   probe grid. The near-vertical path now recovers yaw from camera-right. There
   is no API change; extreme-pitch probe placement and resulting spray can change.
3. **Particles inherit the water surface and current.** Bursts always launched
   around world-up and ignored river flow because emitters received only crest
   strength. Emitters now receive a safe query normal and the owning body's
   authored current. Spray visibly leans with crests and drifts with rivers.
   There is no Rust API change; sloped or flowing spray trajectories can change.

See [`MIGRATION.md`](MIGRATION.md) for the full ELI5 handoff and limits.

## Examples

The examples are small, fixed scenes. Each one demonstrates one public feature
without command-line configuration, external assets, or platform-specific
source code.

| Example | Demonstrates | Screenshot |
|---|---|---|
| `ocean` | Minimal analytic ocean | <img src="docs/images/examples/ocean.jpg" alt="ocean example" width="220"> |
| `spectral_waves` | FFT spectral wave producer | <img src="docs/images/examples/spectral_waves.jpg" alt="spectral_waves example" width="220"> |
| `foam` | Persistent whitecaps on rough water | <img src="docs/images/examples/foam.jpg" alt="foam example" width="220"> |
| `bounded_water` | Local circular and polygonal water bodies | <img src="docs/images/examples/bounded_water.jpg" alt="bounded_water example" width="220"> |
| `river` | Curved river flow with changing width and speed | <img src="docs/images/examples/river.jpg" alt="river example" width="220"> |
| `terrain_bed` | Terrain height input, shoaling, and shallow-water optics | <img src="docs/images/examples/terrain_bed.jpg" alt="terrain_bed example" width="220"> |
| `debug_views` | Automatic cycle through all Aqua diagnostics | <img src="docs/images/examples/debug_views.jpg" alt="debug_views example" width="220"> |
| `water_optics` | Water appearance presets shown side by side | <img src="docs/images/examples/water_optics.jpg" alt="water_optics example" width="220"> |
| `planar_reflection` | Planar reflection of marked scene geometry | <img src="docs/images/examples/planar_reflection.jpg" alt="planar_reflection example" width="220"> |
| `wave_query` | GPU surface queries driving a procedural buoy | <img src="docs/images/examples/wave_query.jpg" alt="wave_query example" width="220"> |

Run any scene natively:

```sh
cargo run --example ocean
```

The same source runs with browser WebGPU:

```sh
rustup target add wasm32-unknown-unknown
cargo install wasm-server-runner
CARGO_TARGET_WASM32_UNKNOWN_UNKNOWN_RUNNER=wasm-server-runner \
  cargo run --target wasm32-unknown-unknown --example ocean
```

See [`examples/README.md`](examples/README.md) for the full command list.

## Screen-space refraction limitation

Refraction validates one opaque depth texel, while transmission color uses linear
filtering. At a foreground silhouette, a neighboring above-water opaque texel can
therefore contribute color even when `RefractionValidity` accepts the sample.
Viewport-edge color clamping does not prevent this interior silhouette case.
`RefractionValidity` is not proof that every filtered color contributor lies
behind the water.

## Lighting appearance (unreleased)

Cubemap-only oceans now use the same dielectric Fresnel response as planar
reflections. Grazing-angle reflections can therefore be stronger than before.
`WaterOptics::sun_roughness` controls direct-light highlight width, not the
Fresnel curve; negative values still inherit the ocean setting. Existing scenes
may need an appearance review. That Fresnel change does not alter public fields
or GPU layouts; the separate wave-spectrum and roughness changes do. See
[MIGRATION.md](MIGRATION.md) for `BinSpec` and surface-uniform updates.

Water lighting also retains resolved wave-normal slopes in both near and far
shading. This differs from the former GodotOceanWaves-style exponential
lighting-normal fade, so existing scenes can show stronger wave reflections at
a distance. Near and far shading share the normal and wave-roughness calculation.

The direct-sun GGX alpha floor now defaults to `0.04` (previously `0.4`) for
narrower base highlights. Filtered wave variance still adds roughness. This floor
also feeds the existing subsurface-light mask; it is an appearance setting, not a
physically calibrated water parameter. Explicit per-body `sun_roughness` values
still override the global floor; negative values inherit it.

The wave-roughness cap, Fresnel model, and foam fade are unchanged. Detail and
capillary slopes still fade toward the far tier, but their removed energy now
moves into unresolved roughness instead of disappearing. Body scatter scaling
also applies in the far tier, and transmissive accepted paths remain on near
shading. Footprint-based roughness and detail mip filtering remain;
geometric/FFT normals are not fully convolved over each pixel footprint. This
is not a guarantee of alias-free rendering.

## Debug-mode migration (unreleased)

Two `AquaDebug` variants have been removed:

- Replace `ShallowComposite` with `Shaded`; both selected the same rendering path.
- Replace `FoamDensityBilinear` with `FoamDensity` to inspect foam using its normal
  reconstruction filter. The old bilinear-only comparison is no longer available.

The normal foam filters and remaining diagnostic views are unchanged.

## AI disclosure

This project was developed with assistance from AI coding agents.

## Attribution

Aqua adapts established ocean-rendering techniques and includes attributed
third-party assets. See [ATTRIBUTION.md](ATTRIBUTION.md) for sources and
licenses.
