# Examples

Every example is a fixed visual scene with no command-line options or external
assets. The same source runs natively and with browser WebGPU.

| Name | Visual focus | Expected look | Try online |
|---|---|---|---|
| `ocean` | Minimal analytic ocean | <img src="../docs/images/examples/ocean.jpg" alt="ocean example" width="260"> | [Launch](https://sayhisam1.github.io/bevy-aqua/ocean/) |
| `spectral_waves` | FFT spectral wave producer | <img src="../docs/images/examples/spectral_waves.jpg" alt="spectral_waves example" width="260"> | [Launch](https://sayhisam1.github.io/bevy-aqua/spectral_waves/) |
| `foam` | Persistent whitecaps on rough water | <img src="../docs/images/examples/foam.jpg" alt="foam example" width="260"> | [Launch](https://sayhisam1.github.io/bevy-aqua/foam/) |
| `bounded_water` | Local circular and polygonal water bodies | <img src="../docs/images/examples/bounded_water.jpg" alt="bounded_water example" width="260"> | [Launch](https://sayhisam1.github.io/bevy-aqua/bounded_water/) |
| `river` | Curved river flow with changing width and speed | <img src="../docs/images/examples/river.jpg" alt="river example" width="260"> | [Launch](https://sayhisam1.github.io/bevy-aqua/river/) |
| `terrain_bed` | Terrain height input, shoaling, and shallow-water optics | <img src="../docs/images/examples/terrain_bed.jpg" alt="terrain_bed example" width="260"> | [Launch](https://sayhisam1.github.io/bevy-aqua/terrain_bed/) |
| `debug_views` | Automatic cycle through all Aqua diagnostics | <img src="../docs/images/examples/debug_views.jpg" alt="debug_views example" width="260"> | [Launch](https://sayhisam1.github.io/bevy-aqua/debug_views/) |
| `water_optics` | Water appearance presets shown side by side | <img src="../docs/images/examples/water_optics.jpg" alt="water_optics example" width="260"> | [Launch](https://sayhisam1.github.io/bevy-aqua/water_optics/) |
| `planar_reflection` | Planar reflection of marked scene geometry | <img src="../docs/images/examples/planar_reflection.jpg" alt="planar_reflection example" width="260"> | [Launch](https://sayhisam1.github.io/bevy-aqua/planar_reflection/) |
| `wave_query` | GPU surface queries driving a procedural buoy | <img src="../docs/images/examples/wave_query.jpg" alt="wave_query example" width="260"> | [Launch](https://sayhisam1.github.io/bevy-aqua/wave_query/) |
| `underwater` | Open ocean from 20 m down, looking toward the sun | <img src="../docs/images/examples/underwater.jpg" alt="underwater example" width="260"> | [Launch](https://sayhisam1.github.io/bevy-aqua/underwater/) |

## Native

```sh
cargo run --example ocean
```

## Browser WebGPU

The hosted gallery is available at
[https://sayhisam1.github.io/bevy-aqua/](https://sayhisam1.github.io/bevy-aqua/).
WebGPU support is required. Current Chrome, Edge, and Firefox Nightly builds are
recommended.

To run a browser build locally, install the target and runner once:

```sh
rustup target add wasm32-unknown-unknown
cargo install wasm-server-runner
```

Then change only the example name:

```sh
CARGO_TARGET_WASM32_UNKNOWN_UNKNOWN_RUNNER=wasm-server-runner \
  cargo run --target wasm32-unknown-unknown --example ocean
```

## Enabling the hosted gallery

Repository administrators must select **GitHub Actions** under
**Settings → Pages → Build and deployment → Source** once. The
`Deploy WebGPU examples` workflow handles later deployments from `main`.
