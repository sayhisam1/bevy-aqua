# Third-party attribution

## Crest Ocean System

Aqua's ocean techniques were developed by studying the [Crest Ocean System](https://github.com/wave-harmonic/crest/tree/db0658ff0b2e93e4a9e28cc2867509658b0ecc00) and reimplementing its approaches for Bevy in Aqua's own Rust and WGSL. Two Crest texture assets are redistributed verbatim. Their source-repository and upstream paths are:

- `crates/bevy-aqua-core/assets/WaveNormals.png` from `crest/Assets/Crest/Crest/Textures/WaveNormals/WaveNormals.png` (packaged by `bevy-aqua-core` as `assets/WaveNormals.png`).
- `crates/bevy-aqua-foam/assets/Foam2.png` from `crest/Assets/Crest/Crest/Textures/Foam2.png` (packaged by `bevy-aqua-foam` as `assets/Foam2.png`; GUID `02e417d5711139342884479f53dbecea`, bound to `_FoamTexture` by the shipped `Ocean.mat`).

Crest is used under the following MIT License:

```text
MIT License

Copyright (c) 2019 Wave Harmonic and contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## GodotOceanWaves

Aqua's FFT foam reconstruction, close-range sub-surface scattering, and roughness-damped Fresnel model reimplement techniques studied in [GodotOceanWaves](https://github.com/2Retr0/GodotOceanWaves/tree/a171446f8174348895aaafc426576c26261058b9).

GodotOceanWaves is used under the following MIT License:

```text
MIT License

Copyright (c) 2024 Ethan Truong

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Reference scope

Crest references include `Scripts/OceanBuilder.cs`, `Scripts/LodData/LodTransform.cs`, `Shaders/Ocean.shader`, `Shaders/OceanEmission.hlsl`, `Shaders/OceanFoam.hlsl`, and the shipped `Materials/Ocean.mat`. Aqua's source comments and tests record retained implementation-specific divergences where they matter.

GodotOceanWaves references include `assets/shaders/spatial/water.gdshader` and its FFT spectrum/resolve shaders. Aqua adapts these mechanisms to Bevy's pre-exposed lighting and render graph rather than claiming engine or visual parity.

The screenshots under `docs/images/` are deterministic renders produced by bevy-aqua's examples. The planar-reflection and sunset frames include the local CC0 buoy test model, which is not distributed in the crate package.

## Package license

Aqua packages are distributed under `MIT OR Apache-2.0`; see `LICENSE-MIT` and `LICENSE-APACHE`. The upstream copyright and permission notices reproduced above travel with every Aqua package.
