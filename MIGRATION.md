# Migration notes (unreleased)

## Wave spectrum and source-derived roughness

### `BinSpec.min_wavelength`

Every `bevy_aqua_fft::BinSpec` literal now needs `min_wavelength` in metres.
The active interval is `[min_wavelength, max_wavelength)`. To preserve an old
single-octave band, set `min_wavelength` to `0.5 * max_wavelength`.
For wider bands, set the lower bound explicitly; do not derive it from the
new upper bound.

Aqua keeps the first four spectral octave bands unchanged. Its coarsest
spectral band now extends from its existing lower bound to one quarter of
its texture period (`texel_width * texture_res / 4.0`). The analytic bands,
core cascade metadata, and texture periods do not change. More long-wave
energy is intentional. Global spectral height normalization redistributes
energy across the supported modes, so existing spectral scenes can have
different swell, short-wave detail, and reflection roughness at the same
settings. Review visual baselines rather than assuming old images still match.

### `SurfaceParams.wave_slope_variance`

`bevy_aqua_core::cascade::SurfaceParams` adds a trailing
`wave_slope_variance: [Vec4; 2]` field. Struct literals must provide it, or use
`..Default::default()`. The default is `[Vec4::ZERO; 2]`.

Custom GPU uniform mirrors must append
`wave_slope_variance: array<vec4<f32>, 2>` in the same position. This adds
**32 bytes** to the surface uniform, not a new binding. Update custom buffer
sizes and binding-size checks. Existing fields keep their offsets.

The eight scalar slots store exclusive, not cumulative, phase-mean vertical
height-gradient energy, ordered fine to coarse:

- Analytic: five cascade bands in slots 0–4; slots 5–7 are zero.
- Spectral: four fine octave bands in slots 0–3. Slots 4–7 split the coarsest
  band into four equal log-wavelength intervals ending at texture period / 4.

`AquaWavesPlugin` measures both startup sources before their GPU upload and
caches the results. After material updates, it selects the active wave model's
cache and publishes it to cascade materials, including newly added materials.
Unchanged values do not trigger an extra material modification. Model switches
select a cache; they do not regenerate the startup sea state. Set sea state,
wind direction, wind speed, and fetch before startup. Runtime authoring changes
do not regenerate these sources or their cached variance.

Custom producers must provide matching exclusive variance and model metadata
if they bypass this plugin. The plugin owns this field while active. A core
material without a wave producer starts with zero wave variance.

The removed shader constants `GERSTNER_SLOPE_VARIANCE` and
`FFT_JONSWAP_SLOPE_VARIANCE` are no longer exported by
`aqua::waves::displace`; remove custom imports and use the uniform instead.

### Scope and limitations

This is **linear-source variance**, not a measurement of final rendered
normal variance. It does not include local depth attenuation/shoaling,
horizontal choppy-displacement Jacobians, river deformation, or geometry and
texture reconstruction filters. It is not the actual local slope covariance.
The coarsest spectral band's shallow-water attenuation still uses four
log-wavelength bins. Each covers 1.25 octaves in the shipped five-octave long
band, so this remains a coarse shoaling approximation, not per-mode local
attenuation. Partial-band filtering assumes a uniform
energy distribution in log wavelength within each stored interval. Detail
normal and capillary variance still use their separate existing estimates.

Near-water roughness now uses the selected source variance with the existing
detail-strength multiplier, lighting-normal fade, grazing boost, and roughness
cap. `WaterOptics::sun_roughness` still controls direct-light highlight width.
The far tier still uses the existing fixed roughness cap; it does not integrate
the new variance table. This change does not alter far-tier opacity or replace
the resolved-normal pipeline.
