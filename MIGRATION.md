# Migration notes

## Unreleased changes after 0.1.3

These public struct additions require changes to existing full struct literals.

### `bevy_aqua_fft::BinSpec`

Set the new `min_wavelength` field explicitly. To preserve the previous band,
use `min_wavelength: 0.5 * max_wavelength`. The wavelength interval is now
`[min_wavelength, max_wavelength)`; extending the upper bound no longer changes
the lower bound automatically.

### `bevy_aqua_core::SurfaceParams`

The new `wave_slope_variance` field stores wave-slope energy for lighting.
For full struct literals, add `wave_slope_variance: [Vec4::ZERO; 2]`, or use
`..Default::default()` for unspecified fields. The waves plugin supplies the
variance during normal use. Without a producer, zero is the default.

Custom WGSL copies of the surface uniform must append the matching field:

```wgsl
wave_slope_variance: array<vec4<f32>, 2>,
```

Keep the field order aligned with Aqua's `SurfaceParams`; custom uniform buffers
must also include the additional 32 bytes.

See the README's [lighting appearance compatibility](README.md#lighting-appearance-compatibility)
section for visual changes.
