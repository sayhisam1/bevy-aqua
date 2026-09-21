# Underwater integration

Enable the opt-in Cargo feature `underwater`. `AquaPlugin` then installs the
fullscreen medium pass and compiles two-sided water-surface shading. Builds
without the feature retain the original culling and above-water shader path.

`WaterOptics::scatter_tint` and `scattering_asymmetry` control the homogeneous
single-scattering medium. Existing presets keep their accepted extinction,
scatter scale, and surface colours; the new tint defaults to white. The phase
asymmetry is clamped by the shader to `[-0.99, 0.99]`. Above-water transmission
and far water evaluate the same integral, converted to air as water-leaving
radiance.

The underside uses exact unpolarized water-to-air Fresnel and total internal
reflection. Its air-window distortion is projected through the active camera
and clamped in the active viewport. SSR is intentionally not included. Reflected
water-side rays use the same bounded homogeneous medium as the volume, including
the active body’s scatter tint and phase asymmetry. The open fallback is evaluated
in the resolved facet frame. A continuous shading-normal visibility clamp keeps
resolved shading microdetail incident-facing before Fresnel, reflection,
refraction, and medium evaluation, so the bounce remains in the local water
halfspace even when its world-space Y component points upward. This does not
model self-intersection or a second interface on fully folded wave geometry.

## Current bounds

The volume is a homogeneous approximation with a 256 m path cap and
directional-light single scattering. Wet/dry admission and the fallback surface
plane use a local GPU `WaveQuery` sample at the active camera, with roughly one
frame of readback latency. Before a valid sample arrives, admission conservatively
uses the selected mean level. This is one camera sample, not exact per-pixel
waterline closure. When a visible interface is clipped before the per-pixel near
plane, the pass synthesizes an underside terminal from that sample's local tangent
plane. The unbounded single-sheet ocean uses the same local plane to close upward
no-depth gaps: every upward ray from a wet camera must eventually exit. Bounded
bodies keep the stricter sub-near fallback because their open side walls need
explicit lateral clipping. Fully folded waves and bounded-body side exits remain
unsupported. An upward depth hit
through a missing water-surface fragment cannot be identified as air-side geometry;
preserving real displaced crest/trough hits takes priority, so that rare gap can
be attenuated as water. Aqua currently maintains one active `OceanView`; the extracted medium belongs to that view.
Auxiliary water views are excluded by the facade. Multi-camera rendering is not
yet a supported Aqua contract.

Opaque receiver relighting is disabled by default because multiplying the
already-shaded scene also affects emissive and local-light contributions. Set
`UnderwaterSettings::receiver_relighting` only when that approximation is
acceptable. Orthographic cameras currently skip the volume composite. Multisampled depth is
resolved conservatively to the nearest reverse-Z sample before the single-sample
post-process. Native Vulkan captures cover MSAA off and the default MSAA mode,
including non-zero viewport preservation. Other configurations and browser
runtime behavior remain unverified.
