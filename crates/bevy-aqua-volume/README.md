# bevy-aqua-volume

Fullscreen underwater volume for Aqua. When the camera is below its GPU-sampled local displaced
water level, a pass applies RGB Beer-Lambert transmittance and
closed-form in-scatter along the underwater segment. Particle scatter is a
weak coefficient times `scatter_scale` and `scatter_tint`, plus molecular
Rayleigh.

The shared `aqua::medium` library is registered for the volume pass. The existing
above-water shading remains unchanged.

Directional lights are refracted at the surface, then fall off along
`depth / L.y`, so sun elevation changes how fast the water goes dark.
Looking toward the sun is brighter via Henyey-Greenstein using
`WaterOptics::scattering_asymmetry`.

The pass automatically samples active `OceanView` cameras with `WaveQuery` and runs
when the camera is below that local displaced level. GPU readback gives this camera
sample roughly one frame of latency; an invalid initial sample conservatively uses the
selected mean level. The sample fills a near-plane crossing only.

See `../../docs/underwater.md` for supported behavior and current bounds.
