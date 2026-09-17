# bevy-aqua-volume

Fullscreen underwater volume for Aqua. When the camera is below the local
water surface, a pass applies RGB Beer-Lambert transmittance and
closed-form in-scatter along the underwater segment. Particle scatter is a
weak coefficient times `scatter_scale` and `scatter_tint`, plus molecular
Rayleigh.

The same `aqua::medium` integral shades the cascade surface from above as
water-leaving radiance.

Directional lights are refracted at the surface, then fall off along
`depth / L.y`, so sun elevation changes how fast the water goes dark.
Looking toward the sun is brighter via Henyey-Greenstein using
`WaterOptics::scattering_asymmetry`.

The pass runs whenever an `OceanView` camera is below the local surface.
