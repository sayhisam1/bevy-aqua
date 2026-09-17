# bevy-aqua-reflect

Planar scene reflections for Aqua. `AquaReflectPlugin` maintains at most two
`Rgba16Float` mirror views for the nearest visible water
levels. Add `ReflectedInWater` to opaque or alpha-masked mesh entities that
should appear in the water. Materials must write the depth prepass; alpha-blended
materials and custom materials without that prepass are not supported.
Directional lights and the main camera's environment light are inherited.
Auxiliary mirror cameras do not generate atmosphere sky or environment maps.
Aqua falls back to its environment cubemap outside a mirror view and wherever
its depth buffer contains no geometry, including empty sky pixels.

Select `ReflectionMode::Cubemap` for the cubemap-only path, or
`ReflectionMode::Planar { scale, distortion }` through `AquaSettings`.

## GPU budget

Measured on an NVIDIA RTX 3070 at 2560x1440 with the former fixed island
profiling scene, using 300 measured frames after 300 warmup frames. Values are
medians of three paired runs from Bevy's reported GPU span sum. Run
`cargo run --release --example planar_reflection` for the current focused
reflection example; it is a visual example rather than the historical benchmark.

| mode | reported span sum (ms) | paired reflection delta (ms) |
| --- | ---: | ---: |
| Cubemap | 2.684 | — |
| Planar, scale 0.5 | 2.889 | **0.194** |

These historical timings predate the depth-coverage export pass and must not be
used as the current reflection cost. The export adds one compute pass and a
second HDR texture per mirror. Current GPU cost has not been established by this
measurement.
