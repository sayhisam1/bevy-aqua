# bevy-aqua-reflect

Planar scene reflections for Aqua. `AquaReflectPlugin` maintains at most two
`Rgba16Float` mirror views for distinct eligible water levels below the camera.
The ocean has priority; bounded bodies are ranked by center distance in world XZ.
Selection does not test the camera frustum, so an offscreen body can occupy a
mirror slot. This is a bounded selection heuristic, not a visibility guarantee.
Add `ReflectedInWater` to terrain, cloud, and large static-mesh
entities that should appear in the water. Directional lights are included
automatically. Aqua falls back to its environment cubemap outside a mirror
view.

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

The paired delta includes the mirror camera and the added water-material
sample. It is below Aqua's 1.0 ms reflection budget.
