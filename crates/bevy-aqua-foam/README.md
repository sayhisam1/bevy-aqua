# bevy-aqua-foam

Persistent foam for Aqua: a Crest-style simulation cascade (advect with the
current, fade, accumulate from wave crests + shoreline depth) plus the
foam terms the surface material shades through `aqua::foam::shade`.

## Owns

- The sim compute shader (`foam.wgsl`): reprojection across layout changes,
  global-current transport, wave-crest/shoreline injection, fixed-step catch-up.
- `shade.wgsl` (`aqua::foam::shade`): bicubic cascade sampling, river-bank
  streaks, breakup mask, bubble tint, foam lighting for the composed material.
- `Textures` (double-buffered state + published surface + pattern) and
  the render-world write node ordered after `bevy_aqua_core::AnimWavesWritten`.

## Public API

`AquaFoamPlugin`, `Textures`. Settings live on `bevy_aqua_core::OceanWaves`
(model gate and global `flow`) and the material uniform; this crate adds no config.

Persistent density stays in world coordinates. Each fixed simulation step
backtraces its history by `flow * dt`; wave-source sampling uses `flow * time`
to match the rendered waves. Layout-only reprojection does not advance time.
This transport uses the global ocean current, not the local river flow field.
River-bank streaks remain a separate surface-shading contribution.

Large accumulated `flow * time` offsets can reach wave-texture boundaries.
This change does not solve that existing finite-domain limitation.

## Test alone

```
cd crates/bevy-aqua-foam && cargo test
```

Isolation example: `cargo run --example foam-only` (core + foam on a
synthetic wave input).
