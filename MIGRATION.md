# Migration notes (unreleased)

## Spray admission and surface motion

The `spray` feature remains opt-in. Its public Rust API does not change. These
fixes change which pending bursts survive a frame and how their particles move.

### ELI5 handoff

1. **Only retained bursts spend the shared budget.** Context: each CPU emitter
   can hold one pending burst before Hanabi consumes it. What was wrong: more
   candidates than emitters could wrap around and overwrite earlier pending
   bursts, although every overwritten burst still spent particles and put its
   probe on cooldown. Why: admission counted candidate decisions instead of the
   emitter slots that actually retain them. Fix: reserve each emitter at most
   once per dispatch, then charge tokens, screen coverage, cursor movement, and
   cooldown only after a burst gets a slot. Visible result: crowded crests no
   longer make spray look sparse by silently wasting the frame budget. Breaking:
   no API change; burst choice and density can change in busy views. Admission
   still means a retained CPU request, not proof of later GPU particle creation.

2. **Near-vertical cameras keep a stable probe heading.** Context: spray probes
   form a grid ahead of the active camera. What was wrong: looking almost
   straight up or down made the camera-forward projection nearly zero, so tiny
   floating-point changes could rotate or collapse the grid. Why: a near-zero
   horizontal vector cannot provide a reliable heading. Fix: use the projected
   forward direction normally, but recover yaw from camera-right near vertical
   pitch and use a fixed fallback only if both are degenerate. Visible result:
   the sampled spray region stays aimed with the camera yaw at extreme pitch.
   Breaking: no API change; probe locations and resulting spray can change near
   straight-up or straight-down views.

3. **Spray launches with the queried surface and authored current.** Context:
   particles should leave a tilted wave and moving river with that water's
   motion. What was wrong: every burst launched around world-up and ignored
   river flow. Why: the Hanabi effect only received strength. Fix: pass a safe
   normalized `WaveSurface::normal` and the owning resolved body's finite flow
   into each emitter; the shader adds current without scaling it by spray
   strength. Flat still water keeps the old random launch distribution. Visible
   result: spray leans away from sloped crests and drifts with authored river
   current. Breaking: no public Rust API change; spray trajectories change on
   sloped or flowing surfaces, and custom visual baselines may need updates.

## Water spatial consistency

The public Rust API, bind groups, and uniform layouts do not change. These
fixes can change rendering, motion vectors, wave-query samples, or baked shore
fields. Review relevant image and gameplay baselines.

### ELI5 handoff

1. **Bounded-water culling uses the shape of its stored extent.** Context: each
   bounded body stores a world-axis square half-extent that encloses the body,
   and both the forward and motion passes use it to skip water beyond the far
   tier. What was wrong: the shader subtracted that half-extent as if it were a
   circle radius, which makes the square's corners appear farther away than they
   are. Why: the meaning of `BodyParams::extent.w` and the distance formula had
   diverged. Fix: measure point-to-square distance in both passes and document
   the field as a square half-extent. Visible result: bounded-water corners no
   longer disappear early, and forward and motion coverage agree. Breaking: no
   API or GPU layout change; far-away edge pixels and their motion vectors can
   change.

2. **Ocean query LOD is anchored to the requested world position.** Context:
   world flow translates the phase used to sample waves, while ocean LOD rings
   are fixed around the world-space cascade centre. What was wrong: GPU queries
   selected and blended LOD from the flow-shifted sampling coordinate. A probe
   that did not move could therefore cross an LOD boundary merely because time
   passed. Why: one coordinate was reused for two different jobs. Fix: select
   and blend LOD from `WaveQuery`'s requested world XZ, but retain the
   flow-shifted coordinate for displacement and derivatives. Visible result:
   stationary probes no longer get flow-driven LOD transitions; the waves still
   move with flow. Breaking: no API/layout change; query displacement, normal,
   and crest values can change near LOD boundaries.

3. **Shore-field bounds match the baked texel grid.** Context: a requested
   world region is rounded up to an integer image width and height before Aqua
   bakes body ownership and flow at texel centres. What was wrong: field
   metadata still described the smaller pre-rounding region, so texture
   sampling did not map back to the centres used during baking. Why: dimensions
   were rounded without recomputing their world span. Fix: publish a region size
   of `(width * texel, height * texel)`. Visible result: ownership, flow,
   shoreline foam, and shallow-water lookups stay aligned through the positive
   X/Z edge texels. Breaking: no public API/layout change; edge texels and the
   region's positive bounds can move by less than one texel.

4. **`BedHeightMap::size` is documented as a texel-centre span.** Context:
   `origin` is the world-XZ centre of texel `(0, 0)`. What was wrong: the docs
   called `size` the whole image's edge-to-edge width, although the mapping uses
   it as the distance from the first texel centre to the last. Why: the prose
   did not match the existing coordinate convention. Fix: state that an axis
   with `N` texels spaced by `step` uses `(N - 1) * step`. Visible result: none
   for already correct inputs; newly authored maps line up with their intended
   world coordinates. Breaking: runtime behavior is unchanged, but callers that
   followed the old wording should correct their `size` metadata.

5. **Unowned support vertices stay still in pond-only worlds.** Context: the
   shared support mesh includes vertices around bounded bodies, and an absent
   `Ocean` means that unowned area is not water. What was wrong: unowned
   vertices still took the ocean displacement path, including its previous-frame
   motion reconstruction, even though fragments outside a pond were discarded.
   Why: the deformation path checked bounded ownership but not ocean presence.
   Fix: keep unowned vertices at their current flat position when there is no
   ocean; river-owned vertices retain the river path. Visible result: pond edges
   no longer inherit stray moving ocean geometry, and motion history matches the
   still geometry. Breaking: no API/layout change; pond-only edge rendering and
   motion-vector baselines can change.

## Caustics and viewport refraction

These changes do not alter the public Rust API, bind groups, or uniform layouts.
They can change rendered water, so update image baselines after reviewing the
following behavior.

### ELI5 handoff

1. **Incoming caustic attenuation follows the refracted sun ray.** Context:
   caustics fade according to the distance that sunlight travels underwater.
   What was wrong: that distance extended the air-side ray straight through the
   water. Why: the attenuation path did not apply Snell refraction. Fix: compute
   the transmitted air-to-water direction before measuring the underwater path.
   Visible result: low-sun caustics keep a more local, physically plausible
   contribution; overhead sun is effectively unchanged. Breaking: no API or GPU
   layout change, but low-angle visual baselines can change.

2. **Caustic textures use footprint-selected mipmaps.** Context: a screen pixel
   can cover many cells of the caustic pattern. What was wrong: always sampling
   mip zero made distant or small cells shimmer and alias. Why: the shader did
   not match texture detail to the pixel's world-space footprint. Fix: generate
   an area-average R8 mip chain from full-precision means and use explicit,
   trilinear LODs for both pattern layers. Visible result: minified caustics are
   steadier while close detail remains. Breaking: no API/layout change; the
   texture has extra mip storage and visual baselines can change.

3. **Caustics attach to the accepted transmission receiver.** Context: the bed
   seen through water can move after refraction and can belong to a bounded water
   body. What was wrong: caustics used the water fragment's undisplaced XZ and
   water depth instead of the accepted background hit. Why: the old path never
   carried the reconstructed receiver into caustic lighting. Fix: reuse the raw
   or accepted refracted world position, reject exposed or cross-body hits, and
   derive a conservative four-neighbor footprint. Visible result: caustics stay
   on the transmitted bed instead of sliding with the water surface or leaking
   across bodies. Breaking: no public API/layout change; caustics can disappear
   at rejected depth edges, and this is still a heuristic rather than a physical
   sun-to-bed trace.

4. **Refracted depth uses viewport-sized pixel addressing.** Context: depth
   validation converts a continuous viewport UV to a depth texel. What was
   wrong: multiplying by `viewport_size - 1` compressed the mapping and could
   select the neighboring pixel. Why: endpoint interpolation was used where a
   containing-texel lookup was required. Fix: multiply by the full viewport size
   and clamp only the endpoint. Visible result: refraction validity and caustic
   receiver checks line up with viewport pixels. Breaking: no API/layout change;
   edge and one-pixel visual decisions can change.

5. **Transmission color filtering stays inside its viewport.** Context: the
   transmission texture can contain several viewports or unused backing area.
   What was wrong: clamping only to the whole texture allowed linear filtering
   near an edge to blend a neighboring viewport texel. Why: normalized viewport
   coordinates were converted without a half-texel viewport inset. Fix: clamp
   the color sample center from the first to last texel center of this viewport.
   Visible result: viewport borders no longer bleed adjacent transmission color.
   Breaking: no API/layout change; border pixels can change.

6. **Foreground silhouettes remain a documented limit.** Context: depth
   admission checks one opaque texel, while transmission color is linearly
   filtered. What is still wrong: an accepted sample can blend a neighboring
   foreground texel above the water at an interior silhouette. Why: one depth
   decision cannot prove that every color-filter contributor lies behind the
   water. Current choice: retain the existing admission and sampler rather than
   ship an unvalidated footprint rejection. Visible result: rare silhouette
   color bleed can remain even when `RefractionValidity` is accepted. Breaking:
   none; this item documents unchanged behavior.

## Water shading consistency

The water shader now samples body ownership as a discrete integer ID. Custom
field textures and shader integrations must keep ownership IDs exact. Do not
filter or blend IDs between neighboring texels.

Far water now applies each body's scatter scale and only enters the opaque far
path after checking the accepted camera or refraction path with the same
beauty-extinction rule as near water. Transparent beds can therefore remain on
the near path at distances where older versions switched to opaque far shading.

Resolved geometric and FFT wave slopes no longer fade out with lighting
distance. Only detail-normal and capillary amplitudes fade toward the far tier;
their removed energy transfers into unresolved roughness. Custom shader copies
of `NearSurface` must replace `lighting_normal_strength` with
`near_detail_weight` in the same field position and pass it to
`unresolved_wave_roughness`.

The default `WaterOptics::sun_roughness` is now `0.04` instead of `0.4`.
Existing scenes that relied on the old broad direct-sun highlight should set
`sun_roughness: 0.4` explicitly. Per-body non-negative overrides still win;
negative values inherit the ocean setting.

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

Near and far water roughness use the selected source variance with the existing
detail-strength multiplier, grazing boost, and roughness cap.
`WaterOptics::sun_roughness` controls direct-light highlight width. Resolved
geometric and FFT slopes remain in the lighting normal; detail and capillary
energy removed by mip or far-tier fades transfers into unresolved roughness.
