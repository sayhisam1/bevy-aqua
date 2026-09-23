// Opt-in screen-space reflections: march a mirrored ray through the opaque
// depth prepass and report where it first passes behind visible geometry.
//
// The march takes quadratic steps along the ray, so samples are dense near
// the surface where contact reflections matter. At the first step behind the
// depth buffer, bisection narrows the crossing and a thickness test decides
// whether it is a surface or a ray passing behind a foreground silhouette.

#define_import_path aqua::ssr

#import bevy_pbr::{
    prepass_utils,
    mesh_view_bindings::view,
}
#import aqua::cascade::{LUMINANCE_EPSILON, surface}
#import aqua::medium::PATH_LENGTH_MAX
#import aqua::light::incident::safe_normalize
#import aqua::screen::camera_view_position

/// Linear march samples. With quadratic spacing the last interval spans
/// about 7.6 m at SSR_MAX_DISTANCE; fewer steps lets thin objects fall
/// between samples.
const SSR_LINEAR_STEPS: u32 = 12u;
/// Bisections of the crossing interval: 7.6 m / 2^5 ≈ 0.24 m, which must stay
/// below SSR_THICKNESS or valid far hits are rejected.
const SSR_BISECTION_STEPS: u32 = 5u;
/// First sample in metres. Skips the texels around the reflecting fragment,
/// whose depth belongs to what lies beneath the water.
const SSR_START: f32 = 0.3;
/// March length in metres. Past this, reflected geometry covers few pixels
/// and the planar or environment reflection already carries it.
const SSR_MAX_DISTANCE: f32 = 48.0;
/// Eye-depth band in metres behind a depth sample still treated as its
/// surface. Wider bands smear reflections behind thin foreground silhouettes.
const SSR_THICKNESS: f32 = 0.4;
/// Screen-UV inset over which hits fade out, so reflections do not pop as
/// their source leaves the viewport.
const SSR_EDGE_FADE: f32 = 0.08;
/// Metres past a rejection plane still accepted, absorbing reconstruction
/// error for hits that lie on the plane itself.
const SSR_PLANE_TOLERANCE: f32 = 0.02;
/// A single sharp ray cannot represent a broad lobe. Hits fade out from this
/// fraction of the surface roughness cap (`surface.reflection.w`) to the cap.
const SSR_ROUGHNESS_FADE_START: f32 = 0.35;
/// Lower bound on the roughness cap so a zero cap cannot disable the fade.
const SSR_MIN_ROUGHNESS_CAP: f32 = 0.05;
/// A floor far below any scene: disables the height rejection.
const SSR_NO_FLOOR: f32 = -3.0e38;

struct SsrRay {
    origin: vec3<f32>,
    /// Unit or near-unit world direction; normalized by the march.
    direction: vec3<f32>,
    /// Hits more than SSR_PLANE_TOLERANCE past the plane through `origin`
    /// with this normal are rejected. Zero keeps every hit.
    outward: vec3<f32>,
    /// Hits below this world height in metres are rejected; SSR_NO_FLOOR
    /// keeps every hit.
    floor: f32,
}

struct SsrProbe {
    valid: bool,
    /// True when the ray sample is in front of (or cannot be compared with)
    /// the depth buffer.
    in_front: bool,
    uv: vec2<f32>,
    /// Ray eye depth minus scene eye depth in metres; positive is behind.
    gap: f32,
}

/// A screen-space hit: the opaque-buffer UV to sample, its blend weight in
/// `0..=1`, and the distance travelled along the ray in metres.
struct ScreenSpaceHit {
    uv: vec2<f32>,
    weight: f32,
    distance: f32,
}

fn ssr_miss() -> ScreenSpaceHit {
    return ScreenSpaceHit(vec2(0.0), 0.0, PATH_LENGTH_MAX);
}

fn ssr_rejected(ray: SsrRay, world: vec3<f32>) -> bool {
    let guarded = dot(ray.outward, ray.outward) > 0.5;
    let past_plane = guarded && dot(world - ray.origin, ray.outward) > SSR_PLANE_TOLERANCE;
    return past_plane || world.y < ray.floor;
}

// Compares one ray sample with the depth prepass at its projected pixel.
fn ssr_probe(ray: SsrRay, travel: f32) -> SsrProbe {
    var probe = SsrProbe(false, false, vec2(0.0), 0.0);
    let world = ray.origin + ray.direction * travel;
    if ssr_rejected(ray, world) {
        return probe;
    }
#ifdef DEPTH_PREPASS
    let clip = view.clip_from_world * vec4(world, 1.0);
    if clip.w <= LUMINANCE_EPSILON {
        return probe;
    }
    let ndc = clip.xyz / clip.w;
    let uv = vec2(ndc.x, -ndc.y) * 0.5 + 0.5;
    if any(uv < vec2(0.0)) || any(uv > vec2(1.0)) {
        return probe;
    }
    let pixel = min(uv * view.viewport.zw, view.viewport.zw - vec2(1.0)) + view.viewport.xy;
    let scene_depth = prepass_utils::prepass_depth(vec4(pixel, 0.0, 1.0), 0u);
    probe.valid = true;
    probe.uv = uv;
    probe.in_front = true;
    probe.gap = -1.0;
    // Reverse-Z zero is the sky: nothing to hit.
    if scene_depth <= 0.0 {
        return probe;
    }
    let scene_view = camera_view_position(uv, scene_depth);
    let scene_world = (view.world_from_view * vec4(scene_view, 1.0)).xyz;
    // A rejected receiver is transparent to the march: the ray passes over it.
    if ssr_rejected(ray, scene_world) {
        return probe;
    }
    let ray_eye = -(view.view_from_world * vec4(world, 1.0)).z;
    probe.gap = ray_eye + scene_view.z;
    probe.in_front = probe.gap <= 0.0;
#endif
    return probe;
}

// Quadratic spacing: travel for linear step `step` in `1..=SSR_LINEAR_STEPS`.
fn ssr_travel(step: u32) -> f32 {
    let u = f32(step) / f32(SSR_LINEAR_STEPS);
    return SSR_START + (SSR_MAX_DISTANCE - SSR_START) * u * u;
}

// Bisects [front, behind] to the depth crossing; returns the behind bound.
fn ssr_refine(ray: SsrRay, front: f32, behind: f32) -> f32 {
    var low = front;
    var high = behind;
    for (var i = 0u; i < SSR_BISECTION_STEPS; i++) {
        let middle = 0.5 * (low + high);
        let probe = ssr_probe(ray, middle);
        let crossed = probe.valid && !probe.in_front;
        low = select(middle, low, crossed);
        high = select(high, middle, crossed);
    }
    return high;
}

// Accepts a refined crossing inside the thickness band and weights it.
fn ssr_resolve(ray: SsrRay, travel: f32, roughness_weight: f32) -> ScreenSpaceHit {
    let hit = ssr_probe(ray, travel);
    let surface_hit = hit.valid && hit.gap > 0.0 && hit.gap < SSR_THICKNESS;
    if !surface_hit {
        return ssr_miss();
    }
    let inset = min(min(hit.uv.x, 1.0 - hit.uv.x), min(hit.uv.y, 1.0 - hit.uv.y));
    let weight = roughness_weight * smoothstep(0.0, SSR_EDGE_FADE, inset);
    return ScreenSpaceHit(hit.uv, weight, min(travel, PATH_LENGTH_MAX));
}

// Returns a miss unless `AquaSettings::screen_space_reflections` is on
// (`surface.far_tier.z`) and the camera has a depth prepass.
fn screen_space_reflection(ray_in: SsrRay, roughness: f32) -> ScreenSpaceHit {
    if surface.far_tier.z < 0.5 {
        return ssr_miss();
    }
    let fade_end = max(surface.reflection.w, SSR_MIN_ROUGHNESS_CAP);
    let roughness_weight = 1.0
        - smoothstep(fade_end * SSR_ROUGHNESS_FADE_START, fade_end, roughness);
    if roughness_weight <= 0.0 {
        return ssr_miss();
    }
#ifdef DEPTH_PREPASS
    var ray = ray_in;
    ray.direction = safe_normalize(ray.direction, vec3(0.0, 1.0, 0.0));
    var front = SSR_START;
    for (var step = 1u; step <= SSR_LINEAR_STEPS; step++) {
        let travel = ssr_travel(step);
        let probe = ssr_probe(ray, travel);
        if !probe.valid {
            break;
        }
        // Only the first crossing is considered; a rejected one is a miss.
        if !probe.in_front {
            return ssr_resolve(ray, ssr_refine(ray, front, travel), roughness_weight);
        }
        front = travel;
    }
#endif
    return ssr_miss();
}
