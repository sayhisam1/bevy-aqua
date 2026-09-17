// Closed-form water medium: RGB Beer-Lambert transmittance and in-scatter
// from directional downwelling.
// The underwater pass evaluates this in water. The cascade surface converts
// the same integral to air as water-leaving radiance along the camera ray
// that sampled transmission. Opaque colour is scaled by the
// surface-to-hit sun path before the camera-path integral.

#define_import_path aqua::medium

#import bevy_pbr::mesh_view_bindings::{lights, view}

const FRAC_4_PI: f32 = 0.07957747154594767;
const FRAC_3_16_PI: f32 = 0.05968310365946022;
const PATH_LENGTH_MAX: f32 = 256.0;
const N_WATER: f32 = 1.333;
const MIN_L_Y: f32 = 0.02;
const KAPPA_EPS: f32 = 1e-5;
const OPTICAL_CLAMP: f32 = 80.0;
const SCATTER_EPS: f32 = 1e-8;
// Particle scatter at 550 nm for scatter_scale = 1 and green tint 1, in 1/m.
const PARTICLE_SCATTER: f32 = 0.02;
// Smith and Baker 1981 molecular scatter at 650/550/450 nm, in 1/m.
const RAYLEIGH: vec3<f32> = vec3(0.00095, 0.00193, 0.00456);

fn henyey_greenstein(l_dot_rd: f32, g: f32) -> f32 {
    let denom = 1.0 + g * g - 2.0 * g * l_dot_rd;
    return FRAC_4_PI * (1.0 - g * g) / (denom * sqrt(denom));
}

fn phase_rayleigh(cos_theta: f32) -> f32 {
    return FRAC_3_16_PI * (1.0 + cos_theta * cos_theta);
}

fn mixed_phase(cos_theta: f32, sigma_p: vec3<f32>, g: f32) -> vec3<f32> {
    let denom = max(sigma_p + RAYLEIGH, vec3(SCATTER_EPS));
    return (sigma_p * henyey_greenstein(cos_theta, g)
        + RAYLEIGH * phase_rayleigh(cos_theta))
        / denom;
}

fn fresnel_dielectric(n1: f32, n2: f32, cos_i: f32) -> f32 {
    let eta = n1 / n2;
    let sin2_t = eta * eta * (1.0 - cos_i * cos_i);
    if sin2_t >= 1.0 {
        return 1.0;
    }
    let cos_t = sqrt(1.0 - sin2_t);
    let rs = (n1 * cos_i - n2 * cos_t) / (n1 * cos_i + n2 * cos_t);
    let rp = (n2 * cos_i - n1 * cos_t) / (n2 * cos_i + n1 * cos_t);
    return 0.5 * (rs * rs + rp * rp);
}

fn fresnel_air_to_water(cos_air: f32) -> f32 {
    return fresnel_dielectric(1.0, N_WATER, cos_air);
}

fn fresnel_water_to_air(cos_water: f32) -> f32 {
    return fresnel_dielectric(N_WATER, 1.0, cos_water);
}

fn downwelling_transmittance(sigma: vec3<f32>, depth: f32, l_y: f32) -> vec3<f32> {
    return exp(-sigma * (max(depth, 0.0) / max(l_y, MIN_L_Y)));
}

// Direct sunlight just under the surface, after Fresnel into water and the
// slanted Beer-Lambert path `depth / L.y`. Above water this is 1.
fn sun_to_water_transmittance(
    sigma: vec3<f32>,
    depth: f32,
    l_air: vec3<f32>,
) -> vec3<f32> {
    if l_air.y <= 0.0 || depth <= 0.0 {
        return vec3(1.0);
    }
    let l_water = refract_air_to_water(l_air);
    return (1.0 - fresnel_air_to_water(l_air.y))
        * downwelling_transmittance(sigma, depth, l_water.y);
}

// Incident scale for opaque mesh shading. The strongest upward directional
// light uses the slanted sun path; with no sun, vertical sky downwelling.
fn mesh_incident_transmittance(sigma: vec3<f32>, depth: f32) -> vec3<f32> {
    if depth <= 0.0 {
        return vec3(1.0);
    }
    var transmittance = downwelling_transmittance(sigma, depth, 1.0);
    var best_illuminance = 0.0;
    let directional_light_count = lights.n_directional_lights;
    for (var light_index = 0u; light_index < directional_light_count; light_index += 1u) {
        let light = &lights.directional_lights[light_index];
        let l_air = (*light).direction_to_light.xyz;
        if l_air.y <= 0.0 {
            continue;
        }
        let illuminance = l_air.y * dot(
            max((*light).color.rgb, vec3(0.0)),
            vec3(0.2126, 0.7152, 0.0722),
        );
        if illuminance > best_illuminance {
            best_illuminance = illuminance;
            transmittance = sun_to_water_transmittance(sigma, depth, l_air);
        }
    }
    return transmittance;
}

fn attenuate_underwater_scene(
    scene: vec3<f32>,
    hit_y: f32,
    surface_y: f32,
    sigma: vec3<f32>,
) -> vec3<f32> {
    return scene * mesh_incident_transmittance(sigma, max(surface_y - hit_y, 0.0));
}

fn refract_air_to_water(l_air: vec3<f32>) -> vec3<f32> {
    let sin2_air = max(1.0 - l_air.y * l_air.y, 0.0);
    let sin2_water = sin2_air / (N_WATER * N_WATER);
    let cos_water = sqrt(max(1.0 - sin2_water, 0.0));
    let horiz_len = length(l_air.xz);
    var xz = vec2(0.0);
    if horiz_len > 1e-8 {
        xz = l_air.xz / horiz_len * sqrt(sin2_water);
    }
    return vec3(xz.x, cos_water, xz.y);
}

// Surface observer: same camera-ray integral as underwater, then n² into air.
// Fresnel transmittance is the later (1-F) mix, not applied here.
fn water_leaving_radiance(
    scene: vec3<f32>,
    to_view: vec3<f32>,
    t_end: f32,
    sigma_t: vec3<f32>,
    scatter_scale: f32,
    scatter_tint: vec3<f32>,
    g: f32,
) -> vec3<f32> {
    let t = min(max(t_end, 0.0), PATH_LENGTH_MAX);
    return medium_radiance(
        scene,
        -to_view,
        t,
        0.0,
        sigma_t,
        scatter_scale,
        scatter_tint,
        g,
    ) / (N_WATER * N_WATER);
}

// ∫_0^t I0 exp(-σ (d0 - s rd.y) / L.y) exp(-σ s) ds
// with I0 the irradiance just under the surface along L.
fn downwelling_integral(
    sigma: vec3<f32>,
    t: f32,
    rd_y: f32,
    l_y: f32,
    d0: f32,
    irradiance: vec3<f32>,
) -> vec3<f32> {
    let ly = max(l_y, MIN_L_Y);
    let i0 = irradiance * exp(-sigma * (d0 / ly));
    let kappa = sigma * (1.0 - rd_y / ly);
    let optical = clamp(kappa * t, vec3(-OPTICAL_CLAMP), vec3(OPTICAL_CLAMP));
    let use_series = abs(kappa) <= vec3(KAPPA_EPS);
    let kappa_safe = select(kappa, vec3(1.0), use_series);
    let integral = select(
        (vec3(1.0) - exp(-optical)) / kappa_safe,
        vec3(t),
        use_series,
    );
    return i0 * integral;
}

fn medium_radiance(
    scene: vec3<f32>,
    rd: vec3<f32>,
    t_end: f32,
    d0: f32,
    sigma_t: vec3<f32>,
    scatter_scale: f32,
    scatter_tint: vec3<f32>,
    g: f32,
) -> vec3<f32> {
    if t_end < 1e-4 {
        return scene;
    }

    let sigma_p = PARTICLE_SCATTER * max(scatter_scale, 0.0) * max(scatter_tint, vec3(0.0));
    let sigma_s = min(sigma_t, sigma_p + RAYLEIGH);
    let exposure = view.exposure;

    var inscatter = vec3<f32>(0.0);
    let directional_light_count = lights.n_directional_lights;
    for (var light_index = 0u; light_index < directional_light_count; light_index += 1u) {
        let light = &lights.directional_lights[light_index];
        let l_air = (*light).direction_to_light.xyz;
        if l_air.y <= 0.0 {
            continue;
        }
        let e_air = (*light).color.rgb * exposure;
        let l_water = refract_air_to_water(l_air);
        let e_water = e_air * (1.0 - fresnel_air_to_water(l_air.y));
        let phase = mixed_phase(dot(l_water, rd), sigma_p, g);
        inscatter += sigma_s * downwelling_integral(
            sigma_t,
            t_end,
            rd.y,
            l_water.y,
            d0,
            e_water * phase,
        );
    }

    let transmittance = exp(-sigma_t * t_end);
    return scene * transmittance + inscatter;
}
