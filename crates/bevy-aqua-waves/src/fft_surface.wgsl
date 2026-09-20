// Aqua extension of Crest OceanHelpersNew.hlsl::SampleDisplacementsNormals.
// Crest evaluates this forward stencil per consumer. FFT caches it once after
// cumulative ShapeCombine so vertex shading, SSS, and foam share the result.
// Interpolating this half-float cache is an explicit Aqua approximation: it is
// not algebraically identical to forming nonlinear normals/Jacobians per sample.

const FFT_RESOLUTION: u32 = 256u;
const LOD_COUNT: u32 = 5u;
struct CascadeParams {
    center: vec2<f32>,
    scale: f32,
    texture_res: f32,
    inv_texture_res: f32,
    texel_width: f32,
    weight: f32,
    max_wavelength: f32,
}
struct CascadeLayout {
    cascades: array<CascadeParams, 6>,
    center: vec4<f32>,
    // XY bed-map first-texel world origin, ZW inverse world extent.
    bed_transform: vec4<f32>,
    // X height minimum, Y height span (negative = no bed map), Z sea level.
    bed_range: vec4<f32>,
}
struct FftUniform {
    cascade_layout: CascadeLayout,
    params: vec4<f32>,
    // x: active attenuation-bin count; reserved for future mode flags.
    mode: vec4<f32>,
}

@group(0) @binding(0) var displacement: texture_2d_array<f32>;
@group(0) @binding(1) var surface_derivatives: texture_storage_2d_array<rgba16float, write>;
@group(0) @binding(2) var<uniform> fft: FftUniform;
@group(0) @binding(3) var height_x: texture_2d_array<f32>;
@group(0) @binding(4) var z_field: texture_2d_array<f32>;

fn spectral_displacement(xy: vec2<i32>, field: u32) -> vec3<f32> {
    let bins = u32(fft.mode.x);
    let cascade = min(field, LOD_COUNT - 1u);
    var value = vec3(0.0);
    // Fine octaves use their complete exclusive field. The long band keeps
    // the existing four logarithmic attenuation bins when they are active.
    var first = 0u;
    var count = bins;
    if field >= LOD_COUNT - 1u {
        if bins == 1u && field > LOD_COUNT - 1u { return vec3(0.0); }
        first = field - (LOD_COUNT - 1u);
        count = 1u;
    }
    for (var bin = first; bin < first + count; bin++) {
        let layer = i32(cascade * bins + bin);
        let packed = textureLoad(height_x, xy, layer, 0);
        value += vec3(packed.y, packed.x, packed.z);
    }
    return value / f32(FFT_RESOLUTION * FFT_RESOLUTION);
}

fn spectral_derivatives(xy: vec2<i32>, field: u32) -> mat2x3<f32> {
    let bins = u32(fft.mode.x);
    let cascade = min(field, LOD_COUNT - 1u);
    var dx = vec3(0.0);
    var dz = vec3(0.0);
    var first = 0u;
    var count = bins;
    if field >= LOD_COUNT - 1u {
        if bins == 1u && field > LOD_COUNT - 1u { return mat2x3(vec3(0.0), vec3(0.0)); }
        first = field - (LOD_COUNT - 1u);
        count = 1u;
    }
    for (var bin = first; bin < first + count; bin++) {
        let layer = i32(cascade * bins + bin);
        let a = textureLoad(height_x, xy, layer, 0);
        let b = textureLoad(z_field, xy, layer, 0);
        dx += vec3(b.y, a.w, b.z);
        dz += vec3(b.z, b.x, b.w);
    }
    let scale = 1.0 / f32(FFT_RESOLUTION * FFT_RESOLUTION);
    return mat2x3(dx * scale, dz * scale);
}

fn store_analytic(xy: vec2<i32>, field: u32) {
    let derivatives = spectral_derivatives(xy, field);
    let dx = derivatives[0];
    let dz = derivatives[1];
    let layer = i32(LOD_COUNT + 2u * field);
    textureStore(surface_derivatives, xy, layer, vec4(dx, dx.y * dx.y));
    textureStore(surface_derivatives, xy, layer + 1, vec4(dz, dz.y * dz.y));
}

// Bed-only overwrite follows residual construction. Legacy layers0..4 and
// the displacement image are never written by this entry point.
@compute @workgroup_size(8, 8, 1)
fn resolve_analytic(@builtin(global_invocation_id) id: vec3<u32>) {
    if any(id.xy >= vec2<u32>(FFT_RESOLUTION)) || id.z >= 8u { return; }
    store_analytic(vec2<i32>(id.xy), id.z);
}

@compute @workgroup_size(8, 8, 1)
fn resolve_surface(@builtin(global_invocation_id) id: vec3<u32>) {
    if any(id.xy >= vec2<u32>(FFT_RESOLUTION)) || id.z >= LOD_COUNT + 8u { return; }
    let xy = vec2<i32>(id.xy);
    if id.z >= LOD_COUNT {
        let field = id.z - LOD_COUNT;
        if u32(fft.mode.x) == 1u {
            // Deep/no-bed bypass: no FD prediction or residual work.
            store_analytic(xy, field);
            return;
        }
        let cascade = fft.cascade_layout.cascades[min(field, LOD_COUNT - 1u)];
        let center = spectral_displacement(xy, field);
        let next_x = (xy + vec2(1, 0)) % vec2(i32(FFT_RESOLUTION));
        let next_z = (xy + vec2(0, 1)) % vec2(i32(FFT_RESOLUTION));
        let dx = (spectral_displacement(next_x, field) - center) / cascade.texel_width;
        let dz = (spectral_displacement(next_z, field) - center) / cascade.texel_width;
        let output_layer = i32(LOD_COUNT + 2u * field);
        textureStore(surface_derivatives, xy, output_layer, vec4(dx, dx.y * dx.y));
        textureStore(surface_derivatives, xy, output_layer + 1, vec4(dz, dz.y * dz.y));
        return;
    }
    // Legacy normal/Jacobian cache is unchanged for vertices, foam and SSS.
    let cascade = fft.cascade_layout.cascades[id.z];
    let maximum = i32(FFT_RESOLUTION) - 1;
    let center = textureLoad(displacement, xy, i32(id.z), 0).xyz;
    let offset_x = textureLoad(displacement, min(xy + vec2(1, 0), vec2(maximum)), i32(id.z), 0).xyz;
    let offset_z = textureLoad(displacement, min(xy + vec2(0, 1), vec2(maximum)), i32(id.z), 0).xyz;
    let derivative_x = (offset_x - center) / cascade.texel_width;
    let derivative_z = (offset_z - center) / cascade.texel_width;
    let determinant = (1.0 + derivative_x.x) * (1.0 + derivative_z.z)
        - derivative_x.z * derivative_z.x;
    let tangent_x = vec3(1.0, 0.0, 0.0) + derivative_x;
    let tangent_z = vec3(0.0, 0.0, 1.0) + derivative_z;
    textureStore(surface_derivatives, xy, i32(id.z),
        vec4(cross(tangent_z, tangent_x), determinant));
}
