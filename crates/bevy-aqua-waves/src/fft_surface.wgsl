// Aqua extension of Crest OceanHelpersNew.hlsl::SampleDisplacementsNormals.
// Crest evaluates this forward stencil per consumer. FFT caches it once after
// cumulative ShapeCombine so vertex shading, SSS, and foam share the result.
// Interpolating this half-float cache is an explicit Aqua approximation: it is
// not algebraically identical to forming nonlinear normals/Jacobians per sample.

#import bevy_aqua_core::waves_sample::{CascadeLayout, LOD_COUNT}

const FFT_RESOLUTION: u32 = 256u;
struct FftUniform {
    cascade_layout: CascadeLayout,
    params: vec4<f32>,
    // x: active attenuation-bin count; reserved for future mode flags.
    mode: vec4<f32>,
}

@group(0) @binding(0) var displacement: texture_2d_array<f32>;
@group(0) @binding(1) var surface_derivatives: texture_storage_2d_array<rgba16float, write>;
@group(0) @binding(2) var<uniform> fft: FftUniform;

@compute @workgroup_size(8, 8, 1)
fn resolve_surface(@builtin(global_invocation_id) id: vec3<u32>) {
    if any(id.xy >= vec2<u32>(FFT_RESOLUTION)) || id.z >= LOD_COUNT {
        return;
    }
    let cascade = fft.cascade_layout.cascades[id.z];
    let maximum = i32(FFT_RESOLUTION) - 1;
    let center_coord = vec2<i32>(id.xy);
    let x_coord = min(center_coord + vec2(1, 0), vec2(maximum));
    let z_coord = min(center_coord + vec2(0, 1), vec2(maximum));
    let center = textureLoad(displacement, center_coord, i32(id.z), 0).xyz;
    let offset_x = textureLoad(displacement, x_coord, i32(id.z), 0).xyz;
    let offset_z = textureLoad(displacement, z_coord, i32(id.z), 0).xyz;
    let derivative_x = (offset_x - center) / cascade.texel_width;
    let derivative_z = (offset_z - center) / cascade.texel_width;
    let determinant = (1.0 + derivative_x.x) * (1.0 + derivative_z.z)
        - derivative_x.z * derivative_z.x;
    let tangent_x = vec3(1.0, 0.0, 0.0) + derivative_x;
    let tangent_z = vec3(0.0, 0.0, 1.0) + derivative_z;
    let normal_cross = cross(tangent_z, tangent_x);
    textureStore(
        surface_derivatives,
        vec2<i32>(id.xy),
        i32(id.z),
        vec4(normal_cross, determinant),
    );
}
