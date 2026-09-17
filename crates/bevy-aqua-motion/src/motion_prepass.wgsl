// Aqua's motion-vector prepass.
// Current geometry uses the shared forward deformation. Previous geometry is
// evaluated at the same stable world anchor from the retained wave history.

#import bevy_pbr::{
    forward_io,
    prepass_bindings,
    mesh_functions,
    prepass_io::{Vertex, VertexOutput, FragmentOutput},
    mesh_view_bindings::view,
    view_transformations::position_world_to_clip,
}
#import bevy_aqua_core::deform::{deform_current}
#import bevy_render::globals::Globals
#import bevy_aqua_core::waves_sample::{CascadeLayout, lod_alpha, world_to_uv}
#import aqua::cascade::{DEBUG_MODE_BEAUTY, DEBUG_MODE_FAR_TIER, MIN_SAMPLE_WEIGHT, begin_invocation, cascade_layout, far_tier_weight, field_params, lod_count, owning_body, sample_field_flow, sample_field_level, surface}
#import bevy_aqua_core::river::river_analytic_displacement

// Retained AnimWaves output and the exact layout/time/flow that produced it.
struct AquaPreviousFrame {
    previous_layout: CascadeLayout,
    time: vec4<f32>,
    flow: vec4<f32>,
    valid: vec4<f32>,
}
@group(4) @binding(0) var aqua_previous_displacement: texture_2d_array<f32>;
@group(4) @binding(1) var aqua_previous_sampler: sampler;
@group(4) @binding(2) var<uniform> aqua_previous_frame: AquaPreviousFrame;
@group(0) @binding(1) var<uniform> prepass_globals: Globals;

fn previous_direct_displacement(world_xz: vec2<f32>, lod: u32) -> vec3<f32> {
    // History contains the already-evolved wave field. Only global current
    // advection remains to place that field in world space at the saved time.
    let sampled_xz = world_xz
        - aqua_previous_frame.flow.xy * aqua_previous_frame.time.x;
    let previous_cascade = aqua_previous_frame.previous_layout.cascades[lod];
    return textureSampleLevel(
        aqua_previous_displacement,
        aqua_previous_sampler,
        world_to_uv(sampled_xz, previous_cascade),
        i32(lod),
        0.0,
    ).xyz;
}

fn previous_near_displacement(
    world_xz: vec2<f32>,
    lod: u32,
    alpha: f32,
) -> vec3<f32> {
    // LOD selection and weights describe today's mesh representation. History
    // layout is used only to map the stable world sample into retained texels.
    let smaller = cascade_layout.cascades[lod];
    let bigger = cascade_layout.cascades[lod + 1u];
    let smaller_weight = (1.0 - alpha) * smaller.weight;
    let bigger_weight = (1.0 - smaller_weight) * bigger.weight;
    var displacement = vec3(0.0);
    if smaller_weight > MIN_SAMPLE_WEIGHT {
        displacement += smaller_weight
            * previous_direct_displacement(world_xz, lod);
    }
    if bigger_weight > MIN_SAMPLE_WEIGHT {
        displacement += bigger_weight
            * previous_direct_displacement(world_xz, lod + 1u);
    }
    return displacement;
}

fn previous_far_displacement(world_xz: vec2<f32>) -> vec3<f32> {
    let outer_lod = lod_count() - 1u;
    // Keep the current representation's outer-ring fade, but read its wave
    // value from the retained outer history layer in the retained layout.
    let weight = 1.0 - lod_alpha(
        world_xz,
        cascade_layout.cascades[outer_lod],
        cascade_layout,
    );
    if weight <= MIN_SAMPLE_WEIGHT {
        return vec3(0.0);
    }
    return weight * previous_direct_displacement(world_xz, outer_lod);
}

@vertex
fn vertex(vertex_in: Vertex) -> VertexOutput {
    var out: VertexOutput;
    // The shared deform contract uses the forward vertex ABI. Translate the
    // prepass input explicitly because Bevy assigns different attribute
    // locations to the two stage ABIs.
    var forward_vertex: forward_io::Vertex;
    forward_vertex.instance_index = vertex_in.instance_index;
    forward_vertex.position = vertex_in.position;
#ifdef VERTEX_UVS_A
    forward_vertex.uv = vertex_in.uv;
#endif
#ifdef VERTEX_UVS_B
    forward_vertex.uv_b = vertex_in.uv_b;
#endif
#ifdef VERTEX_COLORS
    forward_vertex.color = vertex_in.color;
#endif
#ifdef SKINNED
    forward_vertex.joint_indices = vertex_in.joint_indices;
    forward_vertex.joint_weights = vertex_in.joint_weights;
#endif
#ifdef MORPH_TARGETS
    forward_vertex.index = vertex_in.index;
#endif
    let deformation = deform_current(forward_vertex, prepass_globals.time);
    out.world_position = deformation.world_position;
    out.position = position_world_to_clip(out.world_position.xyz);
#ifdef MOTION_VECTOR_PREPASS
    var previous_world_position = out.world_position;
    // Do not touch history until it contains a completed prior AnimWaves frame.
    if aqua_previous_frame.valid.x >= 0.5 {
        if deformation.river {
            // Re-evaluate only height at the saved time. XZ and the water level
            // are current stable anchors, never a previous ring transform.
            let previous_river = river_analytic_displacement(
                deformation.sample_xz,
                deformation.river_flow,
                aqua_previous_frame.time.x,
            );
            previous_world_position = vec4(
                deformation.sample_xz.x,
                deformation.body_level + previous_river.x,
                deformation.sample_xz.y,
                1.0,
            );
        } else if !deformation.bounded {
            let mode = u32(round(surface.debug.x));
            let far_diagnostic = mode == DEBUG_MODE_FAR_TIER;
            let far_tier = select(
                0.0,
                far_tier_weight(deformation.base_world_position),
                mode == DEBUG_MODE_BEAUTY || far_diagnostic,
            );
            var previous_displacement: vec3<f32>;
            if far_tier >= 1.0 {
                previous_displacement = previous_far_displacement(
                    deformation.sample_xz,
                );
            } else {
                previous_displacement = previous_near_displacement(
                    deformation.sample_xz,
                    deformation.lod,
                    deformation.sample_alpha,
                );
                if far_tier > 0.0 {
                    previous_displacement = mix(
                        previous_displacement,
                        previous_far_displacement(deformation.sample_xz),
                        far_tier,
                    );
                }
            }
            previous_world_position = vec4(
                deformation.base_world_position + previous_displacement,
                1.0,
            );
        }
        // Bounded flat water intentionally retains current == previous.
    }
    out.previous_world_position = previous_world_position;
#endif
#ifdef VERTEX_OUTPUT_INSTANCE_INDEX
    out.instance_index = vertex_in.instance_index;
#endif
#ifdef VISIBILITY_RANGE_DITHER
    let world_from_local = mesh_functions::get_world_from_local(vertex_in.instance_index);
    out.visibility_range_dither = mesh_functions::get_visibility_range_dither_level(
        vertex_in.instance_index,
        world_from_local[3],
    );
#endif
    return out;
}

@fragment
fn fragment(in: VertexOutput) -> FragmentOutput {
    // Match forward coverage for the shared ocean/body tile set. In the main
    // ocean showcase this is exactly the current forward runtime path.
    var slot = 0u;
    if field_params.info.x > 0.5 {
        slot = u32(sample_field_level(in.world_position.xz).y + 0.5);
    }
    let bounded = slot > 0u;
    if !bounded && field_params.info.y < 0.5 {
        discard;
    }
    let params = owning_body(slot);
    begin_invocation(bounded, params);
    if bounded {
        let distance_to_extent = length(
            params.extent.xy - view.world_position.xz,
        ) - params.extent.w;
        if distance_to_extent > surface.far_tier.y {
            discard;
        }
        let river_flow = sample_field_flow(in.world_position.xz);
        if params.flags.y > 0.5 && river_flow.z < 0.0 {
            discard;
        }
    }

    var out: FragmentOutput;
#ifdef MOTION_VECTOR_PREPASS
    let clip_t = view.unjittered_clip_from_world * in.world_position;
    let clip = clip_t.xy / clip_t.w;
    let previous_clip_t = prepass_bindings::previous_view_uniforms.clip_from_world
        * in.previous_world_position;
    let saved_previous_clip = previous_clip_t.xy / previous_clip_t.w;
    // Invalid history is a full temporal reset, including camera teleport.
    let previous_clip = select(
        clip,
        saved_previous_clip,
        aqua_previous_frame.valid.x >= 0.5,
    );
    // Bevy/Terra ABI: UV-space offset; clip Y is inverted and NDC is halved.
    let motion = (clip - previous_clip) * vec2(0.5, -0.5);
    // Canonical positive zero is part of the raw Rg16Float reset contract.
    out.motion_vector = select(motion, vec2(0.0), motion == vec2(0.0));
#endif
    return out;
}
