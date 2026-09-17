// Shared current-frame water-surface deformation. This module is the single
// source of truth for forward rendering and later motion-vector consumers.

#define_import_path bevy_aqua_core::deform

#import bevy_pbr::{forward_io::Vertex, mesh_functions}
#import bevy_aqua_core::waves_sample::sample_displacement
#import aqua::cascade::{BodyParams, DEBUG_MODE_BEAUTY, DEBUG_MODE_FAR_TIER, MIN_NORMAL_Y, SAFE_LENGTH_SQUARED, advected_world, begin_invocation, cascade_layout, effective_flow, far_tier_weight, field_params, lod_count, lod_data, lod_sampler, owning_body, sample_field_flow, sample_field_level, set_effective_flow, set_effective_time, snap_and_transition, surface}
#import aqua::waves::displace::{far_displacement, far_normal_cross, sample_fft_normal_cross}
#import bevy_aqua_core::river::river_analytic_displacement

// The full current-frame result is intentionally independent of a render-stage
// output. `sample_xz` is the stable location used to evaluate deformation;
// `undisplaced_xz` preserves the forward material's existing varying contract.
struct DeformationResult {
    world_position: vec4<f32>,
    base_world_position: vec3<f32>,
    world_normal: vec3<f32>,
    sample_xz: vec2<f32>,
    undisplaced_xz: vec2<f32>,
    lod: u32,
    sample_alpha: f32,
    displacement: vec3<f32>,
    wave_height: f32,
    body_slot: u32,
    body_params: BodyParams,
    body_level: f32,
    bounded: bool,
    river: bool,
    river_flow: vec4<f32>,
}

fn deformation_safe_normalize(value: vec3<f32>, fallback: vec3<f32>) -> vec3<f32> {
    let length_squared = dot(value, value);
    let normalized = value * inverseSqrt(max(length_squared, SAFE_LENGTH_SQUARED));
    return select(fallback, normalized, length_squared > SAFE_LENGTH_SQUARED);
}

fn deform_current(vertex: Vertex, time: f32) -> DeformationResult {
    var result: DeformationResult;
    let world_from_local = mesh_functions::get_world_from_local(vertex.instance_index);
    var world_position = mesh_functions::mesh_position_local_to_world(
        world_from_local,
        vec4(vertex.position, 1.0),
    );
    let tile_lod = min(mesh_functions::get_tag(vertex.instance_index), lod_count() - 1u);
    var slot = 0u;
    var params = field_params.bodies[0];
    var vertex_level = 0.0;
    if field_params.info.x > 0.5 {
        let field = sample_field_level(world_position.xz);
        slot = u32(field.y + 0.5);
        params = owning_body(slot);
        vertex_level = field.x;
    }
    let bounded = slot > 0u;
    begin_invocation(bounded, params);
    set_effective_time(time);
    // River bodies advect wave content with their baked local current;
    // everything else keeps the global uniform current.
    var vertex_river_flow = vec4(0.0, 0.0, 1024.0, 0.0);
    if bounded {
        vertex_river_flow = sample_field_flow(world_position.xz);
    }
    set_effective_flow(select(
        surface.advection.xy,
        vertex_river_flow.xy,
        bounded && params.flags.y > 0.5,
    ));

    result.body_slot = slot;
    result.body_params = params;
    result.body_level = vertex_level;
    result.bounded = bounded;
    result.river = bounded && params.flags.y > 0.5;
    result.river_flow = vertex_river_flow;

    // River bodies synthesize their own closed-form waves along the local
    // current instead of sampling the shared ocean cascade.
    if result.river {
        let transitioned = snap_and_transition(
            world_position.xz,
            world_from_local[3].xz,
            cascade_layout.cascades[tile_lod],
        );
        let river_xz = transitioned.xy;
        vertex_river_flow = sample_field_flow(river_xz);
        effective_flow = vertex_river_flow.xy;
        let analytic = river_analytic_displacement(
            river_xz,
            vertex_river_flow,
            time,
        );
        let epsilon = 0.35;
        let height_x = river_analytic_displacement(
            river_xz + vec2(epsilon, 0.0),
            vertex_river_flow,
            time,
        ).x;
        let height_z = river_analytic_displacement(
            river_xz + vec2(0.0, epsilon),
            vertex_river_flow,
            time,
        ).x;
        let displaced = vec4(
            river_xz.x,
            vertex_level + analytic.x,
            river_xz.y,
            1.0,
        );
        result.world_position = displaced;
        result.base_world_position = world_position.xyz;
        result.world_normal = deformation_safe_normalize(
            vec3(
                -(height_x - analytic.x) / epsilon,
                1.0,
                -(height_z - analytic.x) / epsilon,
            ),
            vec3(0.0, 1.0, 0.0),
        );
        result.sample_xz = river_xz;
        result.undisplaced_xz = world_position.xz;
        result.lod = 0u;
        result.sample_alpha = 1.0;
        result.displacement = displaced.xyz - world_position.xyz;
        result.wave_height = analytic.x;
        result.river_flow = vertex_river_flow;
        return result;
    }

    if bounded {
        let transitioned = snap_and_transition(
            world_position.xz,
            world_from_local[3].xz,
            cascade_layout.cascades[tile_lod],
        );
        let displaced = vec4(transitioned.x, vertex_level, transitioned.y, 1.0);
        result.world_position = displaced;
        result.base_world_position = displaced.xyz;
        result.world_normal = vec3(0.0, 1.0, 0.0);
        result.sample_xz = transitioned.xy;
        result.undisplaced_xz = world_position.xz;
        result.lod = 0u;
        result.sample_alpha = 1.0;
        result.displacement = vec3(0.0);
        result.wave_height = 0.0;
        return result;
    }

    let transitioned = select(
        snap_and_transition(
            world_position.xz,
            world_from_local[3].xz,
            cascade_layout.cascades[tile_lod],
        ),
        vec3(world_position.x, world_position.z, 1.0),
        bounded,
    );
    world_position = vec4(
        transitioned.x,
        world_position.y,
        transitioned.y,
        world_position.w,
    );
    // Crest fades wave content that projects below the minimum useful
    // wavelength before the coarse vertex grid can alias it.
    let detail_lod = clamp(cascade_layout.center.z, 0.0, f32(lod_count() - 1u));
    let minimum_lod = u32(floor(detail_lod));
    let lod = max(tile_lod, minimum_lod);
    let detail_alpha = fract(detail_lod);
    var sample_alpha = select(transitioned.z, 1.0, bounded);
    if tile_lod < minimum_lod {
        sample_alpha = select(detail_alpha, 1.0, bounded);
    } else if tile_lod == minimum_lod {
        sample_alpha = select(max(sample_alpha, detail_alpha), 1.0, bounded);
    }
    let mode = u32(round(surface.debug.x));
    let far_diagnostic = mode == DEBUG_MODE_FAR_TIER;
    let base_world_position = world_position.xyz;
    let far_tier = select(
        0.0,
        far_tier_weight(base_world_position),
        mode == DEBUG_MODE_BEAUTY || far_diagnostic,
    );
    var displacement: vec3<f32>;
    var normal_cross: vec3<f32>;
    if far_tier >= 1.0 {
        displacement = far_displacement(transitioned.xy);
        normal_cross = far_normal_cross(transitioned.xy);
    } else {
        let near_displacement = sample_displacement(
            lod_data,
            lod_sampler,
            cascade_layout,
            advected_world(transitioned.xy),
            lod,
            sample_alpha,
        );
        displacement = near_displacement;
        if surface.reflection.x > 0.5 {
            normal_cross = sample_fft_normal_cross(transitioned.xy, lod, sample_alpha);
        } else {
            let texel_width = cascade_layout.cascades[lod].texel_width;
            let displacement_x = sample_displacement(
                lod_data,
                lod_sampler,
                cascade_layout,
                advected_world(transitioned.xy + vec2(texel_width, 0.0)),
                lod,
                sample_alpha,
            );
            let displacement_z = sample_displacement(
                lod_data,
                lod_sampler,
                cascade_layout,
                advected_world(transitioned.xy + vec2(0.0, texel_width)),
                lod,
                sample_alpha,
            );
            let tangent_x = vec3(texel_width, 0.0, 0.0)
                + displacement_x - near_displacement;
            let tangent_z = vec3(0.0, 0.0, texel_width)
                + displacement_z - near_displacement;
            normal_cross = cross(tangent_z, tangent_x);
        }
        normal_cross.y = max(normal_cross.y, MIN_NORMAL_Y);
        if far_tier > 0.0 {
            displacement = mix(
                near_displacement,
                far_displacement(transitioned.xy),
                far_tier,
            );
            normal_cross = deformation_safe_normalize(
                mix(
                    deformation_safe_normalize(normal_cross, vec3(0.0, 1.0, 0.0)),
                    far_normal_cross(transitioned.xy),
                    far_tier,
                ),
                vec3(0.0, 1.0, 0.0),
            );
        }
    }
    normal_cross.y = max(normal_cross.y, MIN_NORMAL_Y);
    world_position = vec4(base_world_position + displacement, world_position.w);

    result.world_position = world_position;
    result.base_world_position = base_world_position;
    result.world_normal = deformation_safe_normalize(normal_cross, vec3(0.0, 1.0, 0.0));
    result.sample_xz = transitioned.xy;
    result.undisplaced_xz = transitioned.xy;
    result.lod = lod;
    result.sample_alpha = sample_alpha;
    result.displacement = displacement;
    result.wave_height = displacement.y;
    return result;
}
