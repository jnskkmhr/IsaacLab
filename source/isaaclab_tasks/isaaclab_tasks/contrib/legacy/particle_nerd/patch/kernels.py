# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Recycle a fixed particle pool and reset neural history at entering boundaries."""

import warp as wp

from isaaclab_tasks.contrib.g1_29dof_mpm_moving_patch.util.kernel import surface_height


@wp.kernel
def update_particles(
    particle_env_ids: wp.array[int],
    patch_centers: wp.array[wp.vec2],
    particle_template_positions: wp.array[wp.vec3],
    env_reset_requested: wp.array[int],
    particle_dynamic_mass: wp.array[float],
    simulated_terrain_size: wp.vec2,
    total_patch_size: wp.vec2,
    background_mesh: wp.uint64,
    ray_z: float,
    ray_length: float,
    particle_reset_positions: wp.array[wp.vec3],
    particle_is_dynamic: wp.array[int],
    particle_q: wp.array[wp.vec3],
    particle_qd: wp.array[wp.vec3],
    particle_mass: wp.array[float],
    particle_inv_mass: wp.array[float],
    history: wp.array[wp.mat33],
    terrain_query_miss_count: wp.array[int],
):
    """Recycle particles into the patch, reset neural history, and assign zero mass to its boundary.

    ``particle_template_positions`` holds fixed world XY coordinates [m] and particle-layer Z offsets [m].
    ``particle_reset_positions`` holds terrain-adjusted world XYZ positions [m]. The environment-indexed
    ``env_reset_requested`` flags request particle reinitialization, rather than recording completed resets.
    The missed-query counter is shared across all particles in this launch.
    Read-only inputs precede the output arrays, which are updated in place and may also be read.
    """
    particle_id = wp.tid()
    env_id = particle_env_ids[particle_id]
    patch_center = patch_centers[env_id]
    particle_template_position = particle_template_positions[particle_id]
    # Toroidal wrapping retains overlap and restores fresh material at the entering edge.
    x = (
        particle_template_position[0]
        + wp.floor((patch_center[0] + 0.5 * total_patch_size[0] - particle_template_position[0]) / total_patch_size[0])
        * total_patch_size[0]
    )
    y = (
        particle_template_position[1]
        + wp.floor((patch_center[1] + 0.5 * total_patch_size[1] - particle_template_position[1]) / total_patch_size[1])
        * total_patch_size[1]
    )
    # Float32 wrapping can round past the patch edge and miss the terrain at its total_patch_size boundary.
    x = wp.clamp(x, patch_center[0] - 0.5 * total_patch_size[0], patch_center[0] + 0.5 * total_patch_size[0])
    y = wp.clamp(y, patch_center[1] - 0.5 * total_patch_size[1], patch_center[1] + 0.5 * total_patch_size[1])
    particle_reset_position = particle_reset_positions[particle_id]
    if (
        wp.abs(particle_reset_position[0] - x) > 1.0e-5
        or wp.abs(particle_reset_position[1] - y) > 1.0e-5
        or env_reset_requested[env_id] != 0
    ):
        hit, height = surface_height(background_mesh, x, y, ray_z, ray_length)
        if not hit:
            wp.atomic_add(terrain_query_miss_count, 0, 1)
        particle_reset_position = wp.vec3(x, y, height + particle_template_position[2])
    is_dynamic = int(
        wp.abs(x - patch_center[0]) < 0.5 * simulated_terrain_size[0]
        and wp.abs(y - patch_center[1]) < 0.5 * simulated_terrain_size[1]
    )
    reset_position_changed = wp.length(particle_reset_position - particle_reset_positions[particle_id]) > 1.0e-5
    escaped_patch = (
        wp.abs(particle_q[particle_id][0] - patch_center[0]) >= 0.5 * total_patch_size[0]
        or wp.abs(particle_q[particle_id][1] - patch_center[1]) >= 0.5 * total_patch_size[1]
    )
    if (
        reset_position_changed
        or escaped_patch
        or is_dynamic != particle_is_dynamic[particle_id]
        or is_dynamic == 0
        or env_reset_requested[env_id] != 0
    ):
        particle_q[particle_id] = particle_reset_position
        particle_qd[particle_id] = wp.vec3(0.0)
        history[particle_id] = wp.matrix_from_rows(
            particle_reset_position, particle_reset_position, particle_reset_position
        )
    target_particle_mass = float(is_dynamic) * particle_dynamic_mass[particle_id]
    particle_mass[particle_id] = target_particle_mass
    particle_inv_mass[particle_id] = 0.0
    if is_dynamic != 0:
        particle_inv_mass[particle_id] = 1.0 / particle_dynamic_mass[particle_id]
    particle_reset_positions[particle_id] = particle_reset_position
    particle_is_dynamic[particle_id] = is_dynamic
