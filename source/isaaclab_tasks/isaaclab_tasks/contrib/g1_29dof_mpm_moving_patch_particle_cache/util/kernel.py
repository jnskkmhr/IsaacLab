# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Warp device functions and kernels for moving terrain height queries and particles."""

import warp as wp


@wp.func
def surface_height(mesh: wp.uint64, x: float, y: float, ray_z: float, ray_length: float):
    """Cast downward at world XY [m] and return whether the mesh was hit and its height [m]."""
    query = wp.mesh_query_ray(mesh, wp.vec3(x, y, ray_z), wp.vec3(0.0, 0.0, -1.0), ray_length)
    height = ray_z - query.t
    return query.result, height


@wp.kernel
def sample_heights(mesh: wp.uint64, ray_z: float, ray_length: float, points: wp.array[wp.vec3], missed: wp.array[int]):
    """Replace query-point Z coordinates with terrain heights [m] and count missed rays."""
    i = wp.tid()
    p = points[i]
    hit, height = surface_height(mesh, p[0], p[1], ray_z, ray_length)
    if hit:
        points[i] = wp.vec3(p[0], p[1], height)
    else:
        wp.atomic_add(missed, 0, 1)


@wp.kernel
def mark_resets(selected: wp.array[wp.bool], pending: wp.array[int]):
    """Mark selected simulation worlds for particle reinitialization."""
    i = wp.tid()
    if selected[i]:
        pending[i] = 1


@wp.kernel
def update_patch_center(
    body_q: wp.array[wp.transform],
    body_indices: wp.array2d[int],
    env_origin: wp.array[wp.vec3],
    background_terrain_center: wp.vec2,
    background_terrain_size: wp.vec2,
    total_patch_size: wp.vec2,
    patch_discretization_step: float,
    patch_center: wp.array2d[wp.vec2],
):
    """Quantize robot-centered patch positions [m] and keep each full patch inside the background."""
    i, j = wp.tid()
    if body_indices[i, j] < 0:
        patch_center[i, j] = wp.vec2(1.0e20)
        return
    p = wp.transform_get_translation(body_q[body_indices[i, j]])
    o = env_origin[i]
    allowed_patch_center_half = (background_terrain_size - total_patch_size) * 0.5
    x = wp.floor((p[0] - o[0]) / patch_discretization_step + 0.5) * patch_discretization_step
    y = wp.floor((p[1] - o[1]) / patch_discretization_step + 0.5) * patch_discretization_step
    patch_center[i, j] = wp.vec2(
        wp.clamp(
            o[0] + x,
            background_terrain_center[0] - allowed_patch_center_half[0],
            background_terrain_center[0] + allowed_patch_center_half[0],
        ),
        wp.clamp(
            o[1] + y,
            background_terrain_center[1] - allowed_patch_center_half[1],
            background_terrain_center[1] + allowed_patch_center_half[1],
        ),
    )


@wp.kernel
def restore_boundary_particles(
    particle_is_dynamic: wp.array[int],
    particle_reset_positions: wp.array[wp.vec3],
    particle_q: wp.array[wp.vec3],
    particle_qd: wp.array[wp.vec3],
):
    """Restore boundary particles to their reset positions [m] and set velocities [m/s] to zero."""
    particle_id = wp.tid()
    if particle_is_dynamic[particle_id] == 0:
        particle_q[particle_id] = particle_reset_positions[particle_id]
        particle_qd[particle_id] = wp.vec3(0.0)


@wp.kernel
def mark_visible_particles(
    particle_env_ids: wp.array[int],
    particle_flags: wp.array[int],
    particle_is_dynamic: wp.array[int],
    env_visibility_mask: wp.array[int],
    filter_envs: bool,
    show_boundary_particles: bool,
    particle_visibility_mask: wp.array[int],
):
    """Select particles by environment and simulated/boundary membership without changing physics flags."""
    particle_id = wp.tid()
    is_visible = (particle_flags[particle_id] & 1) != 0 and (
        show_boundary_particles or particle_is_dynamic[particle_id] != 0
    )
    if filter_envs:
        env_id = particle_env_ids[particle_id]
        if env_id >= 0 and env_id < env_visibility_mask.shape[0]:
            is_visible = is_visible and env_visibility_mask[env_id] != 0
        else:
            is_visible = False
    particle_visibility_mask[particle_id] = wp.where(is_visible, 1, 0)


@wp.kernel
def gather_visible_particles(
    particle_visibility_mask: wp.array[int],
    particle_visible_offsets: wp.array[int],
    particle_q: wp.array[wp.vec3],
    particle_radius: wp.array[float],
    visible_particle_q: wp.array[wp.vec3],
    visible_particle_radius: wp.array[float],
):
    """Compact visible particle positions [m] and radii [m] using inclusive prefix-scan offsets."""
    particle_id = wp.tid()
    if particle_visibility_mask[particle_id] != 0:
        visible_particle_id = particle_visible_offsets[particle_id] - 1
        visible_particle_q[visible_particle_id] = particle_q[particle_id]
        visible_particle_radius[visible_particle_id] = particle_radius[particle_id]


@wp.struct
class ParticleState:
    q: wp.array[wp.vec3]
    qd: wp.array[wp.vec3]
    grad: wp.array[wp.mat33]
    elastic: wp.array[wp.mat33]
    plastic: wp.array[float]
    stress: wp.array[wp.mat33]
    transform: wp.array[wp.mat33]
    mass: wp.array[float]
    radius: wp.array[float]
    flags: wp.array[int]


@wp.func
def copy_particle(source: ParticleState, i: int, target: ParticleState, j: int):
    target.q[j] = source.q[i]
    target.qd[j] = source.qd[i]
    target.grad[j] = source.grad[i]
    target.elastic[j] = source.elastic[i]
    target.plastic[j] = source.plastic[i]
    target.stress[j] = source.stress[i]
    target.transform[j] = source.transform[i]
    target.mass[j] = source.mass[i]
    target.radius[j] = source.radius[i]
    target.flags[j] = source.flags[i]


@wp.kernel
def plan_transitions(
    live: ParticleState,
    frozen: ParticleState,
    worlds: wp.array[int],
    centers: wp.array2d[wp.vec2],
    reset: wp.array[int],
    size: wp.vec2,
    tile_begins: wp.array2d[wp.vec2i],
    tile_shape: wp.vec2i,
    tile_size: float,
    layers: wp.array[int],
    next_layers: wp.array[int],
    outgoing_ids: wp.array[int],
    outgoing_tiles: wp.array[wp.vec2i],
    counts: wp.array[int],
):
    """Classify particles without mutating physics; collect departing slots for capacity checks."""
    i = wp.tid()
    world = worlds[i]
    layer = layers[i]
    new_layer = int(0)
    if layer != 0 and reset[world] == 0:
        p = live.q[i]
        if layer == 2:
            p = frozen.q[i]
        tile = wp.vec2i(int(wp.floor(p[0] / tile_size)), int(wp.floor(p[1] / tile_size)))
        for region in range(centers.shape[1]):
            delta = wp.vec2(p[0], p[1]) - centers[world, region]
            begin = tile_begins[world, region]
            if (
                tile[0] >= begin[0]
                and tile[0] < begin[0] + tile_shape[0]
                and tile[1] >= begin[1]
                and tile[1] < begin[1] + tile_shape[1]
            ):
                if new_layer == 0:
                    new_layer = 2
                if wp.abs(delta[0]) < size[0] * 0.5 and wp.abs(delta[1]) < size[1] * 0.5:
                    new_layer = 1
        if new_layer == 0:
            index = wp.atomic_add(counts, 0, 1)
            outgoing_ids[index] = i
            outgoing_tiles[index] = tile
    next_layers[i] = new_layer
    if layer != new_layer:
        wp.atomic_add(counts, 1, 1)


@wp.kernel
def save_particles(
    live: ParticleState,
    frozen: ParticleState,
    layers: wp.array[int],
    source_ids: wp.array[int],
    saved: ParticleState,
    target_ids: wp.array[int],
):
    """Save dynamic state or the original frozen state, never the zero-mass boundary state."""
    k = wp.tid()
    i, j = source_ids[k], target_ids[k]
    if layers[i] == 2:
        copy_particle(frozen, i, saved, j)
    else:
        copy_particle(live, i, saved, j)


@wp.kernel
def apply_transitions(
    live: ParticleState,
    frozen: ParticleState,
    inactive_position: wp.vec3,
    layers: wp.array[int],
    next_layers: wp.array[int],
    dynamic: wp.array[int],
    anchors: wp.array[wp.vec3],
    inv_mass: wp.array[float],
):
    """Freeze the original state before imposing kinematics; thaw from that original state."""
    i = wp.tid()
    old, new = layers[i], next_layers[i]
    if old == 2:
        copy_particle(frozen, i, live, i)
    if old == 1 and new == 2:
        copy_particle(live, i, frozen, i)
    if new == 0:
        # PicQuadrature bins positions before integrands inspect particle flags.
        # Coincident inactive slots inside the grid create expensive cell loops.
        live.q[i] = inactive_position
        live.flags[i] = 0
        live.mass[i] = 0.0
        live.qd[i] = wp.vec3(0.0)
        live.grad[i] = wp.mat33(0.0)
    elif new == 2:
        live.mass[i] = 0.0
        live.qd[i] = wp.vec3(0.0)
        live.grad[i] = wp.mat33(0.0)
    anchors[i] = live.q[i]
    layers[i] = new
    dynamic[i] = int(new == 1)
    inv_mass[i] = 0.0
    if new == 1:
        inv_mass[i] = 1.0 / live.mass[i]


@wp.func
def finish_loading(
    live: ParticleState,
    frozen: ParticleState,
    i: int,
    centers: wp.array2d[wp.vec2],
    world: int,
    size: wp.vec2,
    layers: wp.array[int],
    dynamic: wp.array[int],
    anchors: wp.array[wp.vec3],
    inv_mass: wp.array[float],
):
    copy_particle(live, i, frozen, i)
    p = live.q[i]
    is_dynamic = bool(False)
    for region in range(centers.shape[1]):
        center = centers[world, region]
        if wp.abs(p[0] - center[0]) < 0.5 * size[0] and wp.abs(p[1] - center[1]) < 0.5 * size[1]:
            is_dynamic = True
    dynamic[i] = int(is_dynamic)
    layers[i] = wp.where(is_dynamic, 1, 2)
    anchors[i] = p
    inv_mass[i] = 0.0
    if is_dynamic:
        inv_mass[i] = 1.0 / live.mass[i]
    else:
        live.mass[i] = 0.0
        live.qd[i] = wp.vec3(0.0)
        live.grad[i] = wp.mat33(0.0)


@wp.kernel
def load_particles(
    saved: ParticleState,
    source_ids: wp.array[int],
    live: ParticleState,
    frozen: ParticleState,
    slots: wp.array[int],
    worlds: wp.array[int],
    centers: wp.array2d[wp.vec2],
    size: wp.vec2,
    layers: wp.array[int],
    dynamic: wp.array[int],
    anchors: wp.array[wp.vec3],
    inv_mass: wp.array[float],
):
    """Restore saved physical fields into fixed simulation slots belonging to the same environment."""
    k = wp.tid()
    i = slots[k]
    copy_particle(saved, source_ids[k], live, i)
    finish_loading(live, frozen, i, centers, worlds[i], size, layers, dynamic, anchors, inv_mass)


@wp.kernel
def generate_particles(
    slots: wp.array[int],
    tiles: wp.array[wp.vec2i],
    template: wp.array[wp.vec3],
    template_elastic: wp.array[wp.mat33],
    template_transform: wp.array[wp.mat33],
    template_stress: wp.array[wp.mat33],
    template_plastic: wp.array[float],
    template_count: int,
    tile_size: float,
    mesh: wp.uint64,
    ray_z: float,
    ray_length: float,
    masses: wp.array[float],
    radius: float,
    live: ParticleState,
    frozen: ParticleState,
    worlds: wp.array[int],
    centers: wp.array2d[wp.vec2],
    size: wp.vec2,
    layers: wp.array[int],
    dynamic: wp.array[int],
    anchors: wp.array[wp.vec3],
    inv_mass: wp.array[float],
    missed: wp.array[int],
):
    """Generate only never-visited terrain from the shared, undeformed background mesh."""
    k = wp.tid()
    i = slots[k]
    tile = tiles[k // template_count]
    entry = worlds[i] * template_count + k % template_count
    p = template[entry]
    # Alternating mirror tiles retain matching wall states (Felix's mirror_tensor construction).
    dx = wp.where(tile[0] % 2 == 0, 1.0, -1.0)
    dy = wp.where(tile[1] % 2 == 0, 1.0, -1.0)
    mirror = wp.diag(wp.vec3(dx, dy, 1.0))
    p = wp.vec3(wp.where(dx < 0.0, tile_size - p[0], p[0]), wp.where(dy < 0.0, tile_size - p[1], p[1]), p[2])
    x = float(tile[0]) * tile_size + p[0]
    y = float(tile[1]) * tile_size + p[1]
    hit, height = surface_height(mesh, x, y, ray_z, ray_length)
    if not hit:
        wp.atomic_add(missed, 0, 1)
    live.q[i] = wp.vec3(x, y, height + p[2])
    live.qd[i] = wp.vec3(0.0)
    live.grad[i] = wp.mat33(0.0)
    live.elastic[i] = mirror * template_elastic[entry] * mirror
    live.plastic[i] = template_plastic[entry]
    live.stress[i] = mirror * template_stress[entry] * mirror
    live.transform[i] = mirror * template_transform[entry] * mirror
    live.mass[i] = masses[i]
    live.radius[i] = radius
    live.flags[i] = 1
    finish_loading(live, frozen, i, centers, worlds[i], size, layers, dynamic, anchors, inv_mass)


@wp.kernel
def refresh_cached_properties(
    mass: wp.array[float],
    radius: wp.array[float],
    flags: wp.array[int],
    volume: wp.array[float],
    density: wp.array[float],
    material_volume: wp.array[float],
    transfer_flags: wp.array[int],
    material_flags: wp.array[int],
):
    """Refresh ordinary material particles without downloading particle arrays or changing their history."""
    i = wp.tid()
    v = 8.0 * radius[i] * radius[i] * radius[i]
    volume[i] = v
    density[i] = mass[i] / v
    transfer_flags[i] = flags[i]
    material_flags[i] = flags[i]
    material_volume[i] = wp.where((flags[i] & 1) != 0, v, 0.0)
