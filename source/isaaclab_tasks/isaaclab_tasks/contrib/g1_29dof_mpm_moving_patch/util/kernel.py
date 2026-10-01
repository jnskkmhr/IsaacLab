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
def update_centers(
    body_q: wp.array[wp.transform],
    bodies: wp.array[int],
    origins: wp.array[wp.vec3],
    centers: wp.array[wp.vec2],
    background_center: wp.vec2,
    background: wp.vec2,
    outer: wp.vec2,
    shift: float,
):
    """Quantize robot-centered patch positions [m] and keep each full patch inside the background."""
    i = wp.tid()
    p = wp.transform_get_translation(body_q[bodies[i]])
    o = origins[i]
    half = (background - outer) * 0.5
    x = wp.floor((p[0] - o[0]) / shift + 0.5) * shift
    y = wp.floor((p[1] - o[1]) / shift + 0.5) * shift
    centers[i] = wp.vec2(
        wp.clamp(o[0] + x, background_center[0] - half[0], background_center[0] + half[0]),
        wp.clamp(o[1] + y, background_center[1] - half[1], background_center[1] + half[1]),
    )


@wp.kernel
def update_particles(
    worlds: wp.array[int],
    centers: wp.array[wp.vec2],
    reference: wp.array[wp.vec3],
    anchors: wp.array[wp.vec3],
    dynamic: wp.array[int],
    reset: wp.array[int],
    moving: wp.vec2,
    outer: wp.vec2,
    q: wp.array[wp.vec3],
    qd: wp.array[wp.vec3],
    mass: wp.array[float],
    inv_mass: wp.array[float],
    dynamic_mass: wp.array[float],
    changed: wp.array[int],
    grad: wp.array[wp.mat33],
    elastic: wp.array[wp.mat33],
    plastic: wp.array[float],
    initial_plastic: wp.array[float],
    stress: wp.array[wp.mat33],
    transform: wp.array[wp.mat33],
    surface_mesh: wp.uint64,
    ray_z: float,
    ray_length: float,
):
    """Recycle particles into the patch, restore material state, and assign zero mass to its boundary."""
    i = wp.tid()
    world = worlds[i]
    center = centers[world]
    ref = reference[i]
    # Toroidal wrapping retains overlap and restores fresh material at the entering edge.
    x = ref[0] + wp.floor((center[0] + 0.5 * outer[0] - ref[0]) / outer[0]) * outer[0]
    y = ref[1] + wp.floor((center[1] + 0.5 * outer[1] - ref[1]) / outer[1]) * outer[1]
    # Float32 wrapping can round past the patch edge and miss the terrain at its outer boundary.
    x = wp.clamp(x, center[0] - 0.5 * outer[0], center[0] + 0.5 * outer[0])
    y = wp.clamp(y, center[1] - 0.5 * outer[1], center[1] + 0.5 * outer[1])
    anchor = anchors[i]
    if wp.abs(anchor[0] - x) > 1.0e-5 or wp.abs(anchor[1] - y) > 1.0e-5 or reset[world] != 0:
        hit, height = surface_height(surface_mesh, x, y, ray_z, ray_length)
        if not hit:
            wp.atomic_add(changed, 1, 1)
        anchor = wp.vec3(x, y, height + ref[2])
    active = int(wp.abs(x - center[0]) < 0.5 * moving[0] and wp.abs(y - center[1]) < 0.5 * moving[1])
    moved = wp.length(anchor - anchors[i]) > 1.0e-5
    escaped = wp.abs(q[i][0] - center[0]) >= 0.5 * outer[0] or wp.abs(q[i][1] - center[1]) >= 0.5 * outer[1]
    if moved or escaped or active != dynamic[i] or active == 0 or reset[world] != 0:
        q[i] = anchor
        qd[i] = wp.vec3(0.0)
        grad[i] = wp.mat33(0.0)
        elastic[i] = wp.identity(n=3, dtype=float)
        plastic[i] = initial_plastic[i]
        stress[i] = wp.mat33(0.0)
        transform[i] = wp.identity(n=3, dtype=float)
    target_mass = float(active) * dynamic_mass[i]
    if mass[i] != target_mass:
        wp.atomic_add(changed, 0, 1)
    mass[i] = target_mass
    inv_mass[i] = 0.0
    if active != 0:
        inv_mass[i] = 1.0 / dynamic_mass[i]
    anchors[i] = anchor
    dynamic[i] = active


@wp.kernel
def restore_boundary_particles(
    dynamic: wp.array[int], anchors: wp.array[wp.vec3], q: wp.array[wp.vec3], qd: wp.array[wp.vec3]
):
    """Restore boundary particle positions [m] to their anchors and set velocities [m/s] to zero."""
    i = wp.tid()
    if dynamic[i] == 0:
        q[i] = anchors[i]
        qd[i] = wp.vec3(0.0)


@wp.kernel
def refresh_density(mass: wp.array[float], volume: wp.array[float], density: wp.array[float]):
    """Update particle densities [kg/m^3] from masses [kg] and fixed reference volumes [m^3]."""
    i = wp.tid()
    density[i] = mass[i] / volume[i]


@wp.kernel
def mark_visible_particles(
    worlds: wp.array[int],
    dynamic: wp.array[int],
    visible_worlds: wp.array[int],
    filter_worlds: bool,
    show_boundary: bool,
    mask: wp.array[int],
):
    """Select particles by world and simulated/boundary membership without changing physics flags."""
    i = wp.tid()
    visible = show_boundary or dynamic[i] != 0
    if filter_worlds:
        world = worlds[i]
        if world >= 0 and world < visible_worlds.shape[0]:
            visible = visible and visible_worlds[world] != 0
        else:
            visible = False
    mask[i] = wp.where(visible, 1, 0)


@wp.kernel
def gather_visible_particles(
    mask: wp.array[int],
    offsets: wp.array[int],
    positions: wp.array[wp.vec3],
    radii: wp.array[float],
    visible_positions: wp.array[wp.vec3],
    visible_radii: wp.array[float],
):
    """Compact visible positions [m] and radii [m] using inclusive prefix-scan offsets."""
    i = wp.tid()
    if mask[i] != 0:
        j = offsets[i] - 1
        visible_positions[j] = positions[i]
        visible_radii[j] = radii[i]
