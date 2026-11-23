# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Every Warp kernel and Warp function of Particle NeRD.

The file holds two Warp modules, because they are compiled with different options:

- the default module of this file (network inputs, active-particle bookkeeping and the rigid coupling); the two
  single-environment feature kernels are differentiated during training, so the backward pass stays enabled;
- :data:`CONTACT_MODULE` (the contact impulse solver, in double precision), compiled without a backward pass and
  without loop unrolling.

Collider kinds: 1 box (half extents), 2 sphere (radius in x), 4 cylinder along z (radius in x, half height in z),
5 capsule along z (radius in x, half height in y); any other kind is the plane z = 0.

The file ends with :func:`warp_stream`, the one helper that every caller needs to launch these kernels in step
with torch.
"""

import torch
import warp as wp

wp.set_module_options({"enable_backward": True})

CONTACT_MODULE = wp.get_module(__name__ + "_contact")
"""Warp module of the contact impulse solver."""
wp.set_module_options({"enable_backward": False, "max_unroll": 0}, module=CONTACT_MODULE)

vec6d = wp.types.vector(length=6, dtype=wp.float64)
mat66d = wp.types.matrix(shape=(6, 6), dtype=wp.float64)
mat36d = wp.types.matrix(shape=(3, 6), dtype=wp.float64)
mat63d = wp.types.matrix(shape=(6, 3), dtype=wp.float64)


# ---------------------------------------------------------------------------------------------------------------------
# Network inputs: collider geometry, particle features, neighbourhood bases
# ---------------------------------------------------------------------------------------------------------------------


@wp.func
def primitive_surface(local: wp.vec3, kind: int, size: wp.vec3):
    """Smooth signed distance and outward normal of a primitive collider, in the collider frame."""
    if kind == 1:
        q = wp.vec3(wp.abs(local[0]) - size[0], wp.abs(local[1]) - size[1], wp.abs(local[2]) - size[2])
        outside = wp.vec3(wp.max(q[0], 0.0), wp.max(q[1], 0.0), wp.max(q[2], 0.0))
        out_sq = wp.dot(outside, outside)
        length = wp.sqrt(out_sq + 1.0e-8)
        distance = (length - 1.0e-4) + wp.min(wp.max(q[0], wp.max(q[1], q[2])), 0.0)
        axis = int(0)
        if q[1] > q[axis]:
            axis = 1
        if q[2] > q[axis]:
            axis = 2
        normal = wp.vec3(0.0)
        for a in range(3):
            sign = float(1.0)
            if local[a] < 0.0:
                sign = -1.0
            if out_sq > 1.0e-8:
                normal[a] = sign * outside[a] / length
            elif a == axis:
                normal[a] = sign
        return distance, normal
    if kind == 2:
        length = wp.sqrt(wp.dot(local, local) + 1.0e-8)
        distance = length - size[0]
        normal = wp.vec3(0.0, 0.0, 1.0)
        if length > 1.0e-4:
            normal = local / length
        return distance, normal
    if kind == 5:
        z_cap = wp.clamp(local[2], -size[1], size[1])
        diff = wp.vec3(local[0], local[1], local[2] - z_cap)
        length = wp.sqrt(wp.dot(diff, diff) + 1.0e-8)
        distance = length - size[0]
        normal = wp.vec3(0.0, 0.0, 1.0)
        if length > 1.0e-4:
            normal = diff / length
        return distance, normal
    if kind == 4:
        radial = wp.sqrt(local[0] * local[0] + local[1] * local[1] + 1.0e-8)
        q2 = wp.vec2(radial - size[0], wp.abs(local[2]) - size[2])
        outside2 = wp.vec2(wp.max(q2[0], 0.0), wp.max(q2[1], 0.0))
        out_sq2 = wp.dot(outside2, outside2)
        length2 = wp.sqrt(out_sq2 + 1.0e-8)
        distance = (length2 - 1.0e-4) + wp.min(wp.max(q2[0], q2[1]), 0.0)
        direction = wp.vec2(1.0, 0.0)
        if radial > 1.0e-4:
            direction = wp.vec2(local[0], local[1]) / radial
        weights = wp.vec2(1.0, 0.0)
        if out_sq2 > 1.0e-8:
            weights = outside2 / length2
        elif q2[1] > q2[0]:
            weights = wp.vec2(0.0, 1.0)
        sign = float(1.0)
        if local[2] < 0.0:
            sign = -1.0
        return distance, wp.vec3(direction[0] * weights[0], direction[1] * weights[0], sign * weights[1])
    return local[2], wp.vec3(0.0, 0.0, 1.0)


@wp.func
def ball_to_cube_radial(d: wp.vec3, inv_radius: float) -> wp.vec3:
    """Volume-preserving ball-to-cube mapping in [-1, 1]^3."""
    s = d * inv_radius
    r2 = wp.dot(s, s)
    if r2 <= 1.0e-12:
        return wp.vec3(0.0, 0.0, 0.0)
    r = wp.sqrt(r2)
    max_c = wp.max(wp.abs(s[0]), wp.max(wp.abs(s[1]), wp.abs(s[2])))
    if max_c <= 1.0e-8:
        return wp.vec3(0.0, 0.0, 0.0)
    return s * (r / max_c)


@wp.func
def poly6_window(r_norm_sq: float) -> float:
    """Cubic poly6 window (1 - (r/R)^2)^3 for r <= R."""
    if r_norm_sq >= 1.0:
        return 0.0
    t = 1.0 - r_norm_sq
    return t * t * t


@wp.kernel
def neighbor_basis_and_packing(
    grid: wp.uint64,
    pos: wp.array(dtype=wp.vec3),
    vel: wp.array(dtype=wp.vec3),
    radius: float,
    max_neighbors: int,
    particles_per_env: int,
    basis_out: wp.array3d(dtype=float),
    packing_out: wp.array2d(dtype=float),
):
    """Continuous-convolution basis [N, 64, 4] over neighbours plus the packing state [N, 6].

    The packing state is `[count, phi, dphi_dt, v_rel_x, v_rel_y, v_rel_z]`. `dphi_dt` is the local compaction rate
    `-sum_j grad(w_ij) . (v_j - v_i)`: positive while the neighbourhood compresses, negative while it loosens.
    """
    i = wp.tid()
    x_i = pos[i]
    v_i = vel[i]
    inv_r = 1.0 / radius
    inv_r2 = inv_r * inv_r
    r_sq = radius * radius

    group = (i // particles_per_env) if particles_per_env > 0 else -1
    query = wp.hash_grid_query(grid, x_i, radius, group)
    neighbor = int(0)
    count = int(0)
    w_sum = float(0.0)
    dphi_dt = float(0.0)
    v_rel_sum = wp.vec3(0.0, 0.0, 0.0)

    while wp.hash_grid_query_next(query, neighbor):
        if neighbor == i:
            continue
        d = pos[neighbor] - x_i
        dist_sq = wp.dot(d, d)
        if dist_sq < r_sq:
            r_norm_sq = dist_sq * inv_r2
            t = 1.0 - r_norm_sq
            w = t * t * t
            dv = vel[neighbor] - v_i
            # Time derivative of poly6 weight w(r_ij(t)): dw/dt = -6 * (1 - r^2/R^2)^2 / R^2 * (d . dv)
            dw_dt = -6.0 * (t * t) * inv_r2 * wp.dot(d, dv)
            w_sum += w
            dphi_dt += dw_dt
            v_rel_sum += w * dv

            u = ball_to_cube_radial(d, inv_r)
            gx = wp.clamp(1.5 * (u[0] + 1.0), 0.0, 2.9999)
            gy = wp.clamp(1.5 * (u[1] + 1.0), 0.0, 2.9999)
            gz = wp.clamp(1.5 * (u[2] + 1.0), 0.0, 2.9999)
            ix = int(gx)
            iy = int(gy)
            iz = int(gz)
            fx = gx - float(ix)
            fy = gy - float(iy)
            fz = gz - float(iz)

            for dx in range(2):
                wx = (1.0 - fx) if dx == 0 else fx
                for dy in range(2):
                    wy = (1.0 - fy) if dy == 0 else fy
                    for dz in range(2):
                        wz = (1.0 - fz) if dz == 0 else fz
                        wt = w * wx * wy * wz
                        cell = (ix + dx) * 16 + (iy + dy) * 4 + (iz + dz)
                        basis_out[i, cell, 0] += wt
                        basis_out[i, cell, 1] += wt * dv[0]
                        basis_out[i, cell, 2] += wt * dv[1]
                        basis_out[i, cell, 3] += wt * dv[2]
            count += 1
            if count >= max_neighbors:
                break

    scale = 1.0 / 16.0
    for cell in range(64):
        for c in range(4):
            basis_out[i, cell, c] = basis_out[i, cell, c] * scale

    packing_out[i, 0] = float(count) / 32.0
    packing_out[i, 1] = w_sum / 16.0
    packing_out[i, 2] = wp.clamp(dphi_dt * 0.05, -5.0, 5.0)
    inv_w = 1.0 / wp.max(w_sum, 1.0e-4)
    packing_out[i, 3] = v_rel_sum[0] * inv_w
    packing_out[i, 4] = v_rel_sum[1] * inv_w
    packing_out[i, 5] = v_rel_sum[2] * inv_w


@wp.kernel
def collider_basis(
    pairs: wp.array3d(dtype=float),
    influence: wp.array2d(dtype=float),
    obs_basis_out: wp.array3d(dtype=float),
):
    """Continuous-convolution basis [N, 64, 9] over the collider surface points that influence a particle."""
    i = wp.tid()
    num_colliders = pairs.shape[1]

    for j in range(num_colliders):
        inf = influence[i, j]
        if inf > 0.0:
            gap_norm = pairs[i, j, 0]
            nx = pairs[i, j, 1]
            ny = pairs[i, j, 2]
            nz = pairs[i, j, 3]
            vx = pairs[i, j, 4]
            vy = pairs[i, j, 5]
            vz = pairs[i, j, 6]
            mu = pairs[i, j, 7]

            r_norm = 1.0 - inf
            w = poly6_window(r_norm * r_norm)
            d = wp.vec3(-gap_norm * nx, -gap_norm * ny, -gap_norm * nz)
            u = ball_to_cube_radial(d, 1.0)
            gx = wp.clamp(1.5 * (u[0] + 1.0), 0.0, 2.9999)
            gy = wp.clamp(1.5 * (u[1] + 1.0), 0.0, 2.9999)
            gz = wp.clamp(1.5 * (u[2] + 1.0), 0.0, 2.9999)
            ix = int(gx)
            iy = int(gy)
            iz = int(gz)
            fx = gx - float(ix)
            fy = gy - float(iy)
            fz = gz - float(iz)

            for dx in range(2):
                wx = (1.0 - fx) if dx == 0 else fx
                for dy in range(2):
                    wy = (1.0 - fy) if dy == 0 else fy
                    for dz in range(2):
                        wz = (1.0 - fz) if dz == 0 else fz
                        wt = w * wx * wy * wz
                        cell = (ix + dx) * 16 + (iy + dy) * 4 + (iz + dz)
                        obs_basis_out[i, cell, 0] += wt
                        obs_basis_out[i, cell, 1] += wt * gap_norm
                        obs_basis_out[i, cell, 2] += wt * nx
                        obs_basis_out[i, cell, 3] += wt * ny
                        obs_basis_out[i, cell, 4] += wt * nz
                        obs_basis_out[i, cell, 5] += wt * vx
                        obs_basis_out[i, cell, 6] += wt * vy
                        obs_basis_out[i, cell, 7] += wt * vz
                        obs_basis_out[i, cell, 8] += wt * mu


@wp.kernel
def particle_features(
    history: wp.array3d(dtype=float),
    material: wp.array(dtype=float),
    dt: float,
    gravity: wp.vec3,
    nodes: wp.array2d(dtype=float),
):
    """Particle features without global position: velocities (3 * (H - 1)), material (12), dt (1), gravity (3)."""
    i = wp.tid()
    h = history.shape[1]
    vel_dim = 3 * (h - 1)
    for t in range(h - 1):
        for a in range(3):
            nodes[i, t * 3 + a] = (history[i, t + 1, a] - history[i, t, a]) / dt
    for k in range(12):
        nodes[i, vel_dim + k] = material[k]
    nodes[i, vel_dim + 12] = dt / 0.02
    for a in range(3):
        nodes[i, vel_dim + 13 + a] = gravity[a] / 9.81


@wp.kernel
def collider_features(
    history: wp.array3d(dtype=float),
    poses: wp.array2d(dtype=float),
    velocities: wp.array2d(dtype=float),
    kinds: wp.array(dtype=int),
    sizes: wp.array2d(dtype=float),
    friction: wp.array(dtype=float),
    material: wp.array(dtype=float),
    dt: float,
    support: float,
    far_support_scale: float,
    far_gap_weight: float,
    far_influence_weight: float,
    pairs: wp.array3d(dtype=float),
    influence: wp.array2d(dtype=float),
):
    """Particle-collider pair features [N, C, 8] (gap, normal, relative surface velocity, friction) and influence."""
    i, j = wp.tid()
    h = history.shape[1]
    x = wp.vec3(history[i, h - 1, 0], history[i, h - 1, 1], history[i, h - 1, 2])
    previous = wp.vec3(history[i, h - 2, 0], history[i, h - 2, 1], history[i, h - 2, 2])
    center = wp.vec3(poses[j, 0], poses[j, 1], poses[j, 2])
    rotation = wp.quat(poses[j, 3], poses[j, 4], poses[j, 5], poses[j, 6])
    local = wp.quat_rotate_inv(rotation, x - center)
    size_vec = wp.vec3(sizes[j, 0], sizes[j, 1], sizes[j, 2])
    distance, normal_local = primitive_surface(local, kinds[j], size_vec)
    normal = wp.quat_rotate(rotation, normal_local)
    surface = x - distance * normal
    linear = wp.vec3(velocities[j, 0], velocities[j, 1], velocities[j, 2])
    angular = wp.vec3(velocities[j, 3], velocities[j, 4], velocities[j, 5])
    relative = linear + wp.cross(angular, surface - center) - (x - previous) / dt
    gap = distance - 0.02 * material[11]
    gap_s = gap / support
    inv_far = 1.0 / wp.max(far_support_scale, 1.0)
    near_gap_w = 1.0 - far_gap_weight
    pairs[i, j, 0] = near_gap_w * wp.tanh(gap_s) + far_gap_weight * wp.tanh(inv_far * gap_s)
    for a in range(3):
        pairs[i, j, 1 + a] = normal[a]
        pairs[i, j, 4 + a] = relative[a]
    pairs[i, j, 7] = friction[j]
    g_pos = wp.max(gap_s, 0.0)
    th_short = wp.tanh(g_pos)
    th_long = wp.tanh(inv_far * g_pos)
    near_inf_w = 1.0 - far_influence_weight
    influence[i, j] = near_inf_w * (1.0 - th_short * th_short) + far_influence_weight * (1.0 - th_long * th_long)


@wp.kernel
def particle_features_batched(
    history: wp.array3d(dtype=float),
    material: wp.array2d(dtype=float),
    particles_per_env: int,
    dt: float,
    gravity: wp.vec3,
    nodes: wp.array2d(dtype=float),
):
    """:func:`particle_features` for [E * M] particles with one material row per environment."""
    i = wp.tid()
    env = i // particles_per_env
    h = history.shape[1]
    vel_dim = 3 * (h - 1)
    for t in range(h - 1):
        for a in range(3):
            nodes[i, t * 3 + a] = (history[i, t + 1, a] - history[i, t, a]) / dt
    for k in range(12):
        nodes[i, vel_dim + k] = material[env, k]
    nodes[i, vel_dim + 12] = dt / 0.02
    for a in range(3):
        nodes[i, vel_dim + 13 + a] = gravity[a] / 9.81


@wp.kernel
def collider_features_batched(
    history: wp.array3d(dtype=float),
    poses: wp.array3d(dtype=float),
    velocities: wp.array3d(dtype=float),
    kinds: wp.array(dtype=int),
    sizes: wp.array3d(dtype=float),
    friction: wp.array2d(dtype=float),
    material: wp.array2d(dtype=float),
    particles_per_env: int,
    dt: float,
    support: float,
    far_support_scale: float,
    far_gap_weight: float,
    far_influence_weight: float,
    pairs: wp.array3d(dtype=float),
    influence: wp.array2d(dtype=float),
):
    """:func:`collider_features` for [E * M] particles with per-environment collider states."""
    i, j = wp.tid()
    env = i // particles_per_env
    h = history.shape[1]
    x = wp.vec3(history[i, h - 1, 0], history[i, h - 1, 1], history[i, h - 1, 2])
    previous = wp.vec3(history[i, h - 2, 0], history[i, h - 2, 1], history[i, h - 2, 2])
    center = wp.vec3(poses[env, j, 0], poses[env, j, 1], poses[env, j, 2])
    rotation = wp.quat(poses[env, j, 3], poses[env, j, 4], poses[env, j, 5], poses[env, j, 6])
    local = wp.quat_rotate_inv(rotation, x - center)
    size_vec = wp.vec3(sizes[env, j, 0], sizes[env, j, 1], sizes[env, j, 2])
    distance, normal_local = primitive_surface(local, kinds[j], size_vec)
    normal = wp.quat_rotate(rotation, normal_local)
    surface = x - distance * normal
    linear = wp.vec3(velocities[env, j, 0], velocities[env, j, 1], velocities[env, j, 2])
    angular = wp.vec3(velocities[env, j, 3], velocities[env, j, 4], velocities[env, j, 5])
    relative = linear + wp.cross(angular, surface - center) - (x - previous) / dt
    gap = distance - 0.02 * material[env, 11]
    gap_s = gap / support
    inv_far = 1.0 / wp.max(far_support_scale, 1.0)
    near_gap_w = 1.0 - far_gap_weight
    pairs[i, j, 0] = near_gap_w * wp.tanh(gap_s) + far_gap_weight * wp.tanh(inv_far * gap_s)
    for a in range(3):
        pairs[i, j, 1 + a] = normal[a]
        pairs[i, j, 4 + a] = relative[a]
    pairs[i, j, 7] = friction[env, j]
    g_pos = wp.max(gap_s, 0.0)
    th_short = wp.tanh(g_pos)
    th_long = wp.tanh(inv_far * g_pos)
    near_inf_w = 1.0 - far_influence_weight
    influence[i, j] = near_inf_w * (1.0 - th_short * th_short) + far_influence_weight * (1.0 - th_long * th_long)


# ---------------------------------------------------------------------------------------------------------------------
# Solver state: particle history, active-particle selection, rigid coupling
# ---------------------------------------------------------------------------------------------------------------------


@wp.kernel
def reset_history(
    q: wp.array[wp.vec3],
    qd: wp.array[wp.vec3],
    worlds: wp.array[int],
    mask: wp.array[wp.bool],
    all_worlds: bool,
    dt: float,
    history: wp.array[wp.mat33],
):
    """Rebuild the three-frame position history of the selected worlds from position and velocity."""
    i = wp.tid()
    world = worlds[i]
    if all_worlds or mask[wp.where(world < 0, mask.shape[0] - 1, world)]:
        p, v = q[i], qd[i]
        history[i] = wp.matrix_from_rows(p - 2.0 * dt * v, p - dt * v, p)


@wp.kernel
def active_particle_score(
    pos_in: wp.array(dtype=wp.vec3),
    history: wp.array(dtype=wp.mat33),
    dyn_poses: wp.array3d(dtype=float),
    dyn_sizes: wp.array3d(dtype=float),
    capacity: int,
    n_dyn: int,
    dt: float,
    scores: wp.array(dtype=float),
    min_gaps: wp.array(dtype=float),
):
    """Score for active-particle selection (lower is more active): gap to the nearest collider box minus speed."""
    tid = wp.tid()
    env = tid // capacity
    x = pos_in[tid]
    h = history[tid]
    x_prev = wp.vec3(h[1, 0], h[1, 1], h[1, 2])
    speed = wp.length(x - x_prev) / wp.max(dt, 1.0e-6)
    min_gap = float(1.0e6)
    for c in range(n_dyn):
        center = wp.vec3(dyn_poses[env, c, 0], dyn_poses[env, c, 1], dyn_poses[env, c, 2])
        rot = wp.quat(dyn_poses[env, c, 3], dyn_poses[env, c, 4], dyn_poses[env, c, 5], dyn_poses[env, c, 6])
        local = wp.quat_rotate_inv(rot, x - center)
        half_ext = wp.vec3(dyn_sizes[env, c, 0], dyn_sizes[env, c, 1], dyn_sizes[env, c, 2])
        qx = wp.abs(local[0]) - half_ext[0]
        qy = wp.abs(local[1]) - half_ext[1]
        qz = wp.abs(local[2]) - half_ext[2]
        out_d = wp.length(wp.vec3(wp.max(qx, 0.0), wp.max(qy, 0.0), wp.max(qz, 0.0)))
        in_d = wp.min(wp.max(qx, wp.max(qy, qz)), 0.0)
        d_c = out_d + in_d
        if d_c < min_gap:
            min_gap = d_c
    min_gaps[tid] = min_gap
    scores[tid] = min_gap - 0.12 * wp.min(speed, 2.0)


@wp.kernel
def extract_active_history(
    topk: wp.array2d(dtype=wp.int64),
    pos_in: wp.array(dtype=wp.vec3),
    history: wp.array(dtype=wp.mat33),
    capacity: int,
    hist_sub: wp.array4d(dtype=float),
):
    """Gather the history [E, K, 3, 3] of the selected particles; the last frame is the current position."""
    env, k = wp.tid()
    p_idx = env * capacity + int(topk[env, k])
    h = history[p_idx]
    x = pos_in[p_idx]
    for a in range(3):
        hist_sub[env, k, 0, a] = h[0, a]
        hist_sub[env, k, 1, a] = h[1, a]
    hist_sub[env, k, 2, 0] = x[0]
    hist_sub[env, k, 2, 1] = x[1]
    hist_sub[env, k, 2, 2] = x[2]


@wp.kernel
def advance_stationary_history(
    pos_in: wp.array(dtype=wp.vec3),
    history: wp.array(dtype=wp.mat33),
):
    """Advance the history of every particle as if it did not move; active particles are overwritten afterwards."""
    tid = wp.tid()
    x = pos_in[tid]
    h = history[tid]
    history[tid] = wp.mat33(
        h[1, 0],
        h[1, 1],
        h[1, 2],
        x[0],
        x[1],
        x[2],
        x[0],
        x[1],
        x[2],
    )


@wp.kernel
def scatter_active_step(
    topk: wp.array2d(dtype=wp.int64),
    next_pos_sub: wp.array3d(dtype=float),
    next_vel_sub: wp.array3d(dtype=float),
    gamma: wp.array2d(dtype=float),
    tray_lower: wp.array2d(dtype=float),
    tray_upper: wp.array2d(dtype=float),
    capacity: int,
    dt: float,
    sleep_distance: float,
    pos_in: wp.array(dtype=wp.vec3),
    pos_out: wp.array(dtype=wp.vec3),
    vel_out: wp.array(dtype=wp.vec3),
    history: wp.array(dtype=wp.mat33),
):
    """Write the predicted step of the active particles, blended by ``gamma`` and clamped to the tray.

    A particle that would move less than ``sleep_distance`` in this step stays where it is with zero velocity.
    """
    env, k = wp.tid()
    p_idx = env * capacity + int(topk[env, k])
    x_cur = pos_in[p_idx]
    x_pred = wp.vec3(next_pos_sub[env, k, 0], next_pos_sub[env, k, 1], next_pos_sub[env, k, 2])
    v_pred = wp.vec3(next_vel_sub[env, k, 0], next_vel_sub[env, k, 1], next_vel_sub[env, k, 2])
    g = gamma[env, k]
    x_raw = x_cur + g * (x_pred - x_cur)
    x_new = wp.vec3(
        wp.clamp(x_raw[0], tray_lower[env, 0], tray_upper[env, 0]),
        wp.clamp(x_raw[1], tray_lower[env, 1], tray_upper[env, 1]),
        wp.clamp(x_raw[2], tray_lower[env, 2], tray_upper[env, 2]),
    )
    if wp.length(x_new - x_cur) < sleep_distance:
        x_new = x_cur
        v_pred = wp.vec3(0.0, 0.0, 0.0)
    v_eff = (x_new - x_cur) / wp.max(dt, 1.0e-6)
    pos_out[p_idx] = x_new
    vel_out[p_idx] = v_eff
    h = history[p_idx]
    x_prev = x_new - dt * (g * v_pred)
    x_prev_clamped = wp.vec3(
        wp.clamp(x_prev[0], tray_lower[env, 0], tray_upper[env, 0]),
        wp.clamp(x_prev[1], tray_lower[env, 1], tray_upper[env, 1]),
        wp.clamp(x_prev[2], tray_lower[env, 2], tray_upper[env, 2]),
    )
    history[p_idx] = wp.mat33(
        h[0, 0],
        h[0, 1],
        h[0, 2],
        x_prev_clamped[0],
        x_prev_clamped[1],
        x_prev_clamped[2],
        x_new[0],
        x_new[1],
        x_new[2],
    )


@wp.kernel
def gather_collider_state(
    mapping: wp.array[int],
    source_q: wp.array[wp.transform],
    source_qd: wp.array[wp.spatial_vector],
    target_q: wp.array[wp.transform],
    target_qd: wp.array[wp.spatial_vector],
):
    """Copy the rigid-body state that the neural step reads, from the coupled state to the solver's own buffers."""
    i = wp.tid()
    target_q[i] = source_q[mapping[i]]
    target_qd[i] = source_qd[mapping[i]]


@wp.kernel
def rewind_proxy(
    mapping: wp.array[int],
    forces: wp.array[wp.spatial_vector],
    poses: wp.array[wp.transform],
    inv_mass: wp.array[float],
    inv_inertia: wp.array[wp.mat33],
    velocities: wp.array[wp.spatial_vector],
    dt: float,
):
    """Remove the lagged coupling wrench from the proxy body velocities."""
    i = wp.tid()
    target = mapping[i]
    if target >= 0:
        wrench = forces[target]
        q = wp.transform_get_rotation(poses[i])
        linear = dt * inv_mass[i] * wp.spatial_top(wrench)
        angular = dt * wp.quat_rotate(q, inv_inertia[i] * wp.quat_rotate_inv(q, wp.spatial_bottom(wrench)))
        velocities[i] -= wp.spatial_vector(linear, angular)


@wp.kernel
def scatter_wrench(
    ids: wp.array[int], mapping: wp.array[int], source: wp.array[wp.spatial_vector], target: wp.array[wp.spatial_vector]
):
    """Write the reconstructed sand-on-body wrenches to the coupled body force array."""
    i = wp.tid()
    body = mapping[ids[i]]
    if body >= 0:
        target[body] = source[i]


# ---------------------------------------------------------------------------------------------------------------------
# Contact detection and impulse solve, launched by ContactWrenchSolver (double precision, CONTACT_MODULE)
# ---------------------------------------------------------------------------------------------------------------------


class ContactProblem:
    """One rigid collider and its candidate contact particles."""

    body: int
    shape: int
    env_idx: int
    count: int
    shape_code: int
    mass: float
    friction: float
    surface_pose: wp.array(dtype=wp.transform)
    sizes: wp.array(dtype=wp.vec3)
    free: wp.array(dtype=wp.vec3d)
    initial: wp.array(dtype=wp.vec3d)
    inverse: wp.array(dtype=wp.mat33d)
    offsets: wp.array(dtype=int)
    ids: wp.array(dtype=int)
    frames: wp.array(dtype=wp.mat33d)
    angular: wp.array(dtype=wp.mat33d)
    response: wp.array(dtype=wp.mat33d)
    compliance: wp.array(dtype=wp.mat33d)
    particle_slack: wp.array(dtype=wp.vec2d)
    contact_velocity: wp.array(dtype=wp.vec3d)
    gap: wp.array(dtype=wp.float64)
    particle_mass: wp.array(dtype=float)
    impulses: wp.array(dtype=wp.vec3d)
    stats: wp.array(dtype=wp.float64)


ContactProblem = wp.struct(ContactProblem, module=CONTACT_MODULE)


@wp.func(module=CONTACT_MODULE)
def collider_surface(point: wp.vec3d, pose: wp.transform, size: wp.vec3, shape: int):
    """Signed distance, outward normal and closest surface point of a collider (shape codes of the wrench solver).

    Shape codes: 0 box, 1 sphere, 2 cylinder, 4 capsule; ``size`` is the full extent.
    """
    center = wp.vec3d(wp.transform_get_translation(pose))
    rotation = wp.normalize(wp.quatd(wp.transform_get_rotation(pose)))
    local = wp.quat_rotate_inv(rotation, point - center)
    half = wp.vec3d(size) * wp.float64(0.5)
    normal = wp.vec3d(1.0, 0.0, 0.0)
    distance = wp.float64(0.0)
    eps = wp.float64(1.0e-15)
    if shape == 1:
        length = wp.length(local)
        distance = length - half[0]
        if length > eps:
            normal = local / length
    elif shape == 2:
        radial = wp.sqrt(local[0] * local[0] + local[1] * local[1])
        direction = wp.vec2d(1.0, 0.0)
        if radial > eps:
            direction = wp.vec2d(local[0], local[1]) / radial
        q_cylinder = wp.vec2d(radial - half[0], wp.abs(local[2]) - half[2])
        positive_cylinder = wp.vec2d(wp.max(q_cylinder[0], wp.float64(0.0)), wp.max(q_cylinder[1], wp.float64(0.0)))
        length = wp.length(positive_cylinder)
        distance = length + wp.min(wp.max(q_cylinder[0], q_cylinder[1]), wp.float64(0.0))
        weights = wp.vec2d(1.0, 0.0)
        if length > eps:
            weights = positive_cylinder / length
        elif q_cylinder[1] > q_cylinder[0]:
            weights = wp.vec2d(0.0, 1.0)
        sign_z = wp.float64(1.0)
        if local[2] < wp.float64(0.0):
            sign_z = wp.float64(-1.0)
        normal = wp.vec3d(direction[0] * weights[0], direction[1] * weights[0], sign_z * weights[1])
    elif shape == 4:
        z_cap = wp.clamp(local[2], -half[1], half[1])
        diff = wp.vec3d(local[0], local[1], local[2] - z_cap)
        length = wp.length(diff)
        distance = length - half[0]
        if length > eps:
            normal = diff / length
    else:
        q = wp.vec3d(wp.abs(local[0]), wp.abs(local[1]), wp.abs(local[2])) - half
        positive = wp.vec3d(wp.max(q[0], wp.float64(0.0)), wp.max(q[1], wp.float64(0.0)), wp.max(q[2], wp.float64(0.0)))
        length = wp.length(positive)
        distance = length + wp.min(wp.max(q[0], wp.max(q[1], q[2])), wp.float64(0.0))
        sign = wp.vec3d(1.0, 1.0, 1.0)
        for axis in range(3):
            if local[axis] < wp.float64(0.0):
                sign[axis] = wp.float64(-1.0)
        if length > eps:
            normal = wp.cw_mul(sign, positive) / length
        else:
            axis = int(0)
            if q[1] > q[axis]:
                axis = 1
            if q[2] > q[axis]:
                axis = 2
            normal = wp.vec3d(0.0)
            normal[axis] = sign[axis]
    normal = wp.quat_rotate(rotation, normal)
    return distance, normal, point - distance * normal


@wp.func(module=CONTACT_MODULE)
def friction_disk(a: wp.float64, b: wp.float64, d: wp.float64, rhs: wp.vec2d, radius: wp.float64):
    """Solve the 2x2 tangential problem and project the impulse onto the friction disk of the given radius."""
    result = wp.vec2d(0.0)
    if radius > wp.float64(0.0):
        det = a * d - b * b
        result = wp.vec2d(d * rhs[0] - b * rhs[1], a * rhs[1] - b * rhs[0]) / det
        if wp.length(result) > radius:
            lower = wp.float64(0.0)
            upper = wp.length(rhs) / radius
            multiplier = wp.float64(0.5) * (lower + upper)
            for k in range(45):
                aa = a + multiplier
                dd = d + multiplier
                det = aa * dd - b * b
                result = wp.vec2d(dd * rhs[0] - b * rhs[1], aa * rhs[1] - b * rhs[0]) / det
                norm = wp.length(result)
                if wp.abs(norm - radius) <= wp.float64(1.0e-12) * radius:
                    break
                if norm > radius:
                    lower = multiplier
                else:
                    upper = multiplier
                inverse_x = wp.vec2d(dd * result[0] - b * result[1], aa * result[1] - b * result[0]) / det
                candidate = multiplier + (norm - radius) * norm / wp.dot(result, inverse_x)
                if candidate <= lower or candidate >= upper:
                    candidate = wp.float64(0.5) * (lower + upper)
                multiplier = candidate
    return result


@wp.func(module=CONTACT_MODULE)
def contact_response(
    q: vec6d,
    frame: wp.mat33d,
    angular: wp.mat33d,
    material: wp.vec3d,
    gap: wp.float64,
    mobility_n: wp.float64,
    mobility_t: wp.float64,
    friction: wp.float64,
):
    """Closed-form impulse of one contact at a fixed body twist ``q``, and its derivative."""
    relative = frame * wp.vec3d(q[0], q[1], q[2]) + angular * wp.vec3d(q[3], q[4], q[5]) - material
    normal = wp.max(wp.float64(0.0), -(relative[0] + gap) / mobility_n)
    impulse = wp.vec3d(normal, 0.0, 0.0)
    derivative = wp.mat33d(0.0)
    if normal > wp.float64(0.0):
        derivative[0, 0] = -wp.float64(1.0) / mobility_n
        tangent = -wp.vec2d(relative[1], relative[2]) / mobility_t
        length = wp.length(tangent)
        radius = friction * normal
        if radius > wp.float64(0.0) and length <= radius:
            impulse[1] = tangent[0]
            impulse[2] = tangent[1]
            derivative[1, 1] = -wp.float64(1.0) / mobility_t
            derivative[2, 2] = -wp.float64(1.0) / mobility_t
        elif radius > wp.float64(0.0):
            direction = tangent / length
            impulse[1] = radius * direction[0]
            impulse[2] = radius * direction[1]
            derivative[1, 0] = -friction * direction[0] / mobility_n
            derivative[2, 0] = -friction * direction[1] / mobility_n
            scale = -radius / (length * mobility_t)
            derivative[1, 1] = scale * (wp.float64(1.0) - direction[0] * direction[0])
            derivative[2, 2] = scale * (wp.float64(1.0) - direction[1] * direction[1])
            derivative[1, 2] = -scale * direction[0] * direction[1]
            derivative[2, 1] = derivative[1, 2]
    return impulse, derivative


@wp.func(module=CONTACT_MODULE)
def linear_solve_6(matrix: mat66d, rhs: vec6d):
    """Solve a 6x6 linear system by Gaussian elimination with partial pivoting."""
    a = matrix
    b = rhs
    for k in range(6):
        pivot = k
        for r in range(k + 1, 6):
            if wp.abs(a[r, k]) > wp.abs(a[pivot, k]):
                pivot = r
        for c in range(6):
            tmp = a[k, c]
            a[k, c] = a[pivot, c]
            a[pivot, c] = tmp
        tmp = b[k]
        b[k] = b[pivot]
        b[pivot] = tmp
        if wp.abs(a[k, k]) < wp.float64(1e-20):
            return vec6d(0.0)
        for r in range(k + 1, 6):
            factor = a[r, k] / a[k, k]
            for c in range(k + 1, 6):
                a[r, c] = a[r, c] - factor * a[k, c]
            b[r] = b[r] - factor * b[k]
    x = vec6d(0.0)
    for rev in range(6):
        r = 5 - rev
        v = b[r]
        for c in range(r + 1, 6):
            v -= a[r, c] * x[c]
        x[r] = v / a[r, r]
    return x


@wp.func(module=CONTACT_MODULE)
def contact_mobilities(
    n: wp.vec3d,
    particle_mass: wp.float64,
    base_scale: wp.float64,
    horiz_scale: wp.float64,
    up_scale: wp.float64,
    tangent_scale: wp.float64,
    normal_power: wp.float64,
) -> wp.vec2d:
    """Normal and tangential compliance (1 / effective mass) of a contact particle.

    The effective mass is the particle mass times a scale that depends on the contact normal direction only.
    """
    nz = n[2]
    up_part = wp.pow(wp.max(nz, wp.float64(0.0)), normal_power)
    down_part = wp.pow(wp.max(-nz, wp.float64(0.0)), normal_power)
    horiz_part = wp.max(wp.float64(0.0), wp.float64(1.0) - up_part - down_part)
    eff_n = wp.max(
        base_scale * up_part + horiz_scale * horiz_part + up_scale * down_part,
        wp.float64(0.5),
    )
    eff_t = wp.max(tangent_scale, wp.float64(0.5))
    slack_n = wp.float64(1.0) / (particle_mass * eff_n)
    slack_t = wp.float64(1.0) / (particle_mass * eff_t)
    return wp.vec2d(slack_n, slack_t)


@wp.kernel(module=CONTACT_MODULE)
def gather_active_particles(
    active_idx: wp.array(dtype=wp.int32),
    k_eval: int,
    particles_per_env: int,
    x_src: wp.array(dtype=wp.vec3),
    v_src: wp.array(dtype=wp.vec3),
    m_src: wp.array(dtype=float),
    x_dst: wp.array(dtype=wp.vec3),
    v_dst: wp.array(dtype=wp.vec3),
    m_dst: wp.array(dtype=float),
):
    """Gather position, velocity and mass of the active particles of every environment."""
    env_idx, i = wp.tid()
    dst = env_idx * k_eval + i
    src = env_idx * particles_per_env + int(active_idx[dst])
    x_dst[dst] = x_src[src]
    v_dst[dst] = v_src[src]
    m_dst[dst] = m_src[src]


@wp.kernel(module=CONTACT_MODULE)
def prepare_contact_bodies(
    problems: wp.array(dtype=ContactProblem),
    poses: wp.array(dtype=wp.transform),
    velocities: wp.array(dtype=wp.spatial_vector),
    body_inertia: wp.array(dtype=wp.mat33),
    shape_transform: wp.array(dtype=wp.transform),
):
    """Collider surface pose, free body twist and world-frame inverse inertia of every contact problem."""
    problem_idx = wp.tid()
    p = problems[problem_idx]
    body = p.body
    shape = p.shape
    p.offsets[p.count - 1] = 0
    p.surface_pose[0] = poses[body] * shape_transform[shape]
    p.free[0] = wp.vec3d(wp.spatial_top(velocities[body]))
    p.free[1] = wp.vec3d(wp.spatial_bottom(velocities[body]))
    inertia = wp.mat33d(body_inertia[body])
    trace = inertia[0, 0] + inertia[1, 1] + inertia[2, 2]
    if trace > wp.float64(1.0e-12):
        rotation = wp.quat_to_matrix(wp.normalize(wp.quatd(wp.transform_get_rotation(poses[body]))))
        p.inverse[0] = wp.inverse(rotation * inertia * wp.transpose(rotation))
    else:
        p.inverse[0] = wp.mat33d(0.0)


@wp.kernel(module=CONTACT_MODULE)
def prepare_contacts(
    problems: wp.array(dtype=ContactProblem),
    grid: wp.uint64,
    x_all: wp.array(dtype=wp.vec3),
    v_all: wp.array(dtype=wp.vec3),
    radius: float,
    dt: float,
    contact_band: float,
    base_scale: float,
    horiz_scale: float,
    up_scale: float,
    tangent_scale: float,
    normal_power: float,
    stabilization: float,
    flags_flat: wp.array(dtype=int),
):
    """Detect contacts and build the per-contact frame, compliance, material velocity and gap.

    The particle velocity is extrapolated to the contact point with a weighted least-squares velocity gradient
    over the neighbours within ``radius``.
    """
    problem_idx, i = wp.tid()
    p = problems[problem_idx]
    k_eval = p.count
    if i >= k_eval:
        return
    p.impulses[i] = wp.vec3d(0.0)
    flat_idx = problem_idx * k_eval + i
    flags_flat[flat_idx] = 0
    pm_f = p.particle_mass[i]
    if pm_f <= 0.0:
        return

    env_offset = p.env_idx * k_eval
    xi = x_all[env_offset + i]
    vi = v_all[env_offset + i]
    pose_b = p.surface_pose[0]
    distance, normal, point = collider_surface(wp.vec3d(xi), pose_b, p.sizes[0], p.shape_code)
    if contact_band > 0.0 and distance > wp.float64(contact_band):
        return

    arm = point - wp.vec3d(wp.transform_get_translation(pose_b))
    surface = p.free[0] + wp.cross(p.free[1], arm)
    app0 = wp.dot(normal, surface - wp.vec3d(vi))
    if distance > (wp.max(app0, wp.float64(0.0)) + wp.float64(4.0)) * wp.float64(dt):
        return

    lhs = wp.mat33(0.0)
    rhs = wp.mat33(0.0)
    query = wp.hash_grid_query(grid, xi, radius + 1.0e-6, p.env_idx)
    j = int(0)
    while wp.hash_grid_query_next(query, j):
        dx = wp.vec3(x_all[j]) - xi
        dv = wp.vec3(v_all[j]) - vi
        dist = wp.length(dx)
        q = dist / float(radius)
        if q < float(1.0):
            a = float(1.0) - q
            weight = a * a * a * a * (float(1.0) + float(4.0) * q)
            lhs += weight * wp.outer(dx, dx)
            rhs += weight * wp.outer(dx, dv)
    ridge = float(0.01) * wp.trace(lhs) / float(3.0) + float(1.0e-12)
    lhs += wp.mat33(ridge, 0.0, 0.0, 0.0, ridge, 0.0, 0.0, 0.0, ridge)
    grad = wp.mat33d(wp.inverse(lhs) * rhs)

    velocity = wp.vec3d(vi) + wp.transpose(grad) * (point - wp.vec3d(xi))
    v_rel_world = surface - velocity
    approach = wp.dot(normal, v_rel_world)
    if distance <= wp.max(approach, wp.float64(0.0)) * wp.float64(dt):
        flags_flat[flat_idx] = 1
        n = -normal
        axis = int(0)
        if wp.abs(n[1]) < wp.abs(n[axis]):
            axis = 1
        if wp.abs(n[2]) < wp.abs(n[axis]):
            axis = 2
        basis = wp.vec3d(0.0)
        basis[axis] = wp.float64(1.0)
        t = wp.normalize(wp.cross(n, basis))
        t2 = wp.cross(n, t)
        frame = wp.mat33d(n[0], n[1], n[2], t[0], t[1], t[2], t2[0], t2[1], t2[2])
        a0 = wp.cross(arm, n)
        a1 = wp.cross(arm, t)
        a2 = wp.cross(arm, t2)
        angular = wp.mat33d(a0[0], a0[1], a0[2], a1[0], a1[1], a1[2], a2[0], a2[1], a2[2])
        respond = p.inverse[0] * wp.transpose(angular)
        k = frame * wp.transpose(frame) / wp.float64(p.mass) + angular * respond

        pm = wp.max(wp.float64(pm_f), wp.float64(1.0e-6))
        slacks = contact_mobilities(
            n,
            pm,
            wp.float64(base_scale),
            wp.float64(horiz_scale),
            wp.float64(up_scale),
            wp.float64(tangent_scale),
            wp.float64(normal_power),
        )
        slack_n = slacks[0]
        slack_t = slacks[1]
        k += wp.mat33d(slack_n, 0.0, 0.0, 0.0, slack_t, 0.0, 0.0, 0.0, slack_t)

        p.frames[i] = frame
        p.angular[i] = angular
        p.response[i] = respond
        p.compliance[i] = k
        p.particle_slack[i] = slacks
        p.contact_velocity[i] = frame * velocity
        p.gap[i] = (
            wp.max(distance, wp.float64(0.0)) + wp.float64(stabilization) * wp.min(distance, wp.float64(0.0))
        ) / wp.float64(dt)


@wp.kernel(module=CONTACT_MODULE)
def compact_contacts(
    problems: wp.array(dtype=ContactProblem),
    flags_flat: wp.array(dtype=int),
    offsets_flat: wp.array(dtype=int),
):
    """Write the ids of the flagged contacts of every problem contiguously, and their count."""
    problem_idx, i = wp.tid()
    p = problems[problem_idx]
    k_eval = p.count
    if i >= k_eval:
        return
    base = problem_idx * k_eval
    prev_total = int(0)
    if base > 0:
        prev_total = offsets_flat[base - 1]
    local_off = offsets_flat[base + i] - prev_total
    if flags_flat[base + i] != 0:
        p.ids[local_off - 1] = i
    if i == k_eval - 1:
        p.offsets[k_eval - 1] = local_off


@wp.func(module=CONTACT_MODULE)
def initialize_body_velocity_block(
    lane: int,
    offsets: wp.array(dtype=int),
    ids: wp.array(dtype=int),
    frames: wp.array(dtype=wp.mat33d),
    angular: wp.array(dtype=wp.mat33d),
    response: wp.array(dtype=wp.mat33d),
    particle_slack: wp.array(dtype=wp.vec2d),
    material: wp.array(dtype=wp.vec3d),
    gap: wp.array(dtype=wp.float64),
    free: wp.array(dtype=wp.vec3d),
    mass: float,
    friction: float,
    impulses: wp.array(dtype=wp.vec3d),
    initial: wp.array(dtype=wp.vec3d),
):
    """Warm start: solve the body twist with independent contacts by a damped Newton iteration.

    One thread block of 128 lanes shares the contacts of one collider. The result seeds the Jacobi iterations.
    """
    count = offsets[offsets.shape[0] - 1]
    origin = vec6d(free[0][0], free[0][1], free[0][2], free[1][0], free[1][1], free[1][2])
    q = origin
    error = wp.float64(0.0)
    if count > 0:
        for _it in range(25):
            local_force = vec6d(0.0)
            local_jacobian = mat66d(0.0)
            k = lane
            while k < count:
                i = ids[k]
                slacks = particle_slack[i]
                impulse, derivative = contact_response(
                    q,
                    frames[i],
                    angular[i],
                    material[i],
                    gap[i],
                    slacks[0],
                    slacks[1],
                    wp.float64(friction),
                )
                j = mat36d(0.0)
                b = mat63d(0.0)
                for r in range(3):
                    for c in range(3):
                        j[r, c] = frames[i][r, c]
                        j[r, c + 3] = angular[i][r, c]
                        b[c, r] = frames[i][r, c] / wp.float64(mass)
                        b[c + 3, r] = response[i][c, r]
                local_force += b * impulse
                local_jacobian += b * derivative * j
                k += 128
            tf = wp.tile_sum(wp.tile(local_force), axis=1)
            tj = wp.tile_sum(wp.tile(local_jacobian), axis=2)
            force = vec6d(0.0)
            jacobian = mat66d(0.0)
            for r in range(6):
                force[r] = wp.tile_extract(tf, r)
                for c in range(6):
                    jacobian[r, c] = -wp.tile_extract(tj, r, c)
                jacobian[r, r] += wp.float64(1.0)
            residual = q - origin - force
            error = wp.length(residual)
            if error < wp.float64(1e-11):
                break
            delta = linear_solve_6(jacobian, residual)
            alpha = wp.float64(1.0)
            accepted = bool(False)
            for _trial in range(12):
                candidate = q - alpha * delta
                local_candidate = vec6d(0.0)
                k = lane
                while k < count:
                    i = ids[k]
                    slacks = particle_slack[i]
                    impulse, _unused = contact_response(
                        candidate,
                        frames[i],
                        angular[i],
                        material[i],
                        gap[i],
                        slacks[0],
                        slacks[1],
                        wp.float64(friction),
                    )
                    linear = wp.transpose(frames[i]) * impulse / wp.float64(mass)
                    spin = response[i] * impulse
                    local_candidate += vec6d(linear[0], linear[1], linear[2], spin[0], spin[1], spin[2])
                    k += 128
                tc = wp.tile_sum(wp.tile(local_candidate), axis=1)
                candidate_force = vec6d(0.0)
                for r in range(6):
                    candidate_force[r] = wp.tile_extract(tc, r)
                new_error = wp.length(candidate - origin - candidate_force)
                if new_error < error * (wp.float64(1.0) - wp.float64(1e-4) * alpha):
                    q = candidate
                    accepted = True
                    break
                alpha *= wp.float64(0.5)
            if not accepted:
                break
    local_force = vec6d(0.0)
    k = lane
    while k < count:
        i = ids[k]
        slacks = particle_slack[i]
        impulse, _unused = contact_response(
            q,
            frames[i],
            angular[i],
            material[i],
            gap[i],
            slacks[0],
            slacks[1],
            wp.float64(friction),
        )
        impulses[i] = impulse
        linear = wp.transpose(frames[i]) * impulse / wp.float64(mass)
        spin = response[i] * impulse
        local_force += vec6d(linear[0], linear[1], linear[2], spin[0], spin[1], spin[2])
        k += 128
    tf_final = wp.tile_sum(wp.tile(local_force), axis=1)
    force = vec6d(0.0)
    for r in range(6):
        force[r] = wp.tile_extract(tf_final, r)
    final_error = wp.length(q - origin - force)
    fallback = not wp.isfinite(final_error) or final_error > wp.float64(1e-2)
    if fallback:
        k = lane
        while k < count:
            impulses[ids[k]] = wp.vec3d(0.0)
            k += 128
        force = vec6d(0.0)
    if lane == 0:
        initial[0] = free[0] + wp.vec3d(force[0], force[1], force[2])
        initial[1] = free[1] + wp.vec3d(force[3], force[4], force[5])


@wp.func(module=CONTACT_MODULE)
def solve_contact_block(
    lane: int,
    offsets: wp.array(dtype=int),
    ids: wp.array(dtype=int),
    frames: wp.array(dtype=wp.mat33d),
    angular: wp.array(dtype=wp.mat33d),
    response: wp.array(dtype=wp.mat33d),
    compliance: wp.array(dtype=wp.mat33d),
    particle_slack: wp.array(dtype=wp.vec2d),
    material: wp.array(dtype=wp.vec3d),
    gap: wp.array(dtype=wp.float64),
    free: wp.array(dtype=wp.vec3d),
    mass: float,
    friction: float,
    dt: float,
    max_iterations: int,
    tolerance: wp.float64,
    relaxation_in: float,
    impulses: wp.array(dtype=wp.vec3d),
    wrench: wp.array(dtype=wp.spatial_vector),
    body: int,
    stats: wp.array(dtype=wp.float64),
):
    """Relaxed Jacobi iterations on the contact impulses of one collider, coupled through the body twist.

    Every contact solves its normal impulse (non-negative) and then its tangential impulse (friction disk) against
    the current body twist; the twist is updated once per iteration from the sum of the impulse changes.
    """
    count = offsets[offsets.shape[0] - 1]
    linear = free[0]
    spin = free[1]
    total = wp.vec3d(0.0)
    torque = wp.vec3d(0.0)
    residual = wp.float64(0.0)
    iterations = int(0)
    if count > 0:
        relaxation = wp.max(wp.float64(relaxation_in), wp.float64(1.0) / wp.float64(count))
        for iteration in range(max_iterations):
            local_delta = wp.vec3d(0.0)
            local_spin = wp.vec3d(0.0)
            local_residual = wp.float64(0.0)
            k = lane
            while k < count:
                i = ids[k]
                frame = frames[i]
                aj = angular[i]
                respond = response[i]
                matrix = compliance[i]
                old = impulses[i]
                slacks = particle_slack[i]
                slack_n = slacks[0]
                slack_t = slacks[1]
                relative = (
                    frame * linear
                    + aj * spin
                    - material[i]
                    + wp.vec3d(old[0] * slack_n, old[1] * slack_t, old[2] * slack_t)
                )
                normal = wp.max(wp.float64(0.0), old[0] - (relative[0] + gap[i]) / matrix[0, 0])
                dn = normal - old[0]
                increment = wp.vec3d(dn, 0.0, 0.0)
                local_linear = linear + wp.transpose(frame) * increment / wp.float64(mass)
                local_angular = spin + respond * increment
                relative = (
                    frame * local_linear
                    + aj * local_angular
                    - material[i]
                    + wp.vec3d(normal * slack_n, old[1] * slack_t, old[2] * slack_t)
                )
                rhs = wp.vec2d(
                    matrix[1, 1] * old[1] + matrix[1, 2] * old[2] - relative[1],
                    matrix[2, 1] * old[1] + matrix[2, 2] * old[2] - relative[2],
                )
                tangent = friction_disk(matrix[1, 1], matrix[1, 2], matrix[2, 2], rhs, wp.float64(friction) * normal)
                delta_t = tangent - wp.vec2d(old[1], old[2])
                change = relaxation * wp.vec3d(dn, delta_t[0], delta_t[1])
                impulses[i] = old + change
                local_delta += wp.transpose(frame) * change
                local_spin += respond * change
                local_residual = wp.max(local_residual, wp.max(wp.abs(dn), wp.length(delta_t)))
                k += 128
            td = wp.tile_sum(wp.tile(local_delta), axis=1)
            ts = wp.tile_sum(wp.tile(local_spin), axis=1)
            delta = wp.vec3d(wp.tile_extract(td, 0), wp.tile_extract(td, 1), wp.tile_extract(td, 2))
            ds = wp.vec3d(wp.tile_extract(ts, 0), wp.tile_extract(ts, 1), wp.tile_extract(ts, 2))
            residual = wp.tile_extract(wp.tile_max(wp.tile(local_residual)), 0)
            linear += delta / wp.float64(mass)
            spin += ds
            iterations = iteration + 1
            if residual < tolerance:
                break
        local_total = wp.vec3d(0.0)
        local_torque = wp.vec3d(0.0)
        k = lane
        while k < count:
            i = ids[k]
            local_total += wp.transpose(frames[i]) * impulses[i]
            local_torque += wp.transpose(angular[i]) * impulses[i]
            k += 128
        ttot = wp.tile_sum(wp.tile(local_total), axis=1)
        ttor = wp.tile_sum(wp.tile(local_torque), axis=1)
        total = wp.vec3d(wp.tile_extract(ttot, 0), wp.tile_extract(ttot, 1), wp.tile_extract(ttot, 2))
        torque = wp.vec3d(wp.tile_extract(ttor, 0), wp.tile_extract(ttor, 1), wp.tile_extract(ttor, 2))
    if lane == 0:
        wrench[body] = wp.spatial_vector(wp.vec3(total / wp.float64(dt)), wp.vec3(torque / wp.float64(dt)))
        stats[0] = wp.float64(count)
        stats[1] = wp.float64(iterations)
        stats[2] = residual


@wp.kernel(module=CONTACT_MODULE)
def initialize_contacts(
    problems: wp.array(dtype=ContactProblem),
):
    """Warm start of every contact problem; launch with ``(problems, 128)`` and ``block_dim=128``."""
    problem_idx, lane = wp.tid()
    p = problems[problem_idx]
    initialize_body_velocity_block(
        lane,
        p.offsets,
        p.ids,
        p.frames,
        p.angular,
        p.response,
        p.particle_slack,
        p.contact_velocity,
        p.gap,
        p.free,
        p.mass,
        p.friction,
        p.impulses,
        p.initial,
    )


@wp.kernel(module=CONTACT_MODULE)
def solve_contacts(
    problems: wp.array(dtype=ContactProblem),
    dt: float,
    max_iterations: int,
    tolerance: wp.float64,
    relaxation_in: float,
    wrench: wp.array(dtype=wp.spatial_vector),
):
    """Jacobi solve of every contact problem; launch with ``(problems, 128)`` and ``block_dim=128``."""
    problem_idx, lane = wp.tid()
    p = problems[problem_idx]
    solve_contact_block(
        lane,
        p.offsets,
        p.ids,
        p.frames,
        p.angular,
        p.response,
        p.compliance,
        p.particle_slack,
        p.contact_velocity,
        p.gap,
        p.initial,
        p.mass,
        p.friction,
        dt,
        max_iterations,
        tolerance,
        relaxation_in,
        p.impulses,
        wrench,
        problem_idx,
        p.stats,
    )


# ---------------------------------------------------------------------------------------------------------------------
# Launch helper
# ---------------------------------------------------------------------------------------------------------------------


def warp_stream(device: torch.device) -> wp.Stream:
    """Warp stream that matches the current torch stream, so both libraries queue work in order."""
    torch_stream = torch.cuda.current_stream(device)
    stream = wp.get_stream(str(device))
    if stream.cuda_stream != torch_stream.cuda_stream:
        stream = wp.stream_from_torch(torch_stream)
    return stream
