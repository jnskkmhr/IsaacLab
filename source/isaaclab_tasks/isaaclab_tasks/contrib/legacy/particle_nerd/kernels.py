# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Neural history lifecycle, terrain coordinates and rigid proxy wrench exchange."""

import warp as wp


@wp.kernel
def gather_collider_state(
    mapping: wp.array[int],
    source_q: wp.array[wp.transform],
    source_qd: wp.array[wp.spatial_vector],
    target_q: wp.array[wp.transform],
    target_qd: wp.array[wp.spatial_vector],
):
    i = wp.tid()
    target_q[i] = source_q[mapping[i]]
    target_qd[i] = source_qd[mapping[i]]


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
    i = wp.tid()
    world = worlds[i]
    if all_worlds or mask[wp.where(world < 0, mask.shape[0] - 1, world)]:
        p, v = q[i], qd[i]
        history[i] = wp.matrix_from_rows(p - 2.0 * dt * v, p - dt * v, p)


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
    i = wp.tid()
    body = mapping[ids[i]]
    if body >= 0:
        target[body] = source[i]


@wp.kernel
def terrain_chart(
    points: wp.array[wp.vec3],
    mesh: wp.uint64,
    ray_z: float,
    ray_length: float,
    heights: wp.array[float],
    normals: wp.array[wp.vec3],
):
    i = wp.tid()
    p = points[i]
    query = wp.mesh_query_ray(mesh, wp.vec3(p[0], p[1], ray_z), wp.vec3(0.0, 0.0, -1.0), ray_length)
    if query.result:
        heights[i] = ray_z - query.t
        n = query.normal
        normals[i] = wp.where(n[2] < 0.0, -n, n)
    else:
        heights[i] = 0.0
        normals[i] = wp.vec3(0.0, 0.0, 1.0)


@wp.kernel
def terrain_chart_heights(
    points: wp.array[wp.vec3],
    mesh: wp.uint64,
    ray_z: float,
    ray_length: float,
    heights: wp.array[float],
):
    i = wp.tid()
    p = points[i]
    query = wp.mesh_query_ray(mesh, wp.vec3(p[0], p[1], ray_z), wp.vec3(0.0, 0.0, -1.0), ray_length)
    if query.result:
        heights[i] = ray_z - query.t
    else:
        heights[i] = 0.0


@wp.kernel
def extract_chart_cur_prev(
    slots: wp.array[wp.int64],
    history: wp.array[wp.mat33],
    src: wp.array[wp.vec3],
    has_terrain: bool,
    mesh: wp.uint64,
    ray_z: float,
    ray_length: float,
    x_cur_out: wp.array[wp.vec3],
    x_prev_out: wp.array[wp.vec3],
):
    i = wp.tid()
    s = int(slots[i])
    p_cur = src[s]
    p_prev = history[s][1]
    h_cur = float(0.0)
    h_prev = float(0.0)
    if has_terrain:
        q_cur = wp.mesh_query_ray(mesh, wp.vec3(p_cur[0], p_cur[1], ray_z), wp.vec3(0.0, 0.0, -1.0), ray_length)
        if q_cur.result:
            h_cur = ray_z - q_cur.t
        if p_prev[0] == p_cur[0] and p_prev[1] == p_cur[1]:
            h_prev = h_cur
        else:
            q_prev = wp.mesh_query_ray(mesh, wp.vec3(p_prev[0], p_prev[1], ray_z), wp.vec3(0.0, 0.0, -1.0), ray_length)
            if q_prev.result:
                h_prev = ray_z - q_prev.t
    x_cur_out[i] = wp.vec3(p_cur[0], p_cur[1], p_cur[2] - h_cur)
    x_prev_out[i] = wp.vec3(p_prev[0], p_prev[1], p_prev[2] - h_prev)


@wp.kernel
def extract_chart_history_sub(
    sub_slots: wp.array[wp.int64],
    history: wp.array[wp.mat33],
    src: wp.array[wp.vec3],
    has_terrain: bool,
    mesh: wp.uint64,
    ray_z: float,
    ray_length: float,
    hist_sub_out: wp.array[wp.mat33],
):
    i = wp.tid()
    s = int(sub_slots[i])
    h = history[s]
    p0 = h[0]
    p1 = h[1]
    p2 = src[s]
    if has_terrain:
        q2 = wp.mesh_query_ray(mesh, wp.vec3(p2[0], p2[1], ray_z), wp.vec3(0.0, 0.0, -1.0), ray_length)
        h2 = wp.where(q2.result, ray_z - q2.t, 0.0)
        h1 = h2
        if p1[0] != p2[0] or p1[1] != p2[1]:
            q1 = wp.mesh_query_ray(mesh, wp.vec3(p1[0], p1[1], ray_z), wp.vec3(0.0, 0.0, -1.0), ray_length)
            h1 = wp.where(q1.result, ray_z - q1.t, 0.0)
        h0 = h1
        if p0[0] != p1[0] or p0[1] != p1[1]:
            q0 = wp.mesh_query_ray(mesh, wp.vec3(p0[0], p0[1], ray_z), wp.vec3(0.0, 0.0, -1.0), ray_length)
            h0 = wp.where(q0.result, ray_z - q0.t, 0.0)
        p0 = wp.vec3(p0[0], p0[1], p0[2] - h0)
        p1 = wp.vec3(p1[0], p1[1], p1[2] - h1)
        p2 = wp.vec3(p2[0], p2[1], p2[2] - h2)
    hist_sub_out[i] = wp.matrix_from_rows(p0, p1, p2)


@wp.kernel
def advance_history(
    slots: wp.array[wp.int64],
    src: wp.array[wp.vec3],
    dst: wp.array[wp.vec3],
    history: wp.array[wp.mat33],
):
    i = wp.tid()
    s = int(slots[i])
    h = history[s]
    history[s] = wp.matrix_from_rows(h[1], src[s], dst[s])
