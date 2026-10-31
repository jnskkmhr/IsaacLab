# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Particle displacement integration; supports full and active-indexed updates."""

import warp as wp


@wp.kernel
def update(
    history: wp.array3d(dtype=float),
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    displacement: wp.array2d(dtype=float),
    dt: float,
):
    i = wp.tid()
    delta = wp.vec3(displacement[i, 0], displacement[i, 1], displacement[i, 2])
    next_x = x[i] + delta
    x[i] = next_x
    v[i] = delta / dt
    for t in range(history.shape[1] - 1):
        for c in range(3):
            history[i, t, c] = history[i, t + 1, c]
    for c in range(3):
        history[i, history.shape[1] - 1, c] = next_x[c]


@wp.kernel
def shift_history_zero_vel(
    history: wp.array3d(dtype=float),
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    h = history.shape[1]
    pos = x[i]
    v[i] = wp.vec3(0.0, 0.0, 0.0)
    for t in range(h - 1):
        for c in range(3):
            history[i, t, c] = history[i, t + 1, c]
    for c in range(3):
        history[i, h - 1, c] = pos[c]


@wp.kernel
def apply_active_displacement(
    history: wp.array3d(dtype=float),
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    active_ids: wp.array(dtype=int),
    active_mask: wp.array(dtype=float),
    displacement: wp.array2d(dtype=float),
    dt: float,
):
    slot = wp.tid()
    if active_mask[slot] <= 0.0:
        return
    i = active_ids[slot]
    h = history.shape[1]
    delta = wp.vec3(displacement[slot, 0], displacement[slot, 1], displacement[slot, 2])
    next_x = x[i] + delta
    x[i] = next_x
    v[i] = delta / dt
    for c in range(3):
        history[i, h - 1, c] = next_x[c]


@wp.kernel
def reset_particles(
    mask: wp.array(dtype=wp.bool),
    particles: int,
    initial: wp.array3d(dtype=float),
    history: wp.array3d(dtype=float),
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    dt: float,
):
    i = wp.tid()
    if not mask[i // particles]:
        return
    h = history.shape[1]
    for t in range(h):
        for a in range(3):
            history[i, t, a] = initial[i, t, a]
    position = wp.vec3(initial[i, h - 1, 0], initial[i, h - 1, 1], initial[i, h - 1, 2])
    previous = wp.vec3(initial[i, h - 2, 0], initial[i, h - 2, 1], initial[i, h - 2, 2])
    x[i] = position
    v[i] = (position - previous) / dt


@wp.kernel
def reset_rigid(
    mask: wp.array(dtype=wp.bool),
    initial_q: wp.array(dtype=float),
    initial_qd: wp.array(dtype=float),
    initial_pose: wp.array(dtype=wp.transform),
    initial_velocity: wp.array(dtype=wp.spatial_vector),
    q: wp.array(dtype=float),
    qd: wp.array(dtype=float),
    pose: wp.array(dtype=wp.transform),
    velocity: wp.array(dtype=wp.spatial_vector),
    force: wp.array(dtype=wp.spatial_vector),
):
    env = wp.tid()
    if not mask[env]:
        return
    for a in range(7):
        q[7 * env + a] = initial_q[7 * env + a]
    for a in range(6):
        qd[6 * env + a] = initial_qd[6 * env + a]
    pose[env] = initial_pose[env]
    velocity[env] = initial_velocity[env]
    force[env] = wp.spatial_vector(wp.vec3(0.0), wp.vec3(0.0))


@wp.kernel
def reset_loads(
    mask: wp.array(dtype=wp.bool),
    external: wp.array(dtype=wp.spatial_vector),
    contact: wp.array(dtype=wp.spatial_vector),
    stats: wp.array(dtype=wp.float64),
):
    env = wp.tid()
    if mask[env]:
        external[env] = wp.spatial_vector(wp.vec3(0.0), wp.vec3(0.0))
        contact[env] = wp.spatial_vector(wp.vec3(0.0), wp.vec3(0.0))
        for a in range(3):
            stats[3 * env + a] = wp.float64(0.0)
