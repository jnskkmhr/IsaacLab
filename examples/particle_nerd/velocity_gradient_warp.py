# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compute the same weighted particle velocity fit on a Warp device."""

import numpy as np
import warp as wp


@wp.kernel
def fit_gradient(
    grid: wp.uint64,
    query_position: wp.array(dtype=wp.vec3),
    position: wp.array(dtype=wp.vec3d),
    velocity: wp.array(dtype=wp.vec3d),
    radius: wp.float64,
    regularization: wp.float64,
    output: wp.array(dtype=wp.mat33d),
):
    i = wp.tid()
    lhs = wp.mat33d(0.0)
    rhs = wp.mat33d(0.0)
    # The hash grid uses float32 positions. Expand only the search; the exact
    # distance and compact-support weight are evaluated in float64 below.
    query = wp.hash_grid_query(grid, query_position[i], wp.float32(radius) + 1.0e-6)
    j = int(0)
    while wp.hash_grid_query_next(query, j):
        dx = position[j] - position[i]
        dv = velocity[j] - velocity[i]
        q = wp.length(dx) / radius
        if q < wp.float64(1.0):
            a = wp.float64(1.0) - q
            weight = a * a * a * a * (wp.float64(1.0) + wp.float64(4.0) * q)
            lhs += weight * wp.outer(dx, dx)
            rhs += weight * wp.outer(dx, dv)
    ridge = regularization * wp.trace(lhs) / wp.float64(3.0) + wp.float64(1.0e-12)
    lhs += wp.mat33d(ridge, 0.0, 0.0, 0.0, ridge, 0.0, 0.0, 0.0, ridge)
    output[i] = wp.inverse(lhs) * rhs


def estimate_velocity_gradient_warp(position, velocity, radius, regularization, device):
    """Fit every particle using every neighbor inside the physical radius."""
    queries = wp.array(np.asarray(position, dtype=np.float32), dtype=wp.vec3, device=device)
    positions = wp.array(position, dtype=wp.vec3d, device=device)
    velocities = wp.array(velocity, dtype=wp.vec3d, device=device)
    gradient = wp.empty(len(position), dtype=wp.mat33d, device=device)
    grid = wp.HashGrid(64, 64, 64, device=device)
    grid.build(queries, radius + 1.0e-6)
    wp.launch(
        fit_gradient,
        dim=len(position),
        inputs=[grid.id, queries, positions, velocities, radius, regularization],
        outputs=[gradient],
        device=device,
    )
    return gradient.numpy()
