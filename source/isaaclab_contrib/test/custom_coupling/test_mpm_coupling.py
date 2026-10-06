# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Force units and world-frame wrench conventions for direct MPM coupling."""

import math

import numpy as np
import warp as wp

from isaaclab_contrib.custom_coupling.kernels import collect_mpm_body_wrenches, prepare_mpm_collider_state


def test_mpm_impulses_accumulate_wrench_about_body_com():
    """Sum collider impulses over the MPM timestep, with torque about the rotated COM."""
    with wp.ScopedDevice("cpu"):
        # Body 1's local COM (1, 0, 0) becomes world COM (1, 3, 3).
        rotation = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), math.pi / 2)
        body_q = wp.array([wp.transform_identity(), wp.transform(wp.vec3(1.0, 2.0, 3.0), rotation)], dtype=wp.transform)
        body_com = wp.array([[0, 0, 0], [1, 0, 0]], dtype=wp.vec3)
        collider_ids = wp.array([0, 0, 1, -1, 2], dtype=int)
        impulses = wp.array([[0, 0, 2], [1, 0, 0], [99, 99, 99], [99, 99, 99], [99, 99, 99]], dtype=wp.vec3)
        positions = wp.array([[1, 4, 3], [1, 3, 3], [0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=wp.vec3)
        # The second collider is static. Invalid and static collider impulses must be ignored.
        collider_bodies = wp.array([1, -1], dtype=int)
        wrenches = wp.zeros(2, dtype=wp.spatial_vector)
        wp.launch(
            collect_mpm_body_wrenches,
            dim=5,
            inputs=[0.5, collider_ids, impulses, positions, collider_bodies, body_q, body_com],
            outputs=[wrenches],
        )
        # F = impulse / 0.5 s; the first impulse has a 1 m lever arm along world Y.
        np.testing.assert_allclose(wrenches.numpy(), [[0, 0, 0, 0, 0, 0], [2, 0, 4, 4, 0, 0]], atol=1e-6)


def test_mpm_collider_velocity_excludes_previous_wrench():
    """Remove lagged feedback once, using world-space inverse inertia and global body indices."""
    with wp.ScopedDevice("cpu"):
        rotation = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), math.pi / 2)
        body_q = wp.array([wp.transform(wp.vec3(1.0, 2.0, 3.0), rotation), wp.transform_identity()], dtype=wp.transform)
        body_qd = wp.array([[2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13]], dtype=wp.spatial_vector)
        body_indices = wp.array([1, 0], dtype=int)
        previous_wrench = wp.array([[0, 0, 0, 0, 0, 0], [8, 0, 0, 0, 6, 0]], dtype=wp.spatial_vector)
        inverse_mass = wp.array([1.0, 0.25], dtype=float)
        inverse_inertia = wp.array([np.eye(3), np.diag([1.0, 2.0, 3.0])], dtype=wp.mat33)
        collider_q = wp.zeros(2, dtype=wp.transform)
        collider_qd = wp.zeros(2, dtype=wp.spatial_vector)
        wp.launch(
            prepare_mpm_collider_state,
            dim=2,
            inputs=[0.5, body_indices, body_q, body_qd, previous_wrench, inverse_mass, inverse_inertia],
            outputs=[collider_q, collider_qd],
        )
        # Rotation maps body X inertia to world Y: delta_v_x = 1 m/s, delta_omega_y = 3 rad/s.
        np.testing.assert_allclose(collider_qd.numpy(), [[8, 9, 10, 11, 12, 13], [1, 3, 4, 5, 3, 7]], atol=1e-6)
