# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Independent mechanics cases: free fall, transmitted support, and friction/torque.

These protect acceleration signs, internal action-reaction, and torque origins.
There was no pre-existing reconstruction implementation or coverage for these contracts.
"""

import numpy as np
import pytest

from particle_nerd.force_estimation.reconstruction import box_contacts, reconstruct_reactions


def test_free_fall_has_no_body_reaction():
    position = np.array([[0.0, 0.0, 2.0]])
    result = reconstruct_reactions(
        position,
        np.zeros((1, 3)),
        np.array([[0.0, 0.0, -0.0981]]),
        np.array([2.0]),
        np.array([[0, 0, -0.1, 0, 0, 0, 1.0]]),
        np.array([[1, 1, 0.2]]),
        np.array([0.5]),
        dt=0.01,
    )
    np.testing.assert_allclose(result.body_wrench, 0, atol=1e-9)
    np.testing.assert_allclose(result.particle_residual, 0, atol=1e-9)


@pytest.mark.parametrize("optimizer,iterations", [("trf", 100), ("osqp", 10000)])
def test_two_particle_column_transmits_total_weight_to_ground(optimizer, iterations):
    # Only the lower particle contacts ground. Its pair force must carry the upper mass.
    position = np.array([[0.25, 0, 0.01], [0.25, 0, 0.11]])
    result = reconstruct_reactions(
        position,
        np.zeros((2, 3)),
        np.zeros((2, 3)),
        np.array([1.0, 2.0]),
        np.array([[0, 0, -0.1, 0, 0, 0, 1.0]]),
        np.array([[1, 1, 0.2]]),
        np.array([0.0]),
        dt=0.01,
        neighbor_radius=0.15,
        contact_distance=0.02,
        regularization=1e-9,
        optimizer=optimizer,
        max_iterations=iterations,
    )
    assert result.converged
    np.testing.assert_allclose(result.body_wrench[0], [0, 0, -29.43, 0, 7.3575, 0], atol=1e-4)
    assert result.relative_residual < 1e-5


def test_sliding_particle_reaction_has_opposite_friction_and_correct_moment():
    result = reconstruct_reactions(
        np.array([[0.2, 0, 0.01]]),
        np.array([[1.0, 0, 0]]),
        np.array([[0.95, 0, 0]]),
        np.array([2.0]),
        np.array([[0, 0, -0.1, 0, 0, 0, 1.0]]),
        np.array([[1, 1, 0.2]]),
        np.array([0.6]),
        dt=0.01,
        contact_distance=0.02,
        regularization=1e-9,
    )
    np.testing.assert_allclose(result.body_wrench[0], [10, 0, -19.62, 0, 4.924, 0], atol=1e-4)


def test_rotated_box_normal_and_interior_distance():
    # 90 degrees around y maps the thin local z axis to world x.
    pose = np.array([1, 0, 0, 0, 2**-0.5, 0, 2**-0.5])
    distance, normal, point = box_contacts(np.array([[1.15, 0, 0], [1.08, 0, 0]]), pose, np.array([2, 2, 0.2]))
    np.testing.assert_allclose(distance, [0.05, -0.02], atol=1e-12)
    np.testing.assert_allclose(normal, [[1, 0, 0], [1, 0, 0]], atol=1e-12)
    np.testing.assert_allclose(point, [[1.1, 0, 0], [1.1, 0, 0]], atol=1e-12)


def test_box_load_transmits_through_material_column_to_ground():
    # A separately estimated box load presses on the upper material particle.
    # Ground must receive both particle weights plus this box load.
    result = reconstruct_reactions(
        np.array([[0, 0, 0.01], [0, 0, 0.11]]),
        np.zeros((2, 3)),
        np.zeros((2, 3)),
        np.array([1.0, 2.0]),
        np.array([[0, 0, -0.1, 0, 0, 0, 1.0]]),
        np.array([[1.0, 1.0, 0.2]]),
        np.array([0.0]),
        dt=0.01,
        neighbor_radius=0.15,
        contact_distance=0.02,
        regularization=1e-9,
        optimizer="osqp",
        max_iterations=10000,
        known_particle_force=np.array([[0, 0, 0], [0, 0, -20.0]]),
    )
    assert result.converged
    np.testing.assert_allclose(result.body_wrench[0, :3], [0, 0, -49.43], atol=1e-4)
    np.testing.assert_allclose(result.particle_residual, 0, atol=1e-4)
