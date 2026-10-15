# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Independent load transmission and inelastic-impact cases for the network solve."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from network_contact_reconstruction import project_contact_network


def test_material_column_transmits_box_weight_to_fixed_support():
    # The predictor supplies stationary material. The contact correction must
    # transmit the additional box weight through two points to fixed ground.
    result = project_contact_network(
        np.array([[0, 0, 0.0], [0, 0, 0.1]]),
        np.zeros((2, 3)),
        np.array([1.0, 1.0]),
        np.array([0.05, 0.05]),
        np.array([[0, 0, -0.1, 0, 0, 0, 1.0], [0, 0, 0.2, 0, 0, 0, 1.0]]),
        np.zeros((2, 6)),
        np.array([0.0, 2.0]),
        np.array([np.eye(3), np.eye(3)]),
        np.array([[1.0, 1.0, 0.2], [0.2, 0.2, 0.2]]),
        1,
        dt=0.01,
    )
    assert result.converged
    np.testing.assert_allclose(result.body_wrench[1, :3], [0, 0, 19.62], atol=1e-5)
    np.testing.assert_allclose(result.body_wrench[0, :3], [0, 0, -19.62], atol=1e-5)
    np.testing.assert_allclose(result.body_velocity, 0, atol=1e-7)
    np.testing.assert_allclose(result.particle_velocity, 0, atol=1e-7)
    assert result.momentum_correction_residual < 1e-6


def test_particle_pair_impact_conserves_momentum_without_body_contact():
    result = project_contact_network(
        np.array([[-0.05, 0, 0], [0.05, 0, 0]]),
        np.array([[1.0, 0, 0], [-1.0, 0, 0]]),
        np.array([1.0, 3.0]),
        np.array([0.05, 0.05]),
        np.array([[10.0, 10.0, 10.0, 0, 0, 0, 1.0]]),
        np.zeros((1, 6)),
        np.array([2.0]),
        np.array([np.eye(3)]),
        np.array([[1.0, 1.0, 1.0]]),
        0,
        dt=0.01,
        gravity=(0, 0, 0),
    )
    assert result.converged
    np.testing.assert_allclose(result.particle_velocity, [[-0.5, 0, 0], [-0.5, 0, 0]], atol=1e-6)
    np.testing.assert_allclose(result.body_wrench, 0, atol=1e-8)
    assert result.momentum_correction_residual < 1e-6
