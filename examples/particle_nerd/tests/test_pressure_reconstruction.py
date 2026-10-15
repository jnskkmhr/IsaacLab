# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Check boundary traction normalization and independently known weight transmission."""

import sys
from pathlib import Path

import numpy as np
from scipy.integrate import quad

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pressure_reconstruction import planar_kernel_weight, reconstruct_pressure_reactions


def test_uniform_pressure_integrates_to_pressure_times_wall_area():
    # Unit pressure over a unit-area wall must transmit one Newton, independently of support width.
    h = 0.08
    integral = quad(lambda d: float(planar_kernel_weight(d, h)), 0, h)[0]
    np.testing.assert_allclose(integral, 1, atol=1e-12)
    np.testing.assert_allclose(planar_kernel_weight(np.array([h, 2 * h]), h), 0, atol=1e-12)


def test_pressure_column_transmits_weight_without_reference_force_input():
    result = reconstruct_pressure_reactions(
        np.array([[0.25, 0, 0.01], [0.25, 0, 0.11]]),
        np.zeros((2, 3)),
        np.zeros((2, 3)),
        np.array([1.0, 2.0]),
        np.array([[0, 0, -0.1, 0, 0, 0, 1.0]]),
        np.array([[1, 1, 0.2]]),
        np.array([0.0]),
        dt=0.01,
        support_radius=0.15,
        boundary_radius=0.02,
        regularization=1e-9,
    )
    assert result.converged
    assert np.min(result.pressure) >= -1e-6
    np.testing.assert_allclose(result.body_wrench[0], [0, 0, -29.43, 0, 7.3575, 0], atol=1e-4)
    assert result.relative_residual < 1e-5
