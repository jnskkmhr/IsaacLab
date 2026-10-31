# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Target geometry and direction reversal for the new box-motion tests."""

import numpy as np

from particle_nerd.data_collection.trajectories import PATHS, drag_offset, drag_target_offset, initial_xy


def test_target_paths_have_requested_geometry_and_reverse_traversal():
    phases = np.linspace(0.0, 1.0, 1001)
    for path in PATHS:
        xy = np.array([drag_offset(path, t)[:2] for t in phases]) + initial_xy(path, 1)
        reverse = np.array([drag_offset(path, t, -1)[:2] for t in phases]) + initial_xy(path, -1)
        np.testing.assert_allclose(reverse, xy[::-1], atol=1e-12)
        assert np.isfinite(xy).all()
        assert np.ptp(xy[:, 0]) > 0.75
        assert np.ptp(xy[:, 1]) > 0.51
        if path == "wide-circle":
            np.testing.assert_allclose(np.linalg.norm(xy, axis=1), 0.38, atol=1e-12)
        elif path == "s-curve":
            np.testing.assert_allclose(xy[[0, -1]], [[-0.38, 0.0], [0.38, 0.0]], atol=1e-12)
        else:
            np.testing.assert_allclose(xy[[0, 500, -1]], 0.0, atol=1e-12)
        # Smooth start/end avoid an unintended step in target velocity.
        for t in (0.0, 1.0):
            delta = drag_offset(path, t + 1e-5) - drag_offset(path, t - 1e-5)
            assert np.linalg.norm(delta) / (2e-5) < 1e-6


def test_center_lead_in_is_continuous_with_zero_endpoint_velocity():
    for path in PATHS:
        for direction in (-1, 1):
            np.testing.assert_allclose(drag_target_offset(path, 0.0, 4.5, direction), 0.0)
            start = np.array([*initial_xy(path, direction), 0.0])
            np.testing.assert_allclose(drag_target_offset(path, 0.75, 4.5, direction), start)
            for t in (0.0, 0.75, 4.5):
                left = drag_target_offset(path, t - 1e-5, 4.5, direction)
                right = drag_target_offset(path, t + 1e-5, 4.5, direction)
                assert np.linalg.norm(right - left) / 2e-5 < 1e-6
