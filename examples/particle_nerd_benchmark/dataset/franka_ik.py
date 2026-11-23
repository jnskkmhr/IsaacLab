# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Franka Panda forward kinematics (modified DH) and a numerical IK used to pick each episode's start pose."""

import numpy as np
from scipy.optimize import least_squares

DH = (  # a_{i-1}, d_i, alpha_{i-1}
    (0.0, 0.333, 0.0),
    (0.0, 0.0, -np.pi / 2),
    (0.0, 0.316, np.pi / 2),
    (0.0825, 0.0, np.pi / 2),
    (-0.0825, 0.384, -np.pi / 2),
    (0.0, 0.0, np.pi / 2),
    (0.088, 0.0, np.pi / 2),
)
FLANGE_D = 0.107
LOW = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
HIGH = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
TOOL_OFFSET = 0.107
"""Distance from the hand origin to the commanded end-effector point along the hand z axis."""


def _transform(a, d, alpha, theta):
    ca, sa, ct, st = np.cos(alpha), np.sin(alpha), np.cos(theta), np.sin(theta)
    return np.array(
        [[ct, -st, 0.0, a], [st * ca, ct * ca, -sa, -d * sa], [st * sa, ct * sa, ca, d * ca], [0, 0, 0, 1.0]]
    )


def hand_pose(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the hand-frame rotation and the end-effector point (hand + 0.107 m along z) in the base frame."""
    t = np.eye(4)
    for (a, d, alpha), theta in zip(DH, q, strict=True):
        t = t @ _transform(a, d, alpha, theta)
    t = t @ _transform(0.0, FLANGE_D, 0.0, 0.0)
    c, s = np.cos(-np.pi / 4), np.sin(-np.pi / 4)
    rot = t[:3, :3] @ np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return rot, t[:3, 3] + rot[:, 2] * TOOL_OFFSET


def quat_wxyz_to_matrix(q) -> np.ndarray:
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def solve(target_point, target_quat_wxyz, seed, prior_weight: float = 0.0) -> tuple[np.ndarray, float, float]:
    """Joint angles (7) that put the end-effector point at ``target_point`` with the given hand orientation.

    ``prior_weight`` adds a penalty on the distance to ``seed``, which keeps consecutive solutions of the redundant
    arm close together. Returns the joints, the position error [m] and the orientation error [Frobenius].
    """
    target_rot = quat_wxyz_to_matrix(target_quat_wxyz)

    def residual(q):
        rot, point = hand_pose(q)
        return np.concatenate(
            (point - target_point, 0.3 * (rot - target_rot).ravel(), prior_weight * (q - np.asarray(seed)))
        )

    result = least_squares(residual, np.asarray(seed, dtype=float), bounds=(LOW, HIGH))
    rot, point = hand_pose(result.x)
    return result.x, float(np.linalg.norm(point - target_point)), float(np.linalg.norm(rot - target_rot))
