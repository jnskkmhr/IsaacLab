# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Smooth external-drag targets for frozen-model trajectory evaluation."""

import numpy as np

PATHS = ("wide-circle", "s-curve", "figure-eight")
RADIUS_X = 0.38
RADIUS_Y = 0.26


def initial_xy(path, direction):
    """Return the path start relative to the bed center."""
    if path == "wide-circle":
        return (-RADIUS_X, 0.0)
    if path == "s-curve":
        return (-direction * RADIUS_X, 0.0)
    if path == "figure-eight":
        return (0.0, 0.0)
    raise ValueError(path)


def drag_offset(path, phase, direction=1):
    """Target displacement from the path start; phase spans [0, 1].

    A quintic time parameter gives zero target speed and acceleration at both
    ends. The controller applies a bounded force; it never assigns body poses.
    """
    if direction not in (-1, 1):
        raise ValueError("Path direction must be -1 or +1")
    t = np.clip(phase, 0.0, 1.0)
    u = t**3 * (10.0 - 15.0 * t + 6.0 * t * t)
    angle = direction * 2.0 * np.pi * u
    if path == "wide-circle":
        x, y = RADIUS_X * (1.0 - np.cos(angle)), RADIUS_X * np.sin(angle)
    elif path == "s-curve":
        x, y = direction * 2.0 * RADIUS_X * u, RADIUS_Y * np.sin(angle)
    elif path == "figure-eight":
        x, y = RADIUS_X * np.sin(angle), RADIUS_Y * np.sin(2.0 * angle)
    else:
        raise ValueError(path)
    return np.array([x, y, -0.02 * u])


def drag_target_offset(path, elapsed, duration, direction=1):
    """Move from settled bed center to the path start, then traverse the path."""
    lead_in = 0.75
    if duration <= lead_in:
        raise ValueError("Dragging duration must exceed the 0.75 s lead-in")
    start = np.array([*initial_xy(path, direction), 0.0])
    if elapsed < lead_in:
        t = np.clip(elapsed / lead_in, 0.0, 1.0)
        return start * t**3 * (10.0 - 15.0 * t + 6.0 * t * t)
    return start + drag_offset(path, (elapsed - lead_in) / (duration - lead_in), direction)
