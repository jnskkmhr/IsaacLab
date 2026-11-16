# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Analytic collider trajectories. Coordinates are world aligned, in metres."""

import math

import numpy as np

from .terrain import surface_height

MOTIONS = ("horizontal", "vertical", "circle", "rolling", "projectile")
TERRAINS = ("flat", "ascend", "descend", "wave")
SETTLE_SECONDS = 1.0


def pose_at(t, motion, terrain):
    """Hold above the bed, enter smoothly, then execute the prescribed path."""
    u = np.clip((t - SETTLE_SECONDS) / 1.0, 0.0, 1.0)
    blend = u * u * (3 - 2 * u)
    travel = np.clip((t - SETTLE_SECONDS - 1.0) / 4.0, 0.0, 1.0)
    x, y = -0.65, 0.0
    radius = 0.10
    if motion == "horizontal" or motion == "rolling":
        x = -0.65 + 1.30 * travel
    elif motion == "circle":
        angle = 2 * math.pi * travel
        x, y = -0.35 * math.cos(angle), 0.35 * math.sin(angle)
    elif motion == "vertical":
        x = 0.0
    height = float(surface_height(x, terrain))
    # Nominal initial surface; actual settled surface is measured in the dataset.
    target = height + (0.035 if motion == "rolling" else -0.025)
    if motion == "vertical":
        target = height + 0.035 - 0.12 * travel
    z = 0.30 + blend * (target - 0.30)
    quat = np.array([0.0, 0.0, 0.0, 1.0])
    if motion == "rolling":
        angle = (x + 0.65) / radius
        quat = np.array([0.0, math.sin(angle / 2), 0.0, math.cos(angle / 2)])
    return np.r_[x, y, z, quat]


def object_command(t, motion, terrain, dt):
    pose = pose_at(t, motion, terrain)
    future = pose_at(t + dt, motion, terrain)
    velocity = (future[:3] - pose[:3]) / dt
    omega = np.zeros(3)
    if motion == "rolling":
        omega[1] = velocity[0] / 0.10
    return np.r_[pose, velocity, omega].astype(np.float32)
