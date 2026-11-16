# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Slow crawl targets for a free-base quadruped; only leg joints are actuated."""

import math

from isaaclab.utils.math import quat_apply_inverse


class CrawlController:
    """Shift the trunk into the support triangle, then swing one foot at a time.

    Damped least-squares IK and PD drives track these foot positions in the base
    frame. No base pose, velocity, force, or torque is written by this controller.
    """

    def __init__(self, env):
        self.robot = env.scene["robot"]
        self.ids, _ = self.robot.find_bodies(["FL_foot", "FR_foot", "RL_foot", "RR_foot"], preserve_order=True)
        root = self.robot.data.root_pose_w.torch
        feet = self.robot.data.body_pose_w.torch[:, self.ids, :3]
        self.nominal = quat_apply_inverse(root[:, None, 3:].expand(-1, 4, -1), feet - root[:, None, :3]).clone()
        # Slightly bent knees give leg IK room to accommodate sinking.
        self.nominal[:, :, 2] = -0.30
        self.order = (3, 1, 2, 0)
        self.stride = 0.14
        self.period = 1.0
        self.device = env.device

    def command(self, t):
        desired = self.nominal.clone()
        if t <= 1.0:
            return desired.reshape(1, 12)
        phase = (t - 1.0) / self.period
        segment = int(phase)
        u = phase - segment
        leg = self.order[segment % 4]
        for foot in range(4):
            local = (phase - self.order.index(foot)) % 4.0
            if local < 1.0:
                # Lift only after shifting weight; hold four contacts between swings.
                swing = min(1.0, max(0.0, (local - 0.2) / 0.6))
                smooth = swing * swing * (3.0 - 2.0 * swing)
                desired[:, foot, 0] += self.stride * (-0.5 + smooth)
                desired[:, foot, 2] += 0.10 * math.sin(math.pi * swing)
            else:
                desired[:, foot, 0] += self.stride * (0.5 - (local - 1.0) / 3.0)
        stance = [i for i in range(4) if i != leg]
        shift = self.nominal[:, stance, :2].mean(1) * 0.7
        weight = min(1.0, u / 0.15, (1.0 - u) / 0.15)
        desired[:, :, :2] -= shift[:, None, :] * weight
        gravity = self.robot.data.projected_gravity_b.torch
        desired[:, :, 2] += 0.35 * (
            gravity[:, 0, None] * self.nominal[:, :, 0] + gravity[:, 1, None] * self.nominal[:, :, 1]
        )
        return desired.reshape(1, 12)
