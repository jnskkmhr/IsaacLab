# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Failure conditions shared by object-pushing tasks."""

import math

import torch

from isaaclab.utils.math import quat_apply


def box_tilted(env, limit=0.6, object_name: str = "box"):
    up = torch.zeros((env.num_envs, 3), device=env.device)
    up[:, 2] = 1
    normal = quat_apply(env.scene[object_name].data.root_quat_w.torch, up)
    return normal[:, 2] < math.cos(limit)


def invalid_root_state(env, object_name: str = "box"):
    """Terminate when either the robot or the box has a nonfinite root state."""
    robot = env.scene["robot"].data
    box = env.scene[object_name].data
    return ~(
        torch.isfinite(robot.root_link_pose_w.torch).all(dim=-1)
        & torch.isfinite(robot.root_com_vel_w.torch).all(dim=-1)
        & torch.isfinite(box.root_link_pose_w.torch).all(dim=-1)
        & torch.isfinite(box.root_com_vel_w.torch).all(dim=-1)
    )
