# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Object-pushing observations."""

import torch

from isaaclab.utils.math import quat_apply_inverse, subtract_frame_transforms


def object_pose_b(env, object_name: str = "box"):
    """Object pose relative to robot root; simulation stand-in for perception."""
    robot, box = env.scene["robot"].data, env.scene[object_name].data
    position, quaternion = subtract_frame_transforms(
        robot.root_pos_w.torch, robot.root_quat_w.torch, box.root_pos_w.torch, box.root_quat_w.torch
    )
    return torch.cat((position, quaternion), dim=-1)


def hand_pose_b(env, command_names=("hand_pose",)):
    """Concatenate actual palm poses in base coordinates, in command order."""
    poses = []
    for name in command_names:
        command = env.command_manager.get_term(name)
        position, quaternion = command.pose_b(*command.hand_pose_w())
        poses.append(torch.cat((position, quaternion), dim=-1).reshape(env.num_envs, -1))
    return torch.cat(poses, dim=-1)


def hand_pose_commands(env, command_names=("hand_pose",)):
    """Concatenate palm target poses in base coordinates, in command order."""
    return torch.cat([env.command_manager.get_command(name) for name in command_names], dim=-1)


def box_velocity_b(env, object_name: str = "box"):
    """Privileged box linear/angular velocity expressed in the robot base frame."""
    box, robot = env.scene[object_name].data, env.scene["robot"].data
    return torch.cat(
        (
            quat_apply_inverse(robot.root_quat_w.torch, box.root_lin_vel_w.torch),
            quat_apply_inverse(robot.root_quat_w.torch, box.root_ang_vel_w.torch),
        ),
        dim=-1,
    )
