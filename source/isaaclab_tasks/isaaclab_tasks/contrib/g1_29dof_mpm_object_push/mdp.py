# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task-local observations, resets, rewards and particle-coverage guards."""

import math

import torch
import warp as wp

from isaaclab.utils.math import quat_apply, quat_apply_inverse, subtract_frame_transforms, yaw_quat


def object_pose_b(env):
    """Object pose relative to robot root; simulation stand-in for perception."""
    robot, box = env.scene["robot"].data, env.scene["box"].data
    position, quaternion = subtract_frame_transforms(
        robot.root_pos_w.torch, robot.root_quat_w.torch, box.root_pos_w.torch, box.root_quat_w.torch
    )
    return torch.cat((position, quaternion), dim=-1)


def hand_pose_b(env):
    command = env.command_manager.get_term("hand_pose")
    position, quaternion = command.pose_b(*command.hand_pose_w())
    return torch.cat((position, quaternion), dim=-1).flatten(1)


def box_velocity_b(env):
    """Privileged box linear/angular velocity expressed in the robot base frame."""
    box, robot = env.scene["box"].data, env.scene["robot"].data
    return torch.cat(
        (
            quat_apply_inverse(robot.root_quat_w.torch, box.root_lin_vel_w.torch),
            quat_apply_inverse(robot.root_quat_w.torch, box.root_ang_vel_w.torch),
        ),
        dim=-1,
    )


def reset_box(env, env_ids, distance_range=(0.85, 0.95)):
    """Place the box ahead of the newly reset robot, above the flat initial sand surface."""
    box, robot = env.scene["box"], env.scene["robot"]
    ids = slice(None) if env_ids is None else env_ids
    position = robot.data.root_pos_w.torch[ids]
    orientation = yaw_quat(robot.data.root_quat_w.torch[ids])
    offset = torch.zeros_like(position)
    offset[:, 0].uniform_(*distance_range)
    pose = torch.cat((position + quat_apply(orientation, offset), orientation), dim=-1)
    pose[:, 2] = env.scene.env_origins[ids, 2] + env.cfg.scene.box.spawn.size[2] / 2 + 0.01
    box.write_root_pose_to_sim_index(root_pose=pose, env_ids=ids)
    box.write_root_velocity_to_sim_index(root_velocity=torch.zeros((len(pose), 6), device=env.device), env_ids=ids)


def hand_position_tracking(env, std=0.25):
    error, _ = env.command_manager.get_term("hand_pose").errors()
    return torch.exp(-error.square() / std**2).mean(dim=-1)


def hand_orientation_tracking(env, std=0.7):
    _, error = env.command_manager.get_term("hand_pose").errors()
    return torch.exp(-error.square() / std**2).mean(dim=-1)


def box_velocity_tracking(env, std=0.25, hand_distance=0.15):
    """Reward object motion weighted by proximity of both hands to their targets."""
    velocity = quat_apply_inverse(
        yaw_quat(env.scene["robot"].data.root_quat_w.torch), env.scene["box"].data.root_lin_vel_w.torch
    )
    target = env.command_manager.get_command("base_velocity")[:, :2]
    error = (velocity[:, :2] - target).square().sum(dim=-1)
    hand_error, _ = env.command_manager.get_term("hand_pose").errors()
    proximity = torch.exp(-hand_error.square().sum(dim=-1) / (2 * hand_distance**2))
    return torch.exp(-error / std**2) * proximity


def box_tilted(env, limit=0.6):
    up = torch.zeros((env.num_envs, 3), device=env.device)
    up[:, 2] = 1
    normal = quat_apply(env.scene["box"].data.root_quat_w.torch, up)
    return normal[:, 2] < math.cos(limit)


def box_outside_patch(env, margin=0.3):
    """Terminate before any box corner reaches recycled/boundary particles.

    Uses the world-aligned particle patch, not the robot's rotating local axes.
    The margin reserves room for patch quantization and motion before the next check.
    """
    box = env.scene["box"].data
    center = wp.to_torch(env.moving_patch_particle.patch_centers)
    half_size = torch.tensor(env.cfg.scene.box.spawn.size, device=env.device) / 2
    signs = torch.tensor([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], device=env.device)
    offsets = (signs * half_size).expand(env.num_envs, -1, -1)
    corners = (
        quat_apply(box.root_quat_w.torch[:, None, :].expand(-1, 8, -1), offsets) + box.root_pos_w.torch[:, None, :]
    )
    allowed = (
        torch.tensor(env.cfg.scene.terrain.moving_patch_terrain.simulated_terrain_size, device=env.device) / 2 - margin
    )
    return ((corners[:, :, :2] - center[:, None, :]).abs() > allowed).any(dim=(1, 2))
