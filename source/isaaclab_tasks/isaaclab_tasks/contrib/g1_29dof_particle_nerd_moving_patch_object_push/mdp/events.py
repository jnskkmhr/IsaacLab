# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reset the object relative to the robot and retain fixed finger targets."""

import torch
import warp as wp

from isaaclab.utils.math import quat_apply, yaw_quat


def reset_object(env, env_ids, object_name, robot_name, forward_range, lateral_range, half_height):
    robot, obj = env.scene[robot_name], env.scene[object_name]
    state = obj.data.default_root_state.torch[env_ids].clone()
    offset = torch.zeros(state.shape[0], 3, device=env.device)
    offset[:, 0].uniform_(*forward_range)
    offset[:, 1].uniform_(*lateral_range)
    q = yaw_quat(robot.data.root_quat_w.torch[env_ids])
    state[:, :3] = robot.data.root_pos_w.torch[env_ids] + quat_apply(q, offset)
    # Terrain chart returns the undisturbed sand surface, not the floor underneath.
    points = wp.from_torch(state[:, :3].contiguous(), dtype=wp.vec3)
    env.scene.terrain.background_mesh.sample_surface_heights(points)
    surface = wp.to_torch(points)[:, 2]
    state[:, 2] = surface + half_height + 0.01
    state[:, 3:7] = q
    state[:, 7:] = 0
    obj.write_root_pose_to_sim(state[:, :7], env_ids=env_ids)
    obj.write_root_velocity_to_sim(state[:, 7:], env_ids=env_ids)


def hold_hands(env, env_ids, asset_cfg):
    robot = env.scene[asset_cfg.name]
    robot.set_joint_position_target(
        robot.data.default_joint_pos.torch[env_ids][:, asset_cfg.joint_ids],
        joint_ids=asset_cfg.joint_ids,
        env_ids=env_ids,
    )
