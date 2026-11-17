# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Object-pushing rewards."""

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse, yaw_quat


def body_lin_vel_z_l2(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Sum squared selected-body COM velocities along each body's local z axis, in (m/s)^2."""
    asset = env.scene[asset_cfg.name]
    velocity = quat_apply_inverse(
        asset.data.body_link_quat_w.torch[:, asset_cfg.body_ids],
        asset.data.body_lin_vel_w.torch[:, asset_cfg.body_ids],
    )
    return velocity[..., 2].square().sum(dim=1)


def body_ang_vel_xy_l2(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Sum squared selected-body angular velocities about each body's local x/y axes, in (rad/s)^2."""
    asset = env.scene[asset_cfg.name]
    velocity = quat_apply_inverse(
        asset.data.body_link_quat_w.torch[:, asset_cfg.body_ids],
        asset.data.body_ang_vel_w.torch[:, asset_cfg.body_ids],
    )
    return velocity[..., :2].square().sum(dim=(1, 2))


def body_flat_orientation_l2(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Sum squared horizontal gravity projections in selected link frames, independent of gravity magnitude."""
    asset = env.scene[asset_cfg.name]
    quaternion = asset.data.body_link_quat_w.torch[:, asset_cfg.body_ids]
    gravity = torch.nn.functional.normalize(asset.data.GRAVITY_VEC_W.torch, dim=-1)
    projected_gravity = quat_apply_inverse(quaternion, gravity[:, None, :].expand(-1, quaternion.shape[1], -1))
    return projected_gravity[..., :2].square().sum(dim=(1, 2))


def _hand_errors(env, command_names):
    errors = [env.command_manager.get_term(name).errors() for name in command_names]
    return tuple(torch.cat([error[i].reshape(env.num_envs, -1) for error in errors], dim=-1) for i in range(2))


def hand_position_tracking(env, std=0.25, command_names=("hand_pose",)):
    error, _ = _hand_errors(env, command_names)
    return torch.exp(-error.square() / std**2).mean(dim=-1)


def hand_orientation_tracking(env, std=0.7, command_names=("hand_pose",)):
    _, error = _hand_errors(env, command_names)
    return torch.exp(-error.square() / std**2).mean(dim=-1)


def box_velocity_tracking(
    env, std=0.25, hand_distance=0.15, command_name="base_velocity", command_names=("hand_pose",), object_name="box"
):
    """Reward object motion weighted by proximity of both hands to their targets."""
    velocity = quat_apply_inverse(
        yaw_quat(env.scene["robot"].data.root_quat_w.torch), env.scene[object_name].data.root_lin_vel_w.torch
    )
    target = env.command_manager.get_command(command_name)[:, :2]
    error = (velocity[:, :2] - target).square().sum(dim=-1)
    hand_error, _ = _hand_errors(env, command_names)
    proximity = torch.exp(-hand_error.square().sum(dim=-1) / (2 * hand_distance**2))
    return torch.exp(-error / std**2) * proximity
