# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward semantics retained from the reference PhysX run."""

import torch

from isaaclab.utils.math import euler_xyz_from_quat


def action_rate_l2(env, joint_idx):
    """Squared action changes in policy action order, independent of backend joint order."""
    return ((env.action_manager.action[:, joint_idx] - env.action_manager.prev_action[:, joint_idx]) ** 2).sum(-1)


def reward_feet_pitch_contact(env, sensor_cfg, asset_cfg):
    """Squared foot pitch at touchdown, matching the reference rigid branch."""
    sensor = env.scene[sensor_cfg.name]
    first = sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    quat = env.scene[asset_cfg.name].data.body_quat_w.torch[:, asset_cfg.body_ids]
    pitch = euler_xyz_from_quat(quat.reshape(-1, 4))[1].reshape(env.num_envs, -1)
    pitch = (pitch + torch.pi) % (2 * torch.pi) - torch.pi
    return (pitch.square() * first).sum(-1)


def feet_slide(env, sensor_cfg, asset_cfg, contact_threshold=5.0):
    """Sum horizontal foot speeds while upward contact force exceeds the threshold [N]."""
    contact = env.scene[sensor_cfg.name].data.net_forces_w.torch[:, sensor_cfg.body_ids, 2] > contact_threshold
    velocity = env.scene[asset_cfg.name].data.body_lin_vel_w.torch[:, asset_cfg.body_ids, :2]
    return (velocity.norm(dim=-1) * contact).sum(dim=1)


def no_fly(env, sensor_cfg, contact_threshold=5.0, command_name="base_velocity", velocity_threshold=1.0):
    """Penalize no feet in contact below the reference horizontal command speed."""
    contact = env.scene[sensor_cfg.name].data.net_forces_w.torch[:, sensor_cfg.body_ids, 2] > contact_threshold
    speed = env.command_manager.get_command(command_name)[:, :2].norm(dim=1)
    return (contact.sum(dim=-1) == 0).float() * (speed < velocity_threshold)
