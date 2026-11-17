# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward semantics retained from the reference PhysX run."""

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import euler_xyz_from_quat


def action_rate_l2(env, joint_idx):
    """Squared action changes in policy action order, independent of backend joint order."""
    return ((env.action_manager.action[:, joint_idx] - env.action_manager.prev_action[:, joint_idx]) ** 2).sum(-1)


def reward_feet_pitch_contact_hybrid(
    env,
    rigid_contact_sensor_cfg: SceneEntityCfg,
    soft_contact_sensor_name="physics_callback",
    asset_cfg=SceneEntityCfg("robot"),
):
    """Penalize foot pitch at touchdown, as in reference run 4blqnxbi."""
    rigid = env.scene.sensors[rigid_contact_sensor_cfg.name]
    soft = env.action_manager.get_term(soft_contact_sensor_name).contact_solver
    rigid_first = rigid.compute_first_contact(env.step_dt)[:, rigid_contact_sensor_cfg.body_ids]
    first = torch.where(soft.data.is_sensor_active, soft.compute_first_contact(env.step_dt), rigid_first)
    quat = env.scene[asset_cfg.name].data.body_quat_w.torch[:, asset_cfg.body_ids]
    pitch = euler_xyz_from_quat(quat.reshape(-1, 4))[1].reshape(env.num_envs, -1)
    pitch = (pitch + torch.pi) % (2 * torch.pi) - torch.pi
    return (pitch.square() * first).sum(-1)
