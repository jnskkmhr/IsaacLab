# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation layout and material normalization from reference run 4blqnxbi."""

import torch

from isaaclab.envs.mdp import root_quat_w
from isaaclab.managers import SceneEntityCfg


def root_quat_wxyz(env, asset_cfg=SceneEntityCfg("robot"), make_quat_unique=False):
    """Keep the reference policy's scalar-first quaternion layout at the observation boundary."""
    return root_quat_w(env, asset_cfg=asset_cfg, make_quat_unique=make_quat_unique)[..., [3, 0, 1, 2]]


def velocity_commands_wxyz(env, command_name):
    """Return linear/yaw commands followed by the target heading quaternion in wxyz order."""
    command = env.command_manager.get_command(command_name)
    return torch.cat((command[:, :3], command[:, [6, 3, 4, 5]]), dim=-1)


def terrain_material_parameters_hybrid(
    env,
    asset_cfg=SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
    rigid_contact_sensor_cfg=SceneEntityCfg("contact_forces"),
    soft_contact_sensor_name="physics_callback",
):
    """Normalized RFT stiffness, one scalar per environment; rigid ground has value one."""
    soft = env.action_manager.get_term(soft_contact_sensor_name).contact_solver
    active = soft.data.is_sensor_active.any(dim=-1)
    density = torch.where(active, soft.terrain_density, 3000.0)
    mu = torch.where(active, soft.terrain_stiffness, 0.9)
    xi = density * 9.81 * (894.0 * mu**3 - 386.0 * mu**2 + 89.0 * mu)
    xi_max = 3000.0 * 9.81 * (894.0 * 0.9**3 - 386.0 * 0.9**2 + 89.0 * 0.9)
    return (xi / xi_max).view(-1, 1)
