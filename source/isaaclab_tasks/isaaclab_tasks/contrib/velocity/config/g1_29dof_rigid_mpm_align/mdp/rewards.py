# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MPM gait formulas with contact supplied by the rigid solver."""

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

from isaaclab_tasks.contrib.velocity.config.g1_29dof_rigid.mdp.observations import foot_contact


def feet_slide(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Sum world-XY foot speeds during contact, using a 5 N force-norm threshold."""
    contact = foot_contact(env, sensor_cfg, threshold=5.0)
    velocity = env.scene[asset_cfg.name].data.body_lin_vel_w.torch[:, asset_cfg.body_ids, :2]
    return (torch.linalg.vector_norm(velocity, dim=-1) * contact).sum(dim=-1)


def no_fly(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    velocity_threshold: float = 1.0,
) -> torch.Tensor:
    """Penalize flight below the reference horizontal command-speed threshold."""
    contact = foot_contact(env, sensor_cfg, threshold=5.0)
    command_speed = torch.linalg.vector_norm(env.command_manager.get_command(command_name)[:, :2], dim=-1)
    return (contact.sum(dim=-1) < 0.5).float() * (command_speed < velocity_threshold)


def feet_air_time_positive_biped(
    env: ManagerBasedRLEnv,
    command_name: str,
    threshold: float,
    sensor_cfg: SceneEntityCfg,
    velocity_threshold: float = 0.05,
) -> torch.Tensor:
    """Use the MPM single-stance formula and its separate linear/angular command gates."""
    rigid = env.scene[sensor_cfg.name]
    air_time = rigid.data.current_air_time.torch[:, sensor_cfg.body_ids]
    contact_time = rigid.data.current_contact_time.torch[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = in_contact.sum(dim=-1) == 1
    reward = torch.where(single_stance[:, None], in_mode_time, 0.0).min(dim=-1).values.clamp(max=threshold)
    command = env.command_manager.get_command(command_name)
    commanded = (command[:, :2].norm(dim=-1) > velocity_threshold) | (command[:, 2].abs() > velocity_threshold)
    return reward * commanded
