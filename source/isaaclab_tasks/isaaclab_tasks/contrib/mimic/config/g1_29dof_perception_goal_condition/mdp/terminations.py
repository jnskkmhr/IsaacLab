# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Reference-based termination applies only to imitation episodes."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

import isaaclab_tasks.contrib.mimic.mdp as mimic_mdp

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def end_of_reference(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Task episodes continue beyond the end of the reference clip."""
    command = env.command_manager.get_term(command_name)
    return mimic_mdp.end_of_reference(env, command_name) & command.is_imitation


def bad_anchor_pos_z_only(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    """Reject excessive reference anchor height error in imitation episodes."""
    command = env.command_manager.get_term(command_name)
    return mimic_mdp.bad_anchor_pos_z_only(env, command_name, threshold) & command.is_imitation


def bad_anchor_ori(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, command_name: str, threshold: float
) -> torch.Tensor:
    """Reject excessive reference anchor tilt error in imitation episodes."""
    command = env.command_manager.get_term(command_name)
    return mimic_mdp.bad_anchor_ori(env, asset_cfg, command_name, threshold) & command.is_imitation


def bad_motion_body_pos_z_only(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, body_names: list[str]
) -> torch.Tensor:
    """Reject excessive reference body height error in imitation episodes."""
    command = env.command_manager.get_term(command_name)
    return mimic_mdp.bad_motion_body_pos_z_only(env, command_name, threshold, body_names) & command.is_imitation
