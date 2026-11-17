# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Deterministic arm reset without modifying the teacher's default joint offsets."""

import torch

from isaaclab.managers import SceneEntityCfg


def reset_joint_positions(env, env_ids, positions: tuple[float, ...], asset_cfg: SceneEntityCfg):
    """Set absolute positions in the configured joint order and zero their velocities."""
    asset = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    elif isinstance(env_ids, slice):
        env_ids = torch.arange(env.num_envs, device=env.device)[env_ids]
    position = torch.tensor(positions, device=env.device).expand(len(env_ids), -1)
    if position.shape[1] != len(asset_cfg.joint_ids):
        raise ValueError("positions must match the selected joints.")
    asset.write_joint_state_to_sim(position, torch.zeros_like(position), joint_ids=asset_cfg.joint_ids, env_ids=env_ids)
