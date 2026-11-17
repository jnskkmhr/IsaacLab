# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Actuator target randomization for fixed-base manipulation."""

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.string import resolve_matching_names_values


def randomize_joint_position_targets(
    env,
    env_ids: torch.Tensor | slice | None,
    position_ranges: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Sample absolute joint targets in radians without changing joint state.

    Only joints matched by ``position_ranges`` are commanded. Targets are clamped
    to the selected environments' soft position limits and held until overwritten.
    """
    asset = env.scene[asset_cfg.name]
    joint_ids, _, ranges = resolve_matching_names_values(position_ranges, asset.joint_names)
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    elif isinstance(env_ids, slice):
        env_ids = torch.arange(env.num_envs, device=env.device)[env_ids]
    bounds = torch.tensor(ranges, device=env.device)
    targets = bounds[:, 0] + torch.rand(len(env_ids), len(joint_ids), device=env.device) * (bounds[:, 1] - bounds[:, 0])
    limits = asset.data.soft_joint_pos_limits.torch[env_ids][:, joint_ids]
    targets = targets.clamp(min=limits[..., 0], max=limits[..., 1])
    asset.actuators.target_command.set_position_index(value=targets, joint_ids=joint_ids, env_ids=env_ids)
