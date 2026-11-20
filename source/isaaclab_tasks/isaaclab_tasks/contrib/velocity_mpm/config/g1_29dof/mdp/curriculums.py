# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Terrain progression for the moving-patch task."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

from isaaclab_tasks.core.velocity.mdp.curriculums import terrain_levels_vel as terrain_levels_vel_base

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def terrain_levels_vel(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int] | torch.Tensor | slice,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Update terrain difficulty from walking progress after the initial placement.

    Args:
        env: Environment whose robots are reset onto generated terrain tiles.
        env_ids: Environments being reset.
        asset_cfg: Robot used to measure progress.

    Returns:
        Mean terrain level over all environments.
    """
    # Before the first reset, robot positions still refer to the scene's clone grid.
    if env.common_step_counter == 0:
        return env.scene.terrain.terrain_levels.float().mean()
    return terrain_levels_vel_base(env, env_ids, asset_cfg)
