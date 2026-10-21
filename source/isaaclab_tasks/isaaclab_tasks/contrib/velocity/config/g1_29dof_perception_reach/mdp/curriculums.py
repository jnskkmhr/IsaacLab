# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Terrain progression based on finishing an episode within the goal pose tolerances."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from .commands import TerrainPoseCommand


def terrain_levels_reach(
    env: ManagerBasedRLEnv, env_ids: Sequence[int] | slice, command_name: str = "base_pose"
) -> torch.Tensor:
    """Advance on final-state success; regress on failure or an unsuccessful timeout.

    Initial and manual resets do not change difficulty. The terrain importer
    retains its level bounds and its random-level behavior after the last level.
    """
    command: TerrainPoseCommand = env.command_manager.get_term(command_name)
    command.update_goal_status()
    env_ids = env.scene._ALL_INDICES[env_ids]
    completed = env.termination_manager.dones[env_ids] & (env.episode_length_buf[env_ids] > 0)
    env_ids = env_ids[completed]
    if len(env_ids):
        success = command.episode_success[env_ids]
        env.scene.terrain.update_env_origins(env_ids, move_up=success, move_down=~success)
    return env.scene.terrain.terrain_levels.float().mean()
