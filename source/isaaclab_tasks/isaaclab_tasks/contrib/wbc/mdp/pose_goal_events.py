# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Independent initial states for the terminal-goal task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def reset_pose_goal_robot(env: ManagerBasedRLEnv, env_ids: torch.Tensor | slice) -> None:
    """Reset only the selected robots; command reset samples independent goals afterwards."""
    env.command_manager.get_term("pose_goal").reset_robot(env_ids)
