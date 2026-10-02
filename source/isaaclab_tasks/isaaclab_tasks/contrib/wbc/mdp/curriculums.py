# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Curricula for progressively larger fixed-stance target changes."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def target_joint_change(
    env: ManagerBasedRLEnv, env_ids: torch.Tensor, final_max_joint_rms: float, num_steps: int
) -> float:
    """Increase the allowed RMS joint-target difference with elapsed environment steps."""
    command = env.command_manager.get_term("whole_body")
    initial = command.cfg.initial_max_joint_rms
    if num_steps <= 0 or final_max_joint_rms < initial:
        raise ValueError("Target curriculum requires positive duration and a nondecreasing joint RMS limit")
    fraction = min(env.common_step_counter / num_steps, 1.0)
    command.max_joint_rms = initial + fraction * (final_max_joint_rms - initial)
    return command.max_joint_rms
