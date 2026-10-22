# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Privileged training observations and destination symmetry."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def imitation_mode(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Expose the reward mode to the critic, never to the actor."""
    return env.command_manager.get_term("motion").is_imitation.float().unsqueeze(-1)


def reference_joint_pos(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reference joints used by the critic to predict imitation returns."""
    return env.command_manager.get_term("motion").joint_pos


def mirror_goal(observation: torch.Tensor) -> torch.Tensor:
    """Reflect local destination Y and sine of yaw error across the robot's sagittal plane."""
    mirrored = observation.clone()
    mirrored[..., 1] *= -1
    mirrored[..., 3] *= -1
    return mirrored
