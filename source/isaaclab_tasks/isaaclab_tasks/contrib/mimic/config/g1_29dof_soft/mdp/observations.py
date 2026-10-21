# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Stance observations for soft-contact G1 motion tracking."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def motion_standing_weight(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Return the current stance blend weight in [0, 1]."""
    return env.command_manager.get_term(command_name).standing_weight.unsqueeze(-1)
