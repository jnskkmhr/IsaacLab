# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Placeholder action term for the `particle` task."""

import torch

from isaaclab.managers import ActionTerm


class NullAction(ActionTerm):
    """One-dimensional action that is stored but never applied.

    The scene is passive (a free sand block), but the action manager needs at least one term.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._actions = torch.zeros((env.num_envs, 1), device=env.device)

    @property
    def action_dim(self) -> int:
        return 1

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._actions

    def process_actions(self, actions: torch.Tensor) -> None:
        self._actions.copy_(actions)

    def apply_actions(self) -> None:
        return
