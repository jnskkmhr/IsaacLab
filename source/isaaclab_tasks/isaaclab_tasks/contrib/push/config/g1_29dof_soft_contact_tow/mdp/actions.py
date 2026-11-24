# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Action terms of the soft-contact tow task."""

from __future__ import annotations

import torch

from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils.configclass import configclass


class TetherHoldAction(ActionTerm):
    """Keep the rope force applied while the analytical contact terms rewrite the permanent wrenches.

    The analytical contact terms set the permanent wrench of their asset once per physics step, which replaces
    the rope force the environment set at the start of the step. This term runs after them and adds the held
    rope force back. It has no policy action.
    """

    cfg: TetherHoldActionCfg

    def __init__(self, cfg: TetherHoldActionCfg, env):
        super().__init__(cfg, env)
        self._raw_actions = torch.zeros(self.num_envs, 0, device=self.device)

    @property
    def action_dim(self) -> int:
        return 0

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._raw_actions

    def process_actions(self, actions: torch.Tensor):
        pass

    def apply_actions(self):
        self._env.tether.reapply()


@configclass
class TetherHoldActionCfg(ActionTermCfg):
    """Configuration of the rope re-apply term."""

    class_type: type[ActionTerm] = TetherHoldAction
