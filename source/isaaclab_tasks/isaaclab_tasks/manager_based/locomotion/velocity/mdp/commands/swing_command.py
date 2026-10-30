# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import CommandTerm, CommandTermCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class SwingCommand(CommandTerm):
    """
    A peak swing foot height commands.
    """

    def __init__(self, cfg: CommandTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.rng = torch.distributions.Uniform(cfg.foot_height[0], cfg.foot_height[1])
        self.target_height = self.rng.sample((self.num_envs,)).to(self.device)

    def _update_metrics(self):
        pass  # no metrics

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        # resample target height
        if self.cfg.resample:
            self.target_height[env_ids] = self.rng.sample((len(env_ids),)).to(self.device)

    def _update_command(self) -> None:
        pass

    def update_target_height(self, env_ids: torch.Tensor = None, new_target_height: torch.Tensor = None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        self.target_height[env_ids] = new_target_height

    @property
    def command(self) -> torch.Tensor:
        return self.target_height
