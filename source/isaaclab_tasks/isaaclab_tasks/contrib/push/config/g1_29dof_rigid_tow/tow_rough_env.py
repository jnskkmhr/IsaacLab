# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The rigid tow environment on rough terrain, with a share of environments whose tether is cut."""

from __future__ import annotations

import torch

from .tow_env import G1RigidTowEnv


class G1RigidTowRoughEnv(G1RigidTowEnv):
    """Rigid tow environment in which every ``cfg.no_load_every``-th environment tows nothing."""

    def load_managers(self) -> None:
        super().load_managers()
        # Spread over the terrain types: the terrain curriculum assigns the types by environment index.
        self.no_load_mask = torch.arange(self.num_envs, device=self.device) % self.cfg.no_load_every == 0
        """Environments whose tether is cut, shape ``(num_envs,)``."""
        self.tether.enabled[self.no_load_mask] = False
        # The parent logs the sled speed by this mask: "curriculum" is then the towing environments.
        self.sled_mass_curriculum_mask = ~self.no_load_mask

    def step(self, action: torch.Tensor):
        obs, reward, terminated, truncated, extras = super().step(action)
        speed = torch.linalg.norm(self.scene["robot"].data.root_lin_vel_w.torch[:, :2], dim=-1)
        extras["log"]["Metrics/tow/robot_speed_no_load"] = speed[self.no_load_mask].mean()
        extras["log"]["Metrics/tow/robot_speed_towing"] = speed[~self.no_load_mask].mean()
        return obs, reward, terminated, truncated, extras
