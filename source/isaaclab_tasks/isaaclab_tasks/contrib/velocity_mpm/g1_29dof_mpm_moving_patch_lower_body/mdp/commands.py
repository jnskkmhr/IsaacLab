# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Height commands relative to the ray-hit rigid surface beneath the robot."""

import math

import torch

from isaaclab.managers import CommandTerm


class UniformHeightCommand(CommandTerm):
    """Sample desired root height above the ray-hit surface, matching base_height_l2."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        lower, upper = cfg.height_range
        if not all(math.isfinite(value) for value in (lower, upper)) or not 0.0 < lower <= upper:
            raise ValueError("height_range must contain finite values satisfying 0 < lower <= upper.")
        self.robot = env.scene[cfg.asset_name]
        self.sensor = env.scene[cfg.sensor_name]
        self._command = torch.full((self.num_envs, 1), lower, device=self.device)
        self.metrics["height_error"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        """Desired root height [m], shape (num_envs, 1)."""
        return self._command

    @property
    def measured_height(self) -> torch.Tensor:
        """Root height above mean finite ray hits [m]; no particle-depth correction."""
        hits = self.sensor.data.ray_hits_w.torch[..., 2]
        finite = torch.isfinite(hits)
        terrain_height = torch.where(finite, hits, torch.zeros_like(hits)).sum(dim=1)
        terrain_height /= finite.sum(dim=1).clamp_min(1)
        return self.robot.data.root_pos_w.torch[:, 2] - terrain_height

    def _resample_command(self, env_ids):
        self._command[env_ids] = torch.empty_like(self._command[env_ids]).uniform_(*self.cfg.height_range)

    def _update_command(self):
        pass

    def _update_metrics(self):
        self.metrics["height_error"][:] = (self.measured_height - self.command[:, 0]).abs()
