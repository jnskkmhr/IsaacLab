# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Mix reference-compatible imitation episodes with reference-free stair ascent."""

import math

import torch

from isaaclab.utils import math as math_utils
from isaaclab.utils.configclass import configclass

from .commands import GoalMotionCommand, GoalMotionCommandCfg


class MixedStairGoalCommand(GoalMotionCommand):
    """Use column zero for imitation and the remaining generated columns for task episodes."""

    cfg: "MixedStairGoalCommandCfg"

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.terrain = env.scene.terrain
        num_task_columns = self.terrain.terrain_origins.shape[1] - 1
        # Terrain types retain the task assignment even while an environment imitates.
        self.terrain.terrain_types[:] = 1 + torch.arange(self.num_envs, device=self.device) % num_task_columns
        self.approach_yaw = torch.zeros(self.num_envs, device=self.device)

    def _adaptive_sampling(self, env_ids):
        super()._adaptive_sampling(env_ids)
        columns = torch.where(self.is_imitation[env_ids], 0, self.terrain.terrain_types[env_ids])
        self.terrain.env_origins[env_ids] = self.terrain.terrain_origins[self.terrain.terrain_levels[env_ids], columns]

    def sample_spawn(self, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample a ground-border position and a cardinal heading into the staircase."""
        patches = self.terrain.flat_patches["spawn"]
        indices = torch.randint(patches.shape[2], (len(env_ids),), device=self.device)
        position = patches[self.terrain.terrain_levels[env_ids], self.terrain.terrain_types[env_ids], indices]
        yaw = torch.randint(4, (len(env_ids),), device=self.device) * (math.pi / 2)
        self.approach_yaw[env_ids] = yaw
        rotation = math_utils.quat_from_euler_xyz(torch.zeros_like(yaw), torch.zeros_like(yaw), yaw)
        origin = self.terrain.env_origins[env_ids]
        position = math_utils.quat_apply(rotation, position - origin) + origin
        position[:, 2] += self.cfg.root_height
        # Set the landing heading after the approach is chosen, before command reset.
        self.goal_quat_w[env_ids] = rotation
        yaw += torch.empty_like(yaw).uniform_(-self.cfg.initial_yaw_noise, self.cfg.initial_yaw_noise)
        return position, math_utils.quat_from_euler_xyz(torch.zeros_like(yaw), torch.zeros_like(yaw), yaw)

    def resample_time_steps(self, env_ids):
        if isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device)[env_ids]
        else:
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        pending = env_ids[self._resample_pending[env_ids]]
        super().resample_time_steps(env_ids)
        task_ids = pending[~self.is_imitation[pending]]
        patches = self.terrain.flat_patches["target"]
        indices = torch.randint(patches.shape[2], (len(task_ids),), device=self.device)
        self.goal_pos_w[task_ids] = patches[
            self.terrain.terrain_levels[task_ids], self.terrain.terrain_types[task_ids], indices
        ]
        self.goal_pos_w[task_ids, 2] += self.cfg.root_height

    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        # Terminations run before the first command update. Do not compare the
        # newly reset robot against zero buffers or the previous episode's pose.
        self.body_pos_relative_w[env_ids] = self.body_pos_w[env_ids]
        self.body_quat_relative_w[env_ids] = self.body_quat_w[env_ids]


@configclass
class MixedStairGoalCommandCfg(GoalMotionCommandCfg):
    """Retain imitation on the original stairs while learning goals on generated stairs."""

    class_type: type = MixedStairGoalCommand
    terrain_mode: str = "scene"
    imitation_probability: float | None = 0.5
    root_height: float = 0.8
    """Root height above a sampled ground or landing surface, in meters."""
    initial_yaw_noise: float = 0.2
