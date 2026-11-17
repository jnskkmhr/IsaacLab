# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sample pelvis height above each environment's flat ground."""

import torch

from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers


class UniformHeightCommand(CommandTerm):
    """A scalar root-link height command in meters above the environment origin."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        lower, upper = cfg.height_range
        if not 0.0 < lower <= upper:
            raise ValueError("height_range must satisfy 0 < lower <= upper.")
        self.robot = env.scene[cfg.asset_name]
        self._command = torch.full((self.num_envs, 1), lower, device=self.device)
        self.metrics["height_error"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        """Desired root height above flat ground, shape (num_envs, 1)."""
        return self._command

    def _resample_command(self, env_ids):
        self._command[env_ids] = torch.empty_like(self._command[env_ids]).uniform_(*self.cfg.height_range)

    def _update_command(self):
        pass

    def _update_metrics(self):
        height = self.robot.data.root_pos_w.torch[:, 2] - self._env.scene.env_origins[:, 2]
        self.metrics["height_error"][:] = (height - self.command[:, 0]).abs()

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "goal_height_visualizer"):
                self.goal_height_visualizer = VisualizationMarkers(self.cfg.goal_height_visualizer_cfg)
                self.current_height_visualizer = VisualizationMarkers(self.cfg.current_height_visualizer_cfg)
            self.goal_height_visualizer.set_visibility(True)
            self.current_height_visualizer.set_visibility(True)
        elif hasattr(self, "goal_height_visualizer"):
            self.goal_height_visualizer.set_visibility(False)
            self.current_height_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return
        current_position = self.robot.data.root_pos_w.torch.clone()
        goal_position = current_position.clone()
        current_position[:, :2] += current_position.new_tensor(self.cfg.current_marker_xy_offset)
        goal_position[:, :2] += goal_position.new_tensor(self.cfg.goal_marker_xy_offset)
        goal_position[:, 2] = self._env.scene.env_origins[:, 2] + self.command[:, 0]
        environment_ids = self._env.scene._ALL_INDICES
        self.current_height_visualizer.visualize(current_position, environment_ids=environment_ids)
        self.goal_height_visualizer.visualize(goal_position, environment_ids=environment_ids)
