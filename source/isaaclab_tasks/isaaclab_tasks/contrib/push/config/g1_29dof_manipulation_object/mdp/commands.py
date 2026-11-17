# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Deterministic wrist trajectories for testing a frozen reaching policy."""

import torch

from isaaclab_tasks.contrib.push.config.g1_29dof_manipulation.mdp.commands import LinkPoseCommand


class ForwardPoseCommand(LinkPoseCommand):
    """Hold, advance along a straight line, then hold the endpoint until reset.

    Positions and the fixed XYZW orientation are expressed in the configured
    reference link. No workspace projection silently changes the requested path.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        if cfg.duration <= 0.0 or cfg.hold_time < 0.0:
            raise ValueError("duration must be positive and hold_time nonnegative.")
        self._start = torch.tensor(cfg.start_position, device=self.device)
        self._end = torch.tensor(cfg.end_position, device=self.device)
        self._orientation = torch.tensor(cfg.orientation, device=self.device)
        if not torch.isclose(self._orientation.norm(), self._orientation.new_tensor(1.0)):
            raise ValueError("orientation must be a unit XYZW quaternion.")
        self.pose_command_b[:, :3] = self._start
        self.pose_command_b[:, 3:] = self._orientation

    def _resample_command(self, env_ids):
        self.pose_command_b[env_ids, :3] = self._start
        self.pose_command_b[env_ids, 3:] = self._orientation

    def _update_command(self):
        elapsed = self._env.episode_length_buf * self._env.step_dt
        progress = ((elapsed - self.cfg.hold_time) / self.cfg.duration).clamp(0.0, 1.0)
        # Smooth start and stop, with an exactly straight Cartesian path.
        progress = progress.square() * (3.0 - 2.0 * progress)
        self.pose_command_b[:, :3] = self._start + progress[:, None] * (self._end - self._start)
        self.pose_command_b[:, 3:] = self._orientation
        super()._update_command()
