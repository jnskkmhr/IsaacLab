# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""G1 towing a sled over rigid ground and analytical sand: a manager-based environment plus a rope."""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv

from isaaclab_tasks.contrib.push.util import Tether

from .tow_env_cfg import G1TowEnvCfg


class G1SoftContactTowEnv(ManagerBasedRLEnv):
    """Walking on rigid ground or analytical (RFT) sand with a sled tied to the robot.

    The rope tension is recomputed once per environment step and held for the whole step, as in the sand task.
    """

    cfg: G1TowEnvCfg

    def load_managers(self) -> None:
        self.tether = Tether(self.cfg.tether, self.scene)
        """The rope between the robot and the sled."""
        self._tow_object = self.scene[self.cfg.tether.object_name]
        self.sled_path = torch.zeros(self.num_envs, device=self.device)
        """Path the sled travelled in the current episode [m], shape ``(num_envs,)``."""
        self.commanded_path = torch.zeros(self.num_envs, device=self.device)
        """Path the velocity command asked for in the current episode [m], shape ``(num_envs,)``."""
        # The environments on the analytical sand column start with a light sled and follow the curriculum.
        generator = self.scene.terrain.cfg.terrain_generator
        column = torch.floor(self.scene.env_origins[:, 1] / generator.size[1] + generator.num_cols / 2)
        self.sled_mass_curriculum_mask = column == 1
        """Environments on analytical sand, whose sled mass follows the curriculum, shape ``(num_envs,)``."""
        super().load_managers()

    def step(self, action: torch.Tensor):
        self.tether.apply()
        obs, reward, terminated, truncated, extras = super().step(action)
        tension = self.tether.tension
        sled_speed = torch.linalg.norm(self._tow_object.data.root_lin_vel_w.torch[:, :2], dim=-1)
        command_speed = torch.linalg.norm(self.command_manager.get_command("base_velocity")[:, :2], dim=-1)
        # Environments reset in this step start again from the zero the reset left.
        running = self.episode_length_buf > 0
        self.sled_path += sled_speed * self.step_dt * running
        self.commanded_path += command_speed * self.step_dt * running
        log = {
            "Metrics/tow/tension": tension.mean(),
            "Metrics/tow/taut_fraction": (tension > 1.0).float().mean(),
            "Metrics/tow/sled_speed": sled_speed.mean(),
        }
        for name, mask in (("sand", self.sled_mass_curriculum_mask), ("rigid", ~self.sled_mass_curriculum_mask)):
            if mask.any():
                log[f"Metrics/tow/sled_speed_{name}"] = sled_speed[mask].mean()
                log[f"Metrics/tow/tension_{name}"] = tension[mask].mean()
        extras["log"] = {**extras.get("log", {}), **log}
        # Keep the training viewer on the last robot, which is on the sand column.
        if self.cfg.camera_follow_eye is not None and self.sim.visualizers:
            position = self.scene["robot"].data.root_pos_w.torch[-1].cpu().numpy()
            for visualizer in self.sim.visualizers:
                visualizer.set_camera_view(tuple(position + self.cfg.camera_follow_eye), tuple(position))
        return obs, reward, terminated, truncated, extras

    def _reset_idx(self, env_ids) -> None:
        # The curriculum reads the paths of the finished episode inside the reset.
        super()._reset_idx(env_ids)
        self.sled_path[env_ids] = 0.0
        self.commanded_path[env_ids] = 0.0
