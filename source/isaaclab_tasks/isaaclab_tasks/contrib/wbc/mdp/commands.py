# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass

from .dataset import WholeBodyDataset

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class WholeBodyCommand(CommandTerm):
    """Switch static targets within one foot stance while the robot continues moving."""

    def __init__(self, cfg: WholeBodyCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.robot = env.scene["robot"]
        self.joint_ids = self.robot.find_joints(cfg.joint_names, preserve_order=True)[0]
        self.body_ids = self.robot.find_bodies(cfg.body_names, preserve_order=True)[0]
        self.dataset = WholeBodyDataset(cfg.dataset_path, cfg.joint_names, cfg.body_names, self.device)
        if not self.dataset.is_static or self.dataset.stance_ids is None:
            raise ValueError("WBC target switching requires static poses generated with --stance-groups")
        if cfg.initial_max_joint_rms <= 0.0 or cfg.tracking_grace_period < 0.0:
            raise ValueError("Joint RMS limit must be positive and tracking grace period nonnegative")
        distance = torch.cdist(self.dataset.joint_pos, self.dataset.joint_pos) / len(cfg.joint_names) ** 0.5
        compatible = self.dataset.stance_ids[:, None] == self.dataset.stance_ids[None, :]
        compatible.fill_diagonal_(False)
        distance[~compatible] = torch.inf
        self.neighbor_distances, self.neighbor_indices = distance.sort(dim=1)
        if torch.any(self.neighbor_distances[:, 0] > cfg.initial_max_joint_rms):
            raise ValueError("Every pose needs a same-stance neighbor within initial_max_joint_rms")
        self.max_joint_rms = cfg.initial_max_joint_rms
        self.frame = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.end_frame = torch.zeros_like(self.frame)
        self.time_since_switch = torch.zeros(self.num_envs, device=self.device)
        self.metrics["position_error_m"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["target_joint_rms"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["target_switches"] = torch.zeros(self.num_envs, device=self.device)
        self.sample(torch.arange(self.num_envs, device=self.device))

    @property
    def command(self) -> torch.Tensor:
        return self.dataset.joint_pos[self.frame]

    @property
    def target_body_pos_w(self) -> torch.Tensor:
        positions = self.dataset.body_pos_w[self.frame].clone()
        positions[:, 1:3] = self.dataset.stance_foot_pos_w[self.dataset.stance_ids[self.frame]]
        return positions + self._env.scene.env_origins[:, None, :]

    @property
    def target_body_quat_w(self) -> torch.Tensor:
        orientations = self.dataset.body_quat_w[self.frame].clone()
        orientations[:, 1:3] = self.dataset.stance_foot_quat_w[self.dataset.stance_ids[self.frame]]
        return orientations

    def sample(self, env_ids: torch.Tensor | slice) -> None:
        """Choose an initial pose for the episode reset event, which also initializes the robot."""
        count = self.frame[env_ids].numel()
        self.frame[env_ids] = torch.randint(len(self.dataset.joint_pos), (count,), device=self.device)
        self.end_frame[env_ids] = self.frame[env_ids]
        self.time_since_switch[env_ids] = 0.0

    def _resample_command(self, env_ids: torch.Tensor | slice) -> None:
        # CommandTerm.reset calls this after the reset event has already chosen a pose.
        ids = torch.arange(self.num_envs, device=self.device)[env_ids]
        ids = ids[self.command_counter[ids] > 0]
        if not len(ids):
            return
        previous = self.frame[ids]
        counts = (self.neighbor_distances[previous] <= self.max_joint_rms).sum(dim=1)
        ranks = (torch.rand(len(ids), device=self.device) * counts).long()
        self.frame[ids] = self.neighbor_indices[previous, ranks]
        self.end_frame[ids] = self.frame[ids]
        self.time_since_switch[ids] = 0.0
        self.metrics["target_joint_rms"][ids] = self.neighbor_distances[previous, ranks]
        self.metrics["target_switches"][ids] += 1

    def _update_command(self) -> None:
        self.time_since_switch += self._env.step_dt

    def _update_metrics(self) -> None:
        difference = self.robot.data.body_pos_w.torch[:, self.body_ids] - self.target_body_pos_w
        self.metrics["position_error_m"][:] = difference.norm(dim=-1).mean(dim=-1)


@configclass
class WholeBodyCommandCfg(CommandTermCfg):
    """Static targets resampled within a common foot stance, without resetting physical state."""

    class_type: type = WholeBodyCommand
    dataset_path: str = str(Path(__file__).resolve().parents[1] / "data" / "stance_poses.npz")
    joint_names: list[str] = MISSING
    body_names: list[str] = MISSING
    resampling_time_range: tuple[float, float] = (4.0, 10.0)
    initial_max_joint_rms: float = 0.25
    tracking_grace_period: float = 2.0
