# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Terrain-derived destinations without a reference trajectory."""

import math
from dataclasses import MISSING

import torch

from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils import math as math_utils
from isaaclab.utils.configclass import configclass


class StairGoalCommand(CommandTerm):
    """Climb from a ground border to the top landing, retaining the pretrained goal interface."""

    cfg: "StairGoalCommandCfg"

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.robot = env.scene[cfg.asset_name]
        self.terrain = env.scene.terrain
        self.body_indexes = self.robot.find_bodies(cfg.body_names, preserve_order=True)[0]
        self.anchor_body_index = self.robot.find_bodies(cfg.anchor_body_name)[0][0]
        self.foot_ids = env.scene["contact_forces"].find_bodies(".*_ankle_roll_link")[0]
        self.is_imitation = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.goal_reached = torch.zeros_like(self.is_imitation)
        self.approach_yaw = torch.zeros(self.num_envs, device=self.device)
        self.goal_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.goal_quat_w = torch.zeros(self.num_envs, 4, device=self.device)
        self.goal_quat_w[:, 3] = 1.0
        self.previous_goal_distance = torch.zeros(self.num_envs, device=self.device)
        self.metrics = {
            name: torch.zeros(self.num_envs, device=self.device) for name in ("goal_distance", "goal_heading_error")
        }

    @property
    def goal_error(self) -> torch.Tensor:
        return self.goal_pos_w - self.robot.data.root_pos_w.torch

    @property
    def heading_error(self) -> torch.Tensor:
        goal_yaw = math_utils.euler_xyz_from_quat(self.goal_quat_w)[2]
        root_yaw = math_utils.euler_xyz_from_quat(self.robot.data.root_quat_w.torch)[2]
        return math_utils.wrap_to_pi(goal_yaw - root_yaw)

    @property
    def position_reached(self) -> torch.Tensor:
        error = self.goal_error
        return (error[:, :2].norm(dim=-1) < self.cfg.position_tolerance) & (
            error[:, 2].abs() < self.cfg.height_tolerance
        )

    @property
    def command(self) -> torch.Tensor:
        """Destination XYZ in the robot yaw frame and sine/cosine of destination yaw error."""
        error_b = math_utils.quat_apply_inverse(math_utils.yaw_quat(self.robot.data.root_quat_w.torch), self.goal_error)
        heading = self.heading_error
        return torch.cat((error_b, heading.sin().unsqueeze(-1), heading.cos().unsqueeze(-1)), dim=-1)

    @property
    def robot_anchor_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w.torch[:, self.anchor_body_index]

    @property
    def robot_anchor_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w.torch[:, self.anchor_body_index]

    @property
    def robot_body_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w.torch[:, self.body_indexes]

    @property
    def robot_body_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w.torch[:, self.body_indexes]

    def sample_spawn(self, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample a verified ground patch and an inward-facing yaw for the reset event."""
        patches = self.terrain.flat_patches["spawn"]
        indices = torch.randint(patches.shape[2], (len(env_ids),), device=self.device)
        positions = patches[self.terrain.terrain_levels[env_ids], self.terrain.terrain_types[env_ids], indices]
        self.approach_yaw[env_ids] = torch.randint(4, (len(env_ids),), device=self.device) * (math.pi / 2)
        yaw = self.approach_yaw[env_ids]
        rotation = math_utils.quat_from_euler_xyz(torch.zeros_like(yaw), torch.zeros_like(yaw), yaw)
        origins = self.terrain.env_origins[env_ids]
        positions = math_utils.quat_apply(rotation, positions - origins) + origins
        positions[:, 2] += self.cfg.root_height
        yaw = yaw + torch.empty_like(yaw).uniform_(-self.cfg.initial_yaw_noise, self.cfg.initial_yaw_noise)
        return positions, math_utils.quat_from_euler_xyz(torch.zeros_like(yaw), torch.zeros_like(yaw), yaw)

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        success = self.goal_reached[env_ids].float().mean()
        extras = super().reset(env_ids)
        extras["task_success_rate"] = success
        extras["task_episode_fraction"] = torch.ones((), device=self.device)
        self.goal_reached[env_ids] = False
        self.previous_goal_distance[env_ids] = self.goal_error[env_ids].norm(dim=-1)
        return extras

    def _resample_command(self, env_ids):
        if isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device)[env_ids]
        patches = self.terrain.flat_patches["target"]
        indices = torch.randint(patches.shape[2], (len(env_ids),), device=self.device)
        self.goal_pos_w[env_ids] = patches[
            self.terrain.terrain_levels[env_ids], self.terrain.terrain_types[env_ids], indices
        ]
        self.goal_pos_w[env_ids, 2] += self.cfg.root_height
        yaw = self.approach_yaw[env_ids]
        self.goal_quat_w[env_ids] = math_utils.quat_from_euler_xyz(torch.zeros_like(yaw), torch.zeros_like(yaw), yaw)

    def _update_metrics(self):
        forces = self._env.scene["contact_forces"].data.net_forces_w[:, self.foot_ids]
        supported = (forces.norm(dim=-1) > self.cfg.contact_threshold).any(dim=-1)
        self.goal_reached |= self.position_reached & (self.heading_error.abs() < self.cfg.heading_tolerance) & supported
        self.metrics["goal_distance"][:] = self.goal_error.norm(dim=-1)
        self.metrics["goal_heading_error"][:] = self.heading_error.abs()

    def _update_command(self):
        self.previous_goal_distance[:] = self.goal_error.norm(dim=-1)

    def _set_debug_vis_impl(self, debug_vis):
        if debug_vis and not hasattr(self, "goal_visualizer"):
            self.goal_visualizer = VisualizationMarkers(self.cfg.goal_visualizer_cfg)
        if hasattr(self, "goal_visualizer"):
            self.goal_visualizer.set_visibility(debug_vis)

    def _debug_vis_callback(self, event):
        if self.robot.is_initialized:
            self.goal_visualizer.visualize(self.goal_pos_w, self.goal_quat_w)


@configclass
class StairGoalCommandCfg(CommandTermCfg):
    """Ascent-only destinations on generated pyramid staircases."""

    class_type: type = StairGoalCommand
    asset_name: str = "robot"
    body_names: list[str] = MISSING
    anchor_body_name: str = "torso_link"
    root_height: float = 0.8
    """Desired root height above the sampled ground or landing surface, in meters."""
    initial_yaw_noise: float = 0.2
    """Uniform yaw noise in radians around the inward-facing cardinal heading."""
    position_tolerance: float = 0.35
    height_tolerance: float = 0.25
    heading_tolerance: float = 0.4
    contact_threshold: float = 5.0
    goal_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/destination")
