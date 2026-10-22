# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Destination commands with separate imitation and task episodes."""

import torch

from isaaclab.markers import VisualizationMarkers
from isaaclab.utils import math as math_utils
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.contrib.mimic.mdp.commands import MotionCommand, MotionCommandCfg


class GoalMotionCommand(MotionCommand):
    """Condition the actor on a destination while retaining references for imitation rewards."""

    cfg: "GoalMotionCommandCfg"

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.is_imitation = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self.goal_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.goal_quat_w = torch.zeros(self.num_envs, 4, device=self.device)
        self.goal_quat_w[:, 3] = 1.0
        self.previous_goal_distance = torch.zeros(self.num_envs, device=self.device)
        self.goal_reached = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._finished_task = torch.zeros_like(self.is_imitation)
        self._finished_success = torch.zeros_like(self.is_imitation)
        self.foot_ids = env.scene["contact_forces"].find_bodies(".*_ankle_roll_link")[0]
        self.metrics = {
            name: torch.zeros(self.num_envs, device=self.device)
            for name in ("goal_distance", "goal_heading_error", "imitation_probability")
        }

    @property
    def sampling_probability(self) -> float:
        """Probability of an imitation episode at the current per-environment step count."""
        if self.cfg.imitation_probability is not None:
            return self.cfg.imitation_probability
        fraction = min(
            max((self._env.common_step_counter - self.cfg.warmup_steps) / self.cfg.transition_steps, 0.0), 1.0
        )
        return 1.0 - fraction * (1.0 - self.cfg.final_imitation_probability)

    @property
    def goal_error(self) -> torch.Tensor:
        """World-frame displacement from the robot root to its destination."""
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

    def _adaptive_sampling(self, env_ids):
        # Episode type is hidden from the actor. RSI is used only for imitation;
        # task episodes always start at the beginning of the paired terrain route.
        self._finished_task[env_ids] = ~self.is_imitation[env_ids]
        self._finished_success[env_ids] = self.goal_reached[env_ids]
        self.goal_reached[env_ids] = False
        self.is_imitation[env_ids] = torch.rand(len(env_ids), device=self.device) < self.sampling_probability
        self.motion_ids[env_ids] = torch.randint(self.num_motions, (len(env_ids),), device=self.device)
        frames = (torch.rand(len(env_ids), device=self.device) * self.motion_lengths[env_ids]).long()
        full_clip = torch.rand(len(env_ids), device=self.device) < self.cfg.full_clip_probability
        self.time_steps[env_ids] = torch.where(self.is_imitation[env_ids] & ~full_clip, frames, 0)
        self.assist_scale[env_ids] = 0.0
        self._start_bin_valid[env_ids] = False

    def resample_time_steps(self, env_ids):
        if isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device)[env_ids]
        else:
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        pending = env_ids[self._resample_pending[env_ids]]
        super().resample_time_steps(env_ids)
        frames = self.motion.frame_indices(self.motion_ids[pending], self.motion_lengths[pending] - 1)
        self.goal_pos_w[pending] = (
            self.motion._body_pos_w[frames, self.motion_root_body_index] + self._env.scene.env_origins[pending]
        )
        self.goal_quat_w[pending] = math_utils.yaw_quat(self.motion._body_quat_w[frames, self.motion_root_body_index])
        # The endpoint lies on the landing platform. Keep this variation small enough
        # to remain on that platform; do not move the terrain or reference trajectory.
        noise = torch.empty(len(pending), 2, device=self.device).uniform_(
            -self.cfg.goal_position_noise, self.cfg.goal_position_noise
        )
        self.goal_pos_w[pending, :2] += noise * (~self.is_imitation[pending]).unsqueeze(-1)

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        extras = super().reset(env_ids)
        tasks = self._finished_task[env_ids]
        extras["task_episode_fraction"] = tasks.float().mean()
        extras["task_success_rate"] = (self._finished_success[env_ids] & tasks).sum() / tasks.sum().clamp_min(1)
        self.previous_goal_distance[env_ids] = self.goal_error[env_ids].norm(dim=-1)
        return extras

    def _update_metrics(self):
        forces = self._env.scene["contact_forces"].data.net_forces_w[:, self.foot_ids]
        supported = (forces.norm(dim=-1) > self.cfg.contact_threshold).any(dim=-1)
        self.goal_reached |= self.position_reached & (self.heading_error.abs() < self.cfg.heading_tolerance) & supported
        self.metrics["goal_distance"][:] = self.goal_error.norm(dim=-1)
        self.metrics["goal_heading_error"][:] = self.heading_error.abs()
        self.metrics["imitation_probability"][:] = self.sampling_probability

    def _update_command(self):
        super()._update_command()
        # Rewards run before this callback, so this is the previous step's distance
        # when the next progress reward is evaluated.
        self.previous_goal_distance[:] = self.goal_error.norm(dim=-1)

    def _set_debug_vis_impl(self, debug_vis):
        if debug_vis and not hasattr(self, "goal_visualizer"):
            self.goal_visualizer = VisualizationMarkers(
                self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/destination")
            )
        if hasattr(self, "goal_visualizer"):
            self.goal_visualizer.set_visibility(debug_vis)

    def _debug_vis_callback(self, event):
        if self.robot.is_initialized:
            self.goal_visualizer.visualize(self.goal_pos_w, self.goal_quat_w)


@configclass
class GoalMotionCommandCfg(MotionCommandCfg):
    """Reference-assisted training of a spatial goal policy on motion-paired terrain."""

    class_type: type = GoalMotionCommand
    warmup_steps: int = 120_000
    """Imitation-only control steps per environment (5,000 rollouts of length 24)."""
    transition_steps: int = 120_000
    """Control steps over which imitation sampling decreases to its final probability."""
    final_imitation_probability: float = 0.5
    imitation_probability: float | None = None
    """Override the sampling schedule, e.g. zero for task evaluation."""
    full_clip_probability: float = 0.2
    initial_position_noise: float = 0.15
    """Uniform initial XY displacement in meters, only in task episodes."""
    initial_yaw_noise: float = 0.15
    """Uniform initial yaw displacement in radians, only in task episodes."""
    goal_position_noise: float = 0.2
    """Uniform destination XY displacement in meters, only in task episodes."""
    position_tolerance: float = 0.35
    height_tolerance: float = 0.25
    heading_tolerance: float = 0.4
    contact_threshold: float = 5.0
