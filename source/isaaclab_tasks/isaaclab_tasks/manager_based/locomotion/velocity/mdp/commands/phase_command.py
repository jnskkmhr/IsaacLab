# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class PhaseCommand(CommandTerm):
    """
    A phase command that repeats from [0, 1].
    Locomotion gait period is randomized between a given range called gait_period.
    """

    def __init__(self, cfg: CommandTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.phase = torch.zeros(self.num_envs, device=self.device)
        # self.step_dt = env.step_dt  # BUG if used in planner, step_dt = physics_dt * decimation (of planner env)
        self.step_dt = cfg.step_dt

        self.rng: torch.distributions.Distribution = None
        if cfg.sampler == "uniform":
            self.rng = torch.distributions.Uniform(cfg.gait_period[0], cfg.gait_period[1])
        elif cfg.sampler == "discrete":
            # bin space 0.4, 0.5, 0.6, 0.7, 0.8
            self.bins = torch.linspace(cfg.gait_period[0], cfg.gait_period[1], 5, device=self.device)
            prob = torch.ones(len(self.bins)) / len(self.bins)
            self.rng = torch.distributions.Categorical(probs=prob)
        else:
            raise ValueError(f"Unsupported sampler type: {cfg.sampler}")

        self.phase_dt = torch.zeros(self.num_envs, device=self.device)
        self.phase_increment_per_step = self.step_dt / self.phase_dt
        self.ss_duration_phase = cfg.ss_duration_phase
        self.ds_duration_phase = cfg.ds_duration_phase
        self.ss_duration_phase_running = cfg.ss_duration_phase_running
        self.flight_duration_phase_running = cfg.flight_duration_phase_running

    def _update_command(self) -> None:
        self.phase += self.phase_increment_per_step  # advance by an amount of a step
        self.phase = torch.fmod(self.phase, 1.0)  # regulate to [0, 1]

        command = self._env.command_manager.get_command(self.cfg.velocity_command_name)
        stand_mask = torch.logical_and(torch.linalg.norm(command[:, :2], dim=1) < 0.01, torch.abs(command[:, 2]) < 0.01)
        self.phase[stand_mask] = 0.0

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        # resample phase freq and dt
        self.phase[env_ids] = 0.0  # reset phase to 0
        if self.cfg.resample:
            if self.cfg.sampler == "uniform":
                self.phase_dt[env_ids] = self.rng.sample((len(env_ids),)).to(self.device)
            elif self.cfg.sampler == "discrete":
                self.phase_dt[env_ids] = self.bins.index_select(0, self.rng.sample((len(env_ids),)).to(self.device))
        self.phase_increment_per_step[env_ids] = self.step_dt / self.phase_dt[env_ids]

    def update_gait_period(self, env_ids: torch.Tensor = None, new_gait_period: torch.Tensor = None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        self.phase_dt[env_ids] = new_gait_period
        self.phase_increment_per_step[env_ids] = self.step_dt / self.phase_dt[env_ids]

    def _update_metrics(self):
        pass  # no metrics

    @property
    def command(self) -> torch.Tensor:
        return torch.stack([torch.sin(2 * torch.pi * self.phase), torch.cos(2 * torch.pi * self.phase)], dim=-1)

    """
    bipedal walking gait
    """

    @property
    def planned_swing_state(self) -> torch.Tensor:
        """
        Returns swing state in each swing duration.

        0 < phase < ds_duration_phase/2 : double support
        ds_duration_phase/2 < phase < ds_duration_phase/2 + ss_duration_phase: left_swing
        ds_duration_phase/2 + ss_duration_phase < phase < 3*ds_duration_phase/2 + ss_duration_phase: double support
        3*ds_duration_phase/2 + ss_duration_phase < phase < 1 - ds_duration_phase/2: right_swing
        1 - ds_duration_phase/2 < phase < 1.0: double support
        """
        left_swing = (self.ds_duration_phase / 2 < self.phase) & (
            self.phase < self.ds_duration_phase / 2 + self.ss_duration_phase
        )
        right_swing = (3 * self.ds_duration_phase / 2 + self.ss_duration_phase < self.phase) & (
            self.phase < 1 - self.ds_duration_phase / 2
        )
        return torch.stack([left_swing, right_swing], dim=-1)

    @property
    def planned_running_swing_state(self) -> torch.Tensor:
        """
        Returns swing state in each swing duration.

        0 < phase < ds_duration_phase/2 : flight phase
        flight_duration_phase_running/2 < phase < flight_duration_phase_running/2 + ss_duration_phase_running: left_swing
        flight_duration_phase_running/2 + ss_duration_phase_running < phase < 3*flight_duration_phase_running/2 + ss_duration_phase_running: flight phase
        3*flight_duration_phase_running/2 + ss_duration_phase_running < phase < 1 - flight_duration_phase_running/2: right_swing
        1 - flight_duration_phase_running/2 < phase < 1.0: flight phase
        """
        left_swing = (self.flight_duration_phase_running / 2 < self.phase) & (
            self.phase < self.flight_duration_phase_running / 2 + self.ss_duration_phase_running
        )
        right_swing = (3 * self.flight_duration_phase_running / 2 + self.ss_duration_phase_running < self.phase) & (
            self.phase < 1 - self.flight_duration_phase_running / 2
        )
        return torch.stack([~right_swing, ~left_swing], dim=-1)

    @property
    def swing_phase(self) -> torch.Tensor:
        """
        Returns phase in each swing duration.

        0 < phase < ds_duration_phase/2 : double support
        ds_duration_phase/2 < phase < ds_duration_phase/2 + ss_duration_phase: left_swing
        ds_duration_phase/2 + ss_duration_phase < phase < 3*ds_duration_phase/2 + ss_duration_phase: double support
        3*ds_duration_phase/2 + ss_duration_phase < phase < 1 - ds_duration_phase/2: right_swing
        1 - ds_duration_phase/2 < phase < 1.0: double support

        or assymetric contact table
        0 < phase < ds_duration_phase: double support
        # ds_duration_phase < phase < ds_duration_phase + ss_duration_phase: left_swing
        # ds_duration_phase + ss_duration_phase < phase < 1 - ds_duration_phase: double support
        # 1 - ds_duration_phase < phase < 1.0: right_swing
        """

        left_swing_start = self.ds_duration_phase / 2
        left_swing_end = self.ds_duration_phase / 2 + self.ss_duration_phase
        left_in_swing = (self.phase > left_swing_start) & (self.phase < left_swing_end)
        phase_left = torch.where(
            left_in_swing,
            (self.phase - left_swing_start) / self.ss_duration_phase,  # normalize to 0-1 during swing
            torch.zeros_like(self.phase),  # 0 during stance/double support
        )

        right_swing_start = 3 * self.ds_duration_phase / 2 + self.ss_duration_phase
        right_swing_end = 1 - self.ds_duration_phase / 2
        right_in_swing = (self.phase > right_swing_start) & (self.phase < right_swing_end)
        phase_right = torch.where(
            right_in_swing,
            (self.phase - right_swing_start) / self.ss_duration_phase,  # normalize to 0-1 during swing
            torch.zeros_like(self.phase),  # 0 during stance/double support
        )

        return torch.stack([phase_left, phase_right], dim=-1)


class PhaseCommandSSP(CommandTerm):
    """
    A phase command that repeats from [0, 1].
    Locomotion gait period is randomized between a given range called gait_period.
    """

    def __init__(self, cfg: CommandTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.phase = torch.zeros(self.num_envs, device=self.device)
        # self.step_dt = env.step_dt  # BUG if used in planner, step_dt = physics_dt * decimation (of planner env)
        self.step_dt = cfg.step_dt

        self.rng: torch.distributions.Distribution = None
        if cfg.sampler == "uniform":
            self.rng = torch.distributions.Uniform(cfg.gait_period[0], cfg.gait_period[1])
        elif cfg.sampler == "discrete":
            # bin space 0.4, 0.5, 0.6, 0.7, 0.8
            self.bins = torch.linspace(cfg.gait_period[0], cfg.gait_period[1], 5, device=self.device)
            prob = torch.ones(len(self.bins)) / len(self.bins)
            self.rng = torch.distributions.Categorical(probs=prob)
        else:
            raise ValueError(f"Unsupported sampler type: {cfg.sampler}")

        self.phase_dt = torch.zeros(self.num_envs, device=self.device)
        self.phase_increment_per_step = self.step_dt / self.phase_dt

        self.robot: Articulation = env.scene[cfg.asset_name]
        body_ids, body_names = self.robot.find_bodies(self.cfg.body_names)
        self.body_ids = body_ids

        self._define_gait_parameters()

    def _define_gait_parameters(self):
        self.ss_duration_phase = self.cfg.ss_duration_phase * torch.ones(self.num_envs, device=self.device)
        self.is_standing = torch.zeros(self.num_envs, len(self.body_ids), device=self.device, dtype=torch.bool)

    def _set_debug_vis_impl(self, debug_vis: bool):
        # set visibility of markers
        # note: parent only deals with callbacks. not their visibility
        if debug_vis:
            if not hasattr(self, "gait_visualizer"):
                self.gait_visualizer = VisualizationMarkers(self.cfg.gait_visualizer_cfg)
            # set their visibility to true
            self.gait_visualizer.set_visibility(True)
        else:
            if hasattr(self, "gait_visualizer"):
                self.gait_visualizer.set_visibility(False)

    def _update_command(self) -> None:
        self.phase += self.phase_increment_per_step  # advance by an amount of a step
        self.phase = torch.fmod(self.phase, 1.0)  # regulate to [0, 1]

        command = self._env.command_manager.get_command(self.cfg.velocity_command_name)
        threshold = 0.01
        stand_mask = torch.logical_and(
            torch.linalg.norm(command[:, :2], dim=1) < threshold, torch.abs(command[:, 2]) < threshold
        )
        self.phase[stand_mask] = 0.0

        self.is_standing[stand_mask] = True
        self.is_standing[~stand_mask] = False

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        # resample phase freq and dt
        self.phase[env_ids] = 0.0  # reset phase to 0
        if self.cfg.resample:
            if self.cfg.sampler == "uniform":
                self.phase_dt[env_ids] = self.rng.sample((len(env_ids),)).to(self.device)
            elif self.cfg.sampler == "discrete":
                self.phase_dt[env_ids] = self.bins.index_select(0, self.rng.sample((len(env_ids),)).to(self.device))
        self.phase_increment_per_step[env_ids] = self.step_dt / self.phase_dt[env_ids]

    def _debug_vis_callback(self, event):
        # check if robot is initialized
        # note: this is needed in-case the robot is de-initialized. we can't access the data
        if not self.robot.is_initialized:
            return

        marker_pos = self.robot.data.body_pos_w[:, self.body_ids, :].clone()
        marker_pos[~self.planned_swing_state] = -5.0
        marker_pos = marker_pos.reshape(-1, 3)
        scale = (
            torch.tensor(
                self.gait_visualizer.cfg.markers["sphere"].radius,
                device=self.device,  # type: ignore
            )
            .unsqueeze(0)
            .repeat(marker_pos.shape[0], 3)
        )
        self.gait_visualizer.visualize(marker_pos, None, scale)

    """
    some helpers
    """

    def update_gait_period(self, env_ids: torch.Tensor = None, new_gait_period: torch.Tensor = None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        self.phase_dt[env_ids] = new_gait_period
        self.phase_increment_per_step[env_ids] = self.step_dt / self.phase_dt[env_ids]

    def _update_metrics(self):
        pass  # no metrics

    @property
    def command(self) -> torch.Tensor:
        return torch.stack([torch.sin(2 * torch.pi * self.phase), torch.cos(2 * torch.pi * self.phase)], dim=-1)

    """
    bipedal walking gait
    """

    @property
    def planned_swing_state(self) -> torch.Tensor:
        """
        Returns swing state in each swing duration.

        0 < phase < ss_duration_phase: left swing
        ss_duration_phase < phase < 1.0: right swing
        """
        left_swing = self.phase < self.ss_duration_phase
        right_swing = self.ss_duration_phase < self.phase
        return torch.stack([left_swing, right_swing], dim=-1) * ~self.is_standing

    @property
    def swing_phase(self) -> torch.Tensor:
        """
        Returns phase in each swing duration.

        0 < phase < ss_duration_phase: left swing
        ss_duration_phase < phase < 1.0: right swing
        """

        left_swing_start = 0
        left_swing_end = self.ss_duration_phase
        left_in_swing = (self.phase > left_swing_start) & (self.phase < left_swing_end)
        phase_left = torch.where(
            left_in_swing,
            (self.phase - left_swing_start) / self.ss_duration_phase,  # normalize to 0-1 during swing
            torch.zeros_like(self.phase),
        )

        right_swing_start = self.ss_duration_phase
        right_swing_end = 1.0
        right_in_swing = (self.phase > right_swing_start) & (self.phase < right_swing_end)
        phase_right = torch.where(
            right_in_swing,
            (self.phase - right_swing_start) / self.ss_duration_phase,  # normalize to 0-1 during swing
            torch.zeros_like(self.phase),
        )

        return torch.stack([phase_left, phase_right], dim=-1)


class PhaseCommandDSP(PhaseCommandSSP):
    """
    A phase command that repeats from [0, 1].
    Locomotion gait period is randomized between a given range called gait_period.
    """

    def __init__(self, cfg: CommandTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

    def _define_gait_parameters(self):
        self.ss_duration_phase = self.cfg.ss_duration_phase * torch.ones(self.num_envs, device=self.device)
        self.ds_duration_phase = self.cfg.ds_duration_phase * torch.ones(self.num_envs, device=self.device)
        self.is_standing = torch.zeros(self.num_envs, len(self.body_ids), device=self.device, dtype=torch.bool)

    def _update_command(self) -> None:
        self.phase += self.phase_increment_per_step  # advance by an amount of a step
        self.phase = torch.fmod(self.phase, 1.0)  # regulate to [0, 1]

        threshold = 0.01
        command = self._env.command_manager.get_command(self.cfg.velocity_command_name)
        stand_mask = torch.logical_and(
            torch.linalg.norm(command[:, :2], dim=1) < threshold, torch.abs(command[:, 2]) < threshold
        )
        self.phase[stand_mask] = 0.0

        self.is_standing[stand_mask] = True
        self.is_standing[~stand_mask] = False

    """
    bipedal walking gait
    """

    @property
    def planned_swing_state(self) -> torch.Tensor:
        """
        Returns swing state in each swing duration.

        0 < phase < ds_duration_phase/2 : double support
        ds_duration_phase/2 < phase < ds_duration_phase/2 + ss_duration_phase: left_swing
        ds_duration_phase/2 + ss_duration_phase < phase < 3*ds_duration_phase/2 + ss_duration_phase: double support
        3*ds_duration_phase/2 + ss_duration_phase < phase < 1 - ds_duration_phase/2: right_swing
        1 - ds_duration_phase/2 < phase < 1.0: double support
        """
        left_swing = (self.ds_duration_phase / 2 < self.phase) & (
            self.phase < self.ds_duration_phase / 2 + self.ss_duration_phase
        )
        right_swing = (3 * self.ds_duration_phase / 2 + self.ss_duration_phase < self.phase) & (
            self.phase < 1 - self.ds_duration_phase / 2
        )
        return torch.stack([left_swing, right_swing], dim=-1) * ~self.is_standing

    @property
    def planned_running_swing_state(self) -> torch.Tensor:
        """
        Returns swing state in each swing duration.

        0 < phase < ds_duration_phase/2 : flight phase
        flight_duration_phase_running/2 < phase < flight_duration_phase_running/2 + ss_duration_phase_running: left_swing
        flight_duration_phase_running/2 + ss_duration_phase_running < phase < 3*flight_duration_phase_running/2 + ss_duration_phase_running: flight phase
        3*flight_duration_phase_running/2 + ss_duration_phase_running < phase < 1 - flight_duration_phase_running/2: right_swing
        1 - flight_duration_phase_running/2 < phase < 1.0: flight phase
        """
        left_swing = (self.flight_duration_phase_running / 2 < self.phase) & (
            self.phase < self.flight_duration_phase_running / 2 + self.ss_duration_phase_running
        )
        right_swing = (3 * self.flight_duration_phase_running / 2 + self.ss_duration_phase_running < self.phase) & (
            self.phase < 1 - self.flight_duration_phase_running / 2
        )
        return torch.stack([~right_swing, ~left_swing], dim=-1) * ~self.is_standing

    @property
    def swing_phase(self) -> torch.Tensor:
        """
        Returns phase in each swing duration.

        0 < phase < ds_duration_phase/2 : double support
        ds_duration_phase/2 < phase < ds_duration_phase/2 + ss_duration_phase: left_swing
        ds_duration_phase/2 + ss_duration_phase < phase < 3*ds_duration_phase/2 + ss_duration_phase: double support
        3*ds_duration_phase/2 + ss_duration_phase < phase < 1 - ds_duration_phase/2: right_swing
        1 - ds_duration_phase/2 < phase < 1.0: double support

        or assymetric contact table
        0 < phase < ds_duration_phase: double support
        # ds_duration_phase < phase < ds_duration_phase + ss_duration_phase: left_swing
        # ds_duration_phase + ss_duration_phase < phase < 1 - ds_duration_phase: double support
        # 1 - ds_duration_phase < phase < 1.0: right_swing
        """

        left_swing_start = self.ds_duration_phase / 2
        left_swing_end = self.ds_duration_phase / 2 + self.ss_duration_phase
        left_in_swing = (self.phase > left_swing_start) & (self.phase < left_swing_end)
        phase_left = torch.where(
            left_in_swing,
            (self.phase - left_swing_start) / self.ss_duration_phase,  # normalize to 0-1 during swing
            torch.zeros_like(self.phase),  # 0 during stance/double support
        )

        right_swing_start = 3 * self.ds_duration_phase / 2 + self.ss_duration_phase
        right_swing_end = 1 - self.ds_duration_phase / 2
        right_in_swing = (self.phase > right_swing_start) & (self.phase < right_swing_end)
        phase_right = torch.where(
            right_in_swing,
            (self.phase - right_swing_start) / self.ss_duration_phase,  # normalize to 0-1 during swing
            torch.zeros_like(self.phase),  # 0 during stance/double support
        )

        return torch.stack([phase_left, phase_right], dim=-1)


class PhaseCommandFLT(PhaseCommandSSP):
    def __init__(self, cfg: CommandTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

    def _define_gait_parameters(self):
        self.ss_duration_phase = self.cfg.ss_duration_phase * torch.ones(self.num_envs, device=self.device)
        self.flight_duration_phase = torch.zeros(self.num_envs, device=self.device)
        self.is_standing = torch.zeros(self.num_envs, len(self.body_ids), device=self.device, dtype=torch.bool)

    def _update_command(self) -> None:
        self.phase += self.phase_increment_per_step  # advance by an amount of a step
        self.phase = torch.fmod(self.phase, 1.0)  # regulate to [0, 1]

        threshold = 0.01
        command = self._env.command_manager.get_command(self.cfg.velocity_command_name)
        stand_mask = torch.logical_and(
            torch.linalg.norm(command[:, :2], dim=1) < threshold, torch.abs(command[:, 2]) < threshold
        )
        self.phase[stand_mask] = 0.0

        # update gait
        self._update_gait_parameters()

        self.is_standing[stand_mask] = True
        self.is_standing[~stand_mask] = False

    def _update_gait_parameters(self):
        command = self._env.command_manager.get_command(self.cfg.velocity_command_name)
        flt_ratio = torch.clamp(
            (torch.abs(command[:, 0]) - self.cfg.velocity_flight_threshold)
            / (self.cfg.velocity_flight_maximum - self.cfg.velocity_flight_threshold),
            min=0.0,
            max=1.0,
        )
        flt_duration_phase = self.cfg.max_flt_duration_phase * flt_ratio

        self.ss_duration_phase = self.cfg.ss_duration_phase - flt_duration_phase
        self.flight_duration_phase = flt_duration_phase

    @property
    def planned_swing_state(self) -> torch.Tensor:
        """
        Symmetric flight gait. Constraint: 2*ss + 2*flt = 1

        [0,          ss)        : left  swing  (left=air,  right=ground)
        [ss,         ss+flt)    : flight        (both=air)
        [ss+flt,     2*ss+flt)  : right swing  (right=air, left=ground)
        [2*ss+flt,   1)         : flight        (both=air)

        phase=0 → left foot in air (left swing starts)
        """
        flt = self.flight_duration_phase
        ss = self.ss_duration_phase
        # left foot in air: left-swing [0, ss) + both flight windows
        left_swing = (self.phase < ss + flt) | (self.phase >= 2 * ss + flt)
        # right foot in air: right-swing [ss+flt, 2*ss+flt) + both flight windows
        right_swing = self.phase >= ss
        return torch.stack([left_swing, right_swing], dim=-1) * ~self.is_standing

    @property
    def swing_phase(self) -> torch.Tensor:
        """
        Returns normalized phase [0, 1] within each foot's single-support swing window, 0 otherwise.
        Flight windows are excluded (no single-support phase to normalize over).

        [0,       ss)       : left  -> 0..1
        [ss+flt,  2*ss+flt) : right -> 0..1
        0 during flight windows.
        """
        flt = self.flight_duration_phase
        ss = self.ss_duration_phase

        # Left swing: [0, ss)
        left_in_swing = self.phase < ss
        phase_left = torch.where(
            left_in_swing,
            self.phase / ss,
            torch.zeros_like(self.phase),
        )

        # Right swing: [ss+flt, 2*ss+flt)
        right_swing_start = ss + flt
        right_swing_end = 2 * ss + flt
        right_in_swing = (self.phase >= right_swing_start) & (self.phase < right_swing_end)
        phase_right = torch.where(
            right_in_swing,
            (self.phase - right_swing_start) / ss,
            torch.zeros_like(self.phase),
        )

        return torch.stack([phase_left, phase_right], dim=-1) * ~self.is_standing
