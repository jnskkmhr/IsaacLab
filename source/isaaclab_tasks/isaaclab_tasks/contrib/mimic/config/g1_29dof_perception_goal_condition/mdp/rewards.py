# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Reference rewards for imitation episodes and spatial rewards for task episodes."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

import isaaclab.envs.mdp as mdp
from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

import isaaclab_tasks.contrib.mimic.mdp as mimic_mdp

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def imitation_reward(
    env: ManagerBasedRLEnv, command_name: str, reward_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    """Apply an existing motion tracking reward only to imitation episodes."""
    command = env.command_manager.get_term(command_name)
    reward_func = getattr(mimic_mdp, reward_name)
    params = {"command_name": command_name, "std": std}
    if body_names is not None:
        params["body_names"] = body_names
    return reward_func(env, **params) * command.is_imitation


class TaskMotionStyle(ManagerTermBase):
    """Match local limb motion during task episodes, independent of world pose.

    A separate reference phase advances monotonically at one of ``phase_rates``
    times the recorded rate. Select the next phase using joint position/velocity
    and root-relative ankle/wrist position errors. This permits bounded timing
    variation without independently jumping to a matching frame each step.
    Neither this phase nor the reference features are added to actor observations.

    Stationary reference frames earn reward only at the goal position. Once there,
    a stationary final frame supplies the standing target. Otherwise the phase
    clamps at the clip end; this finite-clip objective does not synthesize new steps.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.command = env.command_manager.get_term(cfg.params["command_name"])
        body_names = cfg.params["body_names"]
        self.body_ids = [self.command.robot.body_names.index(name) for name in body_names]
        motion_body_ids = [self.command.motion.body_names.index(name) for name in body_names]
        motion = self.command.motion
        root_id = self.command.motion_root_body_index
        root_yaw = yaw_quat(motion._body_quat_w[:, root_id])
        self.reference_body_pos = quat_apply_inverse(
            root_yaw[:, None, :].expand(-1, len(body_names), -1),
            motion._body_pos_w[:, motion_body_ids] - motion._body_pos_w[:, root_id, None],
        )
        phase_rates = cfg.params["phase_rates"]
        if not phase_rates or any(rate <= 0 for rate in phase_rates):
            raise ValueError("Style phase_rates must contain positive playback rates.")
        for name in ("joint_pos_std", "joint_vel_std", "body_pos_std"):
            if cfg.params[name] <= 0:
                raise ValueError(f"Style {name} must be positive.")
        self.phase_rates = torch.tensor(phase_rates, device=self.device)
        self.phase = torch.zeros(self.num_envs, device=self.device)
        self._initialized = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._last_step = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        self._reward = torch.zeros(self.num_envs, device=self.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Initialize phase from the new command on its first reward evaluation."""
        if env_ids is None:
            env_ids = slice(None)
        self._initialized[env_ids] = False
        self._last_step[env_ids] = -1
        self._reward[env_ids] = 0.0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        body_names: list[str],
        joint_pos_std: float,
        joint_vel_std: float,
        body_pos_std: float,
        phase_rates: tuple[float, ...],
        stationary_velocity_threshold: float,
    ) -> torch.Tensor:
        """Return a task-only style score in [0, 1], updated once per control step."""
        command = self.command
        motion = command.motion
        data = command.robot.data
        phase = torch.where(self._initialized, self.phase, command.time_steps.float())
        last_frame = motion.lengths[command.motion_ids] - 1
        candidates = torch.minimum(phase[:, None] + self.phase_rates * (env.step_dt * motion.fps), last_frame[:, None])
        final_indices = motion.frame_indices(command.motion_ids, last_frame)
        final_stationary = motion.joint_vel[final_indices].square().mean(dim=-1) < stationary_velocity_threshold**2
        settle = command.position_reached & final_stationary
        candidates = torch.where(settle[:, None], last_frame[:, None], candidates)

        lower = candidates.long()
        upper = torch.minimum(lower + 1, last_frame[:, None])
        alpha = (candidates - lower)[..., None]
        lower = motion.frame_indices(command.motion_ids[:, None], lower)
        upper = motion.frame_indices(command.motion_ids[:, None], upper)
        joint_pos = torch.lerp(motion.joint_pos[lower], motion.joint_pos[upper], alpha)
        joint_vel = torch.lerp(motion.joint_vel[lower], motion.joint_vel[upper], alpha)
        body_pos = torch.lerp(self.reference_body_pos[lower], self.reference_body_pos[upper], alpha[..., None])
        robot_body_pos = quat_apply_inverse(
            yaw_quat(data.root_quat_w.torch)[:, None, :].expand(-1, len(self.body_ids), -1),
            data.body_pos_w.torch[:, self.body_ids] - data.root_pos_w.torch[:, None],
        )
        # Scale reference velocity consistently with each candidate playback rate.
        error = (data.joint_pos.torch[:, None] - joint_pos).square().mean(dim=-1) / joint_pos_std**2
        error += (data.joint_vel.torch[:, None] - joint_vel * self.phase_rates[None, :, None]).square().mean(
            dim=-1
        ) / joint_vel_std**2
        error += (robot_body_pos[:, None] - body_pos).square().sum(dim=-1).mean(dim=-1) / body_pos_std**2
        best = error.argmin(dim=-1)
        next_phase = candidates.gather(1, best[:, None]).squeeze(1)
        moving = joint_vel.square().mean(dim=-1).gather(1, best[:, None]).squeeze(1)
        eligible = (moving >= stationary_velocity_threshold**2) | command.position_reached
        reward = torch.exp(-error.gather(1, best[:, None]).squeeze(1)) * eligible * ~command.is_imitation
        update = self._last_step != env.common_step_counter
        self.phase[:] = torch.where(update, next_phase, self.phase)
        self._reward[:] = torch.where(update, reward, self._reward)
        self._last_step[:] = env.common_step_counter
        self._initialized[:] = True
        return self._reward.clone()


def goal_progress(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Signed change in root-to-destination distance per second in task episodes."""
    command = env.command_manager.get_term("motion")
    progress = (command.previous_goal_distance - command.goal_error.norm(dim=-1)) / env.step_dt
    return progress * ~command.is_imitation


def goal_position(env: ManagerBasedRLEnv, std: float) -> torch.Tensor:
    """Smooth destination position reward, independent of the reference clock."""
    command = env.command_manager.get_term("motion")
    return torch.exp(-command.goal_error.square().sum(dim=-1) / std**2) * ~command.is_imitation


def goal_heading(env: ManagerBasedRLEnv, std: float) -> torch.Tensor:
    """Reward the final yaw once the root is within the destination position tolerances."""
    command = env.command_manager.get_term("motion")
    return torch.exp(-command.heading_error.square() / std**2) * command.position_reached * ~command.is_imitation


def goal_settle(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Discourage base motion near the destination, without penalizing the airborne jump."""
    command = env.command_manager.get_term("motion")
    velocity = command.robot.data.root_lin_vel_w.torch.square().sum(dim=-1)
    velocity += command.robot.data.root_ang_vel_w.torch.square().sum(dim=-1)
    return velocity * command.position_reached * ~command.is_imitation


def goal_joint_deviation_l1(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize deviation from the default joint pose while supported at the goal.

    Use the command's position tolerances and foot-contact threshold. The penalty
    applies only to task episodes and does not constrain imitation, approach, or
    airborne motion. The target is the asset's default pose, not the last motion frame.
    """
    command = env.command_manager.get_term(command_name)
    forces = env.scene["contact_forces"].data.net_forces_w[:, command.foot_ids]
    supported = (forces.norm(dim=-1) > command.cfg.contact_threshold).any(dim=-1)
    standing = command.position_reached & supported & ~command.is_imitation
    return mdp.joint_deviation_l1(env, asset_cfg) * standing
