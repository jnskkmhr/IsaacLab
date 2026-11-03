# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Agile G1 reward formulas, kept local to this task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.utils import math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def command_tracking_width(
    command_xy: torch.Tensor, std: float, max_std: float, threshold: float = 0.5, full_speed: float = 1.0
) -> torch.Tensor:
    """Quadrature-interpolated isotropic width, saturated at full_speed, in m/s."""
    magnitude = torch.linalg.vector_norm(command_xy, dim=-1, keepdim=True)
    progress = ((magnitude - threshold) / (full_speed - threshold)).clamp(0.0, 1.0)
    return (std**2 + (max_std**2 - std**2) * progress.square()).sqrt()


def tracking_reward(
    command_xy: torch.Tensor,
    actual_xy: torch.Tensor,
    std: float,
    max_std: float,
    threshold: float = 0.5,
    full_speed: float = 1.0,
) -> torch.Tensor:
    width = command_tracking_width(command_xy, std, max_std, threshold, full_speed)
    error = (command_xy - actual_xy).square()
    return torch.exp(-error.sum(dim=-1) / width.squeeze(-1).square())


def track_lin_vel_xy_command_width(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    max_std: float,
    threshold: float = 0.5,
    full_speed: float = 1.0,
    asset_name: str = "robot",
) -> torch.Tensor:
    """Match Isaac Lab's native body-frame XY error; vary only kernel width."""
    command = env.command_manager.get_command(command_name)[:, :2]
    actual = env.scene[asset_name].data.root_lin_vel_b.torch[:, :2]
    return tracking_reward(command, actual, std, max_std, threshold, full_speed)


def heading(quat: torch.Tensor) -> torch.Tensor:
    """Extract yaw from Isaac Lab 3.0's XYZW quaternions."""
    x, y, z, w = quat.unbind(-1)
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def signed_clearance_cost(positions: torch.Tensor, root_yaw: torch.Tensor, minimum: float = 0.10) -> torch.Tensor:
    """Cost in [0, 3] below minimum left-minus-right ankle separation in the root yaw frame.

    Positions must have shape (environments, 2, 3), ordered left then right.
    Forward/vertical separation does not mask crossing; wider stances have zero cost.
    """
    delta = positions[:, 0, :2] - positions[:, 1, :2]
    lateral = -root_yaw.sin() * delta[:, 0] + root_yaw.cos() * delta[:, 1]
    return ((minimum - lateral) / minimum).clamp(0, 3)


def feet_signed_clearance(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, minimum: float = 0.10) -> torch.Tensor:
    """Read left/right ankle link origins, preserving the configured body order."""
    data = env.scene[asset_cfg.name].data
    return signed_clearance_cost(
        data.body_link_pos_w.torch[:, asset_cfg.body_ids], heading(data.root_link_quat_w.torch), minimum
    )


def interval_loading_cost(force_history: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Mean squared positive force increments, normalized by 100 N.

    History is newest first. Four samples cover the current 50 Hz control
    interval, and a fifth supplies its preceding boundary at 200 Hz physics.
    Both feet contribute; negative vertical forces and unloading are ignored.
    The first transition after reset is excluded, as in the deployed recipe.
    """
    if force_history.shape != (valid.shape[0], 5, 2, 3):
        raise ValueError("Loading cost requires five ordered two-foot force samples")
    vertical = force_history[..., 2].clamp_min(0)
    increments = (vertical[:, :4] - vertical[:, 1:5]).clamp_min(0)
    return (increments / 100.0).square().sum(dim=-1).mean(dim=-1) * valid


class SmoothContactObserver(ManagerTermBase):
    """Compute the unweighted loading cost before reward weighting/autoreset."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv) -> None:
        super().__init__(cfg, env)
        if abs(env.step_dt - 0.02) > 1e-9 or abs(env.physics_dt - 0.005) > 1e-9:
            raise ValueError("Loading cost requires 50 Hz control and 200 Hz physics")
        if getattr(env, "_smooth_contact_observer", None) is not None:
            raise ValueError("Only one loading observer is allowed")
        env._smooth_contact_observer = self
        self.valid = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.zero = torch.zeros(env.num_envs, device=env.device)
        self.costs = {"interval_loading": self.zero}
        self.last_step: int | None = None

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        self.valid[slice(None) if env_ids is None else env_ids] = False

    def __call__(self, env: ManagerBasedRLEnv, interval_sensor_cfg: SceneEntityCfg) -> torch.Tensor:
        step = int(env.common_step_counter)
        if self.last_step == step:
            raise ValueError("Loading observer was evaluated twice for one transition")
        sensor = env.scene.sensors[interval_sensor_cfg.name]
        history = sensor.data.net_forces_w_history.torch[:, :, interval_sensor_cfg.body_ids]
        self.costs["interval_loading"] = interval_loading_cost(history, self.valid)
        self.valid.fill_(True)
        self.last_step = step
        return self.zero


def smooth_contact_cost(env: ManagerBasedRLEnv, mode: str) -> torch.Tensor:
    observer = env._smooth_contact_observer
    if observer.last_step != int(env.common_step_counter):
        raise ValueError("Loading observer must precede the loading reward term")
    return observer.costs[mode]


def base_height_exp(
    env: ManagerBasedRLEnv,
    target_height: float,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Reward for tracking the target base height with an exponential kernel.

    Note:
        For flat terrain, target height is in the world frame. For rough terrain,
        sensor readings can adjust the target height to account for the terrain.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        # Adjust the target height using the sensor data
        adjusted_target_height = target_height + torch.mean(sensor.data.ray_hits_w.torch[..., 2], dim=1)
    else:
        # Use the provided target height directly for flat terrain
        adjusted_target_height = target_height
    height_error = torch.square(asset.data.root_pos_w.torch[:, 2] - adjusted_target_height)
    return torch.exp(-height_error / std**2)


def action_rate_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:  # noqa: ARG001
    """Penalize the rate of change of the actions using L2 squared kernel."""
    return torch.sum(
        torch.square(env.action_manager.action - env.action_manager.prev_action),
        dim=1,
    )


class action_rate_rate_l2(ManagerTermBase):
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.prev_action_rate = torch.zeros_like(env.action_manager.action)

    def __call__(self, env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:  # noqa: ARG002
        """Penalize the rate of change of the rate of change of the actions using L2 squared kernel."""
        action_rate = env.action_manager.action - env.action_manager.prev_action
        action_rate_rate = torch.sum(torch.square(action_rate - self.prev_action_rate), dim=1)
        self.prev_action_rate = action_rate
        return action_rate_rate

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self._env.num_envs)
        self.prev_action_rate[env_ids] = 0


class body_acc_l2(ManagerTermBase):
    """Penalize body linear and angular accelerations using velocity history tracking (Isaac Gym style).

    This reward term computes world-frame accelerations for a specified body/link.
    If no body_names is specified in asset_cfg, it defaults to using the root link.

    Usage:
        # For root acceleration (default):
        body_acc = RewTerm(func=body_acc_l2, weight=-0.01)

        # For a specific link:
        torso_acc = RewTerm(
            func=body_acc_l2,
            weight=-0.01,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=["torso_link"])},
        )
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        # Initialize the base class
        super().__init__(cfg, env)

        # Initialize velocity history buffer
        # Shape: [num_envs, 6] where 6 = 3 (lin_vel) + 3 (ang_vel)
        self.prev_body_vel = torch.zeros(env.num_envs, 6, device=env.device, dtype=torch.float32)

        # Flag to track if this is the first call (skip acceleration computation)
        self.first_call = True

        # Resolve body index if body_names is provided
        self._body_idx: int | None = None
        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        if asset_cfg.body_names is not None:
            asset: Articulation = env.scene[asset_cfg.name]
            self._body_idx = asset.find_bodies(asset_cfg.body_names)[0][0]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        """Compute body acceleration penalty by tracking velocity changes in world frame."""

        # Extract the robot asset
        robot: Articulation = env.scene[asset_cfg.name]

        # Get current velocities (both linear and angular) in world frame
        if self._body_idx is not None:
            # Use specified body velocities
            current_lin_vel = robot.data.body_lin_vel_w.torch[:, self._body_idx, :]  # [num_envs, 3]
            current_ang_vel = robot.data.body_ang_vel_w.torch[:, self._body_idx, :]  # [num_envs, 3]
        else:
            # Default to root velocities
            current_lin_vel = robot.data.root_lin_vel_w.torch  # [num_envs, 3]
            current_ang_vel = robot.data.root_ang_vel_w.torch  # [num_envs, 3]

        # Concatenate to form 6D velocity vector
        current_body_vel = torch.cat([current_lin_vel, current_ang_vel], dim=-1)  # [num_envs, 6]

        if self.first_call:
            # First call: initialize previous velocity and return zeros
            self.prev_body_vel.copy_(current_body_vel)
            self.first_call = False
            return torch.zeros(env.num_envs, device=env.device)

        # Compute acceleration as velocity difference over timestep
        body_acc = (current_body_vel - self.prev_body_vel) / env.step_dt

        # Update velocity history for next call
        self.prev_body_vel.copy_(current_body_vel)

        # Compute L2 penalty on accelerations (sum of squared accelerations)
        return torch.clamp(torch.sum(torch.square(body_acc), dim=-1), max=1e6)


def _feet_roll_l2_from_quat(feet_quat: torch.Tensor) -> torch.Tensor:
    roll, _, _ = math_utils.euler_xyz_from_quat(feet_quat.reshape(-1, 4))
    feet_roll = roll.reshape(feet_quat.shape[:2])
    feet_roll = (feet_roll + torch.pi) % (2 * torch.pi) - torch.pi
    return torch.sum(torch.square(feet_roll), dim=-1)


def _feet_yaw_diff_l2_from_quat(feet_quat: torch.Tensor) -> torch.Tensor:
    _, _, yaw = math_utils.euler_xyz_from_quat(feet_quat.reshape(-1, 4))
    feet_yaw = yaw.reshape(feet_quat.shape[:2])
    feet_yaw = (feet_yaw + torch.pi) % (2 * torch.pi) - torch.pi
    yaw_diff = (feet_yaw[:, 1] - feet_yaw[:, 0] + torch.pi) % (2 * torch.pi) - torch.pi
    return torch.square(yaw_diff)


def _feet_yaw_mean_vs_base_from_quat(feet_quat: torch.Tensor, base_quat: torch.Tensor) -> torch.Tensor:
    base_quat_inv = math_utils.quat_inv(base_quat)
    feet_quat_relative = math_utils.quat_mul(base_quat_inv.unsqueeze(1).expand_as(feet_quat), feet_quat)
    _, _, feet_yaw_relative = math_utils.euler_xyz_from_quat(feet_quat_relative.reshape(-1, 4))
    return torch.square(feet_yaw_relative.reshape(feet_quat.shape[:2])).sum(dim=1)


def _reduce_foot_yaw_penalty_for_turning(
    env: ManagerBasedRLEnv,
    penalty: torch.Tensor,
    command_name: str | None,
    ang_vel_threshold: float,
    reduction_scale: float,
) -> torch.Tensor:
    if command_name is None:
        return penalty
    ang_vel_magnitude = torch.abs(env.command_manager.get_command(command_name)[:, 2])
    scale_factor = torch.where(
        ang_vel_magnitude <= ang_vel_threshold,
        torch.ones_like(ang_vel_magnitude),
        torch.clamp(1.0 - reduction_scale * (ang_vel_magnitude - ang_vel_threshold), min=0.0),
    )
    return penalty * scale_factor


def feet_roll_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize foot roll angles (Isaac Gym style).

    Penalizes feet that are not flat (roll rotation around x-axis).
    """
    asset: Articulation = env.scene[asset_cfg.name]

    # Get feet quaternions
    feet_quat = asset.data.body_quat_w.torch[:, asset_cfg.body_ids]  # [num_envs, num_feet, 4]

    return _feet_roll_l2_from_quat(feet_quat)


def feet_yaw_diff_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str | None = None,
    ang_vel_threshold: float = 0.3,
    reduction_scale: float = 2.0,
) -> torch.Tensor:
    """Penalize yaw difference between left and right feet (Isaac Gym style).

    Encourages both feet to have similar yaw orientation, but reduces penalty
    when turning (angular velocity command is high).

    Args:
        env: The environment.
        asset_cfg: Configuration for the robot asset.
        command_name: Name of the command to get angular velocity from. If None, no reduction.
        ang_vel_threshold: Angular velocity threshold (rad/s) above which to start reducing penalty.
        reduction_scale: How strongly to reduce penalty based on angular velocity magnitude.
    """
    asset: Articulation = env.scene[asset_cfg.name]

    # Get feet quaternions
    feet_quat = asset.data.body_quat_w.torch[:, asset_cfg.body_ids]  # [num_envs, num_feet, 4]

    # Ensure we have exactly 2 feet
    if len(asset_cfg.body_ids) != 2:
        return torch.zeros(env.num_envs, device=env.device)

    penalty = _feet_yaw_diff_l2_from_quat(feet_quat)
    return _reduce_foot_yaw_penalty_for_turning(env, penalty, command_name, ang_vel_threshold, reduction_scale)


def feet_yaw_mean_vs_base(
    env: ManagerBasedRLEnv,
    feet_asset_cfg: SceneEntityCfg,
    base_body_cfg: SceneEntityCfg,
    command_name: str | None = None,
    ang_vel_threshold: float = 0.3,
    reduction_scale: float = 2.0,
) -> torch.Tensor:
    """Penalize the squared yaw of each foot relative to the base frame.

    This encourages the feet to stay rotationally aligned with the base's
    forward direction by minimizing the yaw component of their relative orientation.
    Reduces penalty when turning (angular velocity command is high).

    Args:
        env: The environment.
        feet_asset_cfg: Configuration for the feet bodies.
        base_body_cfg: Configuration for the base body.
        command_name: Name of the command to get angular velocity from. If None, no reduction.
        ang_vel_threshold: Angular velocity threshold (rad/s) above which to start reducing penalty.
        reduction_scale: How strongly to reduce penalty based on angular velocity magnitude.
    """
    asset: Articulation = env.scene[feet_asset_cfg.name]

    # Get feet quaternions and base quaternion
    feet_quat = asset.data.body_quat_w.torch[:, feet_asset_cfg.body_ids]  # [num_envs, 2, 4]
    base_quat = asset.data.body_quat_w.torch[:, base_body_cfg.body_ids].squeeze(1)  # [num_envs, 4]

    # Ensure we have exactly 2 feet
    if len(feet_asset_cfg.body_ids) != 2:
        raise ValueError("Only two feet are supported for feet_yaw_mean_vs_base reward.")

    if len(base_body_cfg.body_ids) != 1:
        raise ValueError("Only one reference body is supported for feet_yaw_mean_vs_base reward.")

    penalty = _feet_yaw_mean_vs_base_from_quat(feet_quat, base_quat)
    return _reduce_foot_yaw_penalty_for_turning(env, penalty, command_name, ang_vel_threshold, reduction_scale)


def flat_body_orientation_exp(
    env: ManagerBasedRLEnv, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward flat orientation using exponential kernel.

    This is computed by rewarding small xy-components of the projected gravity vector.
    Args:
        std: std in rad
        asset_cfg:
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    body_quats = asset.data.body_link_quat_w.torch[:, asset_cfg.body_ids]
    # Newton exposes gravity in m/s^2; this reward's std is an angular tolerance.
    gravity_direction_w = torch.nn.functional.normalize(asset.data.GRAVITY_VEC_W.torch, dim=-1)
    gravity_vec_expanded = gravity_direction_w.unsqueeze(1).expand(-1, len(asset_cfg.body_ids), -1)
    projected_gravity = math_utils.quat_apply_inverse(body_quats, gravity_vec_expanded)
    orientation_error = torch.sum(torch.square(projected_gravity[..., :2]), dim=-1)  # sum over x,y
    orientation_error_per_env = torch.sum(orientation_error, dim=-1)  # sum over k bodies

    return torch.exp(-orientation_error_per_env / std**2)


def jumping(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize if no foot is in contact with the ground.

    Args:
        env: The environment.
        threshold: The force threshold for the jumping.
        sensor_cfg: The configuration for the foot contact sensor.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the penalty
    feet_forces = contact_sensor.data.net_forces_w.torch[:, sensor_cfg.body_ids].norm(dim=2)
    not_in_contact = feet_forces < threshold
    is_jumping = not_in_contact.all(dim=1)

    return is_jumping.float()


def feet_distance_from_ref(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, ref_distance: float = 0.2
) -> torch.Tensor:
    """Absolute error in ankle separation along the root frame's lateral axis [m]."""
    data = env.scene[asset_cfg.name].data
    positions = data.body_pos_w.torch[:, asset_cfg.body_ids] - data.root_pos_w.torch[:, None]
    positions = math_utils.quat_apply_inverse(data.root_quat_w.torch[:, None].expand(-1, 2, -1), positions)
    return ((positions[:, 0, 1] - positions[:, 1, 1]).abs() - ref_distance).abs()


def feet_slip(
    env: ManagerBasedRLEnv, contact_threshold: float, sensor_cfg: SceneEntityCfg, robot_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Sum horizontal ankle speed [m/s] when force history indicates contact."""
    sensor = env.scene.sensors[sensor_cfg.name]
    contact = (
        sensor.data.net_forces_w_history.torch[:, :, sensor_cfg.body_ids].norm(dim=-1).amax(dim=1) > contact_threshold
    )
    velocity = env.scene[robot_cfg.name].data.body_vel_w.torch[:, robot_cfg.body_ids, :2]
    return (velocity.norm(dim=-1) * contact).sum(dim=-1)
