# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Gait rewards driven by the foot contact against the granular bed.

Contact comes from the reaction wrench that the coupler feeds back from the MPM entry to the
proxy feet, so these are the granular counterparts of the hybrid soft-contact rewards of
``g1_29dof_soft``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg
from isaaclab.utils.math import euler_xyz_from_quat, quat_apply, quat_apply_inverse, yaw_quat

from isaaclab_tasks.contrib.velocity.config import vel_mdp

if TYPE_CHECKING:
    from ..mpm_env import G1MovingPatchEnv


def foot_clearance_reward(
    env: G1MovingPatchEnv,
    target_height: float,
    std: float,
    tanh_mult: float,
    asset_cfg: SceneEntityCfg,
    standing_position_foot_z: float = 0.039,
    height_sensor_cfg: SceneEntityCfg | None = None,
    ground_height_offset: float = 0.0,
) -> torch.Tensor:
    """Reward foot clearance with the sand-depth correction only in MPM environments.

    Arguments match :func:`vel_mdp.foot_clearance_reward`. The scanner sees the
    supporting floor in sand environments and the walking surface in rigid ones.
    Mixed terrain uses its sampled MPM environment mask; the sand-only task applies
    the correction to every environment.
    """
    offset = ground_height_offset
    mpm_env_mask = getattr(env.scene.terrain, "mpm_env_mask", None)
    if mpm_env_mask is not None:
        offset = mpm_env_mask.unsqueeze(1) * ground_height_offset
    return vel_mdp.foot_clearance_reward(
        env,
        target_height=target_height,
        std=std,
        tanh_mult=tanh_mult,
        asset_cfg=asset_cfg,
        standing_position_foot_z=standing_position_foot_z,
        height_sensor_cfg=height_sensor_cfg,
        ground_height_offset=offset,
    )


def reward_soft_landing(
    env: G1MovingPatchEnv,
    command_name: str | None = "base_velocity",
    command_threshold: float = 0.05,
) -> torch.Tensor:
    """Penalize foot force magnitude at touchdown using the combined terrain contact forces.

    The cost is in newtons, not impulse. Persistent contacts incur no cost.
    A zero motion command disables the penalty, matching the soft-contact task.
    """
    first_contact = env.foot_first_contact
    landing_force = torch.linalg.vector_norm(env.foot_contact_force, dim=-1) * first_contact
    cost = landing_force.sum(dim=-1)
    env.extras["log"]["Metrics/landing_force_mean"] = landing_force.sum() / first_contact.sum().clamp(min=1)
    if command_name is not None:
        command = env.command_manager.get_command(command_name)
        active = torch.linalg.vector_norm(command[:, :2], dim=-1) + command[:, 2].abs() > command_threshold
        cost = cost * active
    return cost


def feet_air_time_positive_biped(
    env: G1MovingPatchEnv,
    command_name: str,
    threshold: float,
    velocity_threshold: float = 0.05,
) -> torch.Tensor:
    """Reward long single-stance phases while a motion command is active.

    Args:
        env: Environment instance.
        command_name: Name of the velocity command term.
        threshold: Upper bound on the rewarded stance or swing duration [s].
        velocity_threshold: Command magnitude above which the reward applies [m/s or rad/s].

    Returns:
        The reward, shape ``(num_envs,)``.
    """
    air_time = env.foot_air_time
    contact_time = env.foot_contact_time
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no gait is asked of a standing robot
    command = env.command_manager.get_command(command_name)
    commanded = (torch.norm(command[:, :2], dim=1) > velocity_threshold) | (
        torch.abs(command[:, 2]) > velocity_threshold
    )
    return reward * commanded


def break_contact_penalty(
    env: G1MovingPatchEnv,
    command_name: str,
    command_threshold: float = 0.05,
    recovery_linear_velocity: float = 0.2,
    recovery_angular_velocity: float = 0.5,
    recovery_tilt: float = 0.2,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize unsupported feet while standing on sand or rigid ground, allowing recovery."""
    asset = env.scene[asset_cfg.name]
    return vel_mdp.compute_standing_contact_penalty(
        foot_contact=env.foot_contact,
        velocity_command=env.command_manager.get_command(command_name),
        base_lin_vel_b=asset.data.root_lin_vel_b.torch,
        base_ang_vel_b=asset.data.root_ang_vel_b.torch,
        projected_gravity_b=asset.data.projected_gravity_b.torch,
        command_threshold=command_threshold,
        recovery_linear_velocity=recovery_linear_velocity,
        recovery_angular_velocity=recovery_angular_velocity,
        recovery_tilt=recovery_tilt,
    )


def no_fly(
    env: G1MovingPatchEnv,
    command_name: str,
    velocity_threshold: float = 1.0,
) -> torch.Tensor:
    """Penalize flight phases below a command speed at which running is acceptable.

    Args:
        env: Environment instance.
        command_name: Name of the velocity command term.
        velocity_threshold: Command speed above which flight is no longer penalized [m/s].

    Returns:
        The penalty, shape ``(num_envs,)``.
    """
    airborne = torch.sum(env.foot_contact, dim=-1) < 0.5
    command_speed = torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1)
    return airborne.float() * (command_speed < velocity_threshold).float()


def feet_trailing_distance_l2(
    env: G1MovingPatchEnv,
    max_trailing_distance: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize contacting feet farther behind the pelvis than the allowed distance.

    Measure foot body origins relative to the robot root along its horizontal
    forward axis (yaw only). The allowance is in meters; the returned cost is
    the sum of squared excess distances in square meters. Swing feet contribute
    zero. Select foot bodies in the environment's contact-buffer order. Contact
    includes both rigid and MPM support in mixed-terrain environments.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    foot_offset_w = asset.data.body_pos_w.torch[:, asset_cfg.body_ids] - asset.data.root_pos_w.torch[:, None, :]
    root_yaw = yaw_quat(asset.data.root_quat_w.torch)
    foot_offset_b = quat_apply_inverse(root_yaw[:, None, :].expand(-1, foot_offset_w.shape[1], -1), foot_offset_w)
    excess_distance = torch.clamp(-foot_offset_b[..., 0] - max_trailing_distance, min=0.0)
    return torch.where(env.foot_contact.bool(), excess_distance.square(), 0.0).sum(dim=-1)


def feet_pitch_contact(
    env: G1MovingPatchEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize foot pitch at touchdown only, to encourage flat landings.

    Args:
        env: Environment instance.
        asset_cfg: Configuration selecting the foot bodies, in the environment's foot order.

    Returns:
        The penalty, shape ``(num_envs,)``.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    body_quat = asset.data.body_quat_w.torch[:, asset_cfg.body_ids, :]
    _, pitch, _ = euler_xyz_from_quat(body_quat.reshape(-1, 4))
    pitch = pitch.reshape(body_quat.shape[0], body_quat.shape[1])
    return torch.sum(torch.square(pitch) * env.foot_first_contact, dim=-1)


class stance_foot_angle_penalty(ManagerTermBase):
    """Penalize foot pitch relative to the local terrain throughout contact.

    Fit a plane to the existing height scanner's world-space hits and project its
    slope along each foot's horizontal forward direction. Both feet use the same
    terrain estimate; this reward performs no additional raycasts.

    Sum squared angular error [rad²] outside ``angle_tolerance`` over contacting
    feet, including touchdown and continued stance. Swing feet contribute zero.
    The foot body's +X axis must point toward its toes, with its sole parallel to
    its XY plane. Policy observations are unchanged.

    Ignore non-finite hits. Fewer than three non-collinear hits or a non-finite
    foot orientation contribute zero instead of a NaN reward.
    """

    def __init__(self, cfg: RewardTermCfg, env: G1MovingPatchEnv):
        super().__init__(cfg, env)
        asset_cfg = cfg.params["asset_cfg"]
        self._asset: Articulation = env.scene[asset_cfg.name]
        tracked_ids, _ = self._asset.find_bodies(env.cfg.foot_body_expr, preserve_order=True)
        selected_ids = (
            list(range(self._asset.num_bodies))[asset_cfg.body_ids]
            if isinstance(asset_cfg.body_ids, slice)
            else asset_cfg.body_ids
        )
        self._contact_ids = torch.tensor(
            [tracked_ids.index(body_id) for body_id in selected_ids], dtype=torch.long, device=env.device
        )
        self._forward = torch.tensor([1.0, 0.0, 0.0], device=env.device)
        if cfg.params["angle_tolerance"] < 0.0:
            raise ValueError("angle_tolerance must be non-negative.")

    def __call__(
        self,
        env: G1MovingPatchEnv,
        asset_cfg: SceneEntityCfg,
        height_sensor_cfg: SceneEntityCfg,
        angle_tolerance: float,
    ) -> torch.Tensor:
        """Penalize every contacting foot, including continued stance."""
        penalty = self._compute_penalty(env, asset_cfg, height_sensor_cfg, angle_tolerance)
        contact = env.foot_contact[:, self._contact_ids] > 0.0
        return torch.where(contact, penalty, 0.0).sum(dim=-1)

    def _compute_penalty(
        self,
        env: G1MovingPatchEnv,
        asset_cfg: SceneEntityCfg,
        height_sensor_cfg: SceneEntityCfg,
        angle_tolerance: float,
    ) -> torch.Tensor:
        """Compute squared terrain-relative pitch errors for the selected feet."""
        sensor = env.scene[height_sensor_cfg.name]
        hits = sensor.data.ray_hits_w.torch
        valid_hits = torch.isfinite(hits).all(dim=-1, keepdim=True)
        num_hits = valid_hits.sum(dim=1, keepdim=True)
        points = torch.where(valid_hits, hits, 0.0)
        mean = points.sum(dim=1, keepdim=True) / num_hits.clamp_min(1)
        centered = torch.where(valid_hits, points - mean, 0.0)
        covariance = centered.transpose(1, 2) @ centered
        # Least-squares fit z = a*x + b*y + c, after removing the centroid.
        xx, xy, xz = covariance[:, 0].unbind(dim=-1)
        yy, yz = covariance[:, 1, 1], covariance[:, 1, 2]
        determinant = xx * yy - xy.square()
        valid_plane = (num_hits[:, 0, 0] >= 3) & (determinant > (1.0e-6 * xx * yy).clamp_min(1.0e-12))
        denominator = determinant.clamp_min(1.0e-12)
        slope = torch.stack(((yy * xz - xy * yz) / denominator, (xx * yz - xy * xz) / denominator), dim=-1)

        foot_quat = self._asset.data.body_quat_w.torch[:, asset_cfg.body_ids]
        forward = quat_apply(foot_quat, self._forward.expand_as(foot_quat[..., :3]))
        horizontal_length = torch.linalg.vector_norm(forward[..., :2], dim=-1)
        heading = forward[..., :2] / horizontal_length.clamp_min(1.0e-6).unsqueeze(-1)
        terrain_angle = torch.atan((slope.unsqueeze(1) * heading).sum(dim=-1))
        foot_angle = torch.atan2(forward[..., 2], horizontal_length)
        error = (foot_angle - terrain_angle).abs()
        valid = valid_plane.unsqueeze(-1) & torch.isfinite(error)
        penalty = (error - angle_tolerance).clamp_min(0.0).square()
        return torch.where(valid, penalty, 0.0)


class foot_touch_down_angle_penalty(stance_foot_angle_penalty):
    """Apply the terrain-relative foot pitch penalty only at first contact."""

    def __call__(
        self,
        env: G1MovingPatchEnv,
        asset_cfg: SceneEntityCfg,
        height_sensor_cfg: SceneEntityCfg,
        angle_tolerance: float,
    ) -> torch.Tensor:
        """Penalize touchdown; exclude continued stance and swing."""
        penalty = self._compute_penalty(env, asset_cfg, height_sensor_cfg, angle_tolerance)
        touchdown = env.foot_first_contact[:, self._contact_ids] > 0.0
        return torch.where(touchdown, penalty, 0.0).sum(dim=-1)


def metric_sliderbar(
    env: G1MovingPatchEnv,
    obs_term_names: list[str],
    obs_group_name: str = "privileged",
) -> torch.Tensor:
    """Log current observation statistics without contributing to the reward.

    The selected functions are evaluated before reset, without manager noise, modifiers,
    scaling, clipping, delay, or history updates. Use state-reading observation functions;
    stateful observation terms would be advanced by this additional evaluation.
    ``G1MovingPatchEnv.step`` forwards the metrics to ``extras["log"]`` after resets,
    which RSL-RL consumes through its existing logger.

    Each flattened component logs its finite-value mean, maximum absolute finite value,
    and nonfinite fraction across environments. All-invalid components report zero for
    the first two statistics and one for the nonfinite fraction. Physical units match
    those of the selected observation function. No observation values are modified.

    Args:
        env: Moving-patch environment instance.
        obs_term_names: Observation term names within the selected group.
        obs_group_name: Observation group to inspect. Defaults to ``"privileged"``.

    Returns:
        Zeros, shape ``(num_envs,)``. Configure a nonzero reward weight so the reward
        manager invokes this diagnostic term.
    """
    manager = env.observation_manager
    if obs_group_name not in manager.active_terms:
        raise ValueError(f"Unknown observation group: {obs_group_name!r}.")
    group_cfg = manager.cfg[obs_group_name] if isinstance(manager.cfg, dict) else getattr(manager.cfg, obs_group_name)
    metrics = env.extras.setdefault("_observation_metrics", {})
    for name in obs_term_names:
        if name not in manager.active_terms[obs_group_name]:
            raise ValueError(f"Unknown observation term: {obs_group_name}/{name}.")
        term_cfg = group_cfg[name] if isinstance(group_cfg, dict) else getattr(group_cfg, name)
        values = term_cfg.func(env, **term_cfg.params).detach().reshape(env.num_envs, -1)
        finite = torch.isfinite(values)
        safe_values = torch.where(finite, values, 0.0)
        means = safe_values.sum(dim=0) / finite.sum(dim=0).clamp_min(1)
        maxima = safe_values.abs().amax(dim=0)
        nonfinite = (~finite).float().mean(dim=0)
        prefix = f"Metrics/observations/{obs_group_name}/{name}"
        for index in range(values.shape[1]):
            metrics[f"{prefix}/mean_{index}"] = means[index]
            metrics[f"{prefix}/abs_max_{index}"] = maxima[index]
            metrics[f"{prefix}/nonfinite_fraction_{index}"] = nonfinite[index]
    return torch.zeros(env.num_envs, device=env.device)
