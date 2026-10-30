# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to define rewards for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.RewardTermCfg` object to
specify the reward function and its parameters.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import RewardTermCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import euler_xyz_from_quat, quat_apply_inverse, yaw_quat
from isaaclab.utils.string import resolve_matching_names_values

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

"""
gait
"""


def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_air_time_positive_biped(
    env, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg, velocity_threshold: float = 0.05
) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    # reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    linear_norm = torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1)
    angular_norm = torch.abs(env.command_manager.get_command(command_name)[:, 2])
    total_norm = linear_norm + angular_norm
    reward *= total_norm > velocity_threshold
    return reward


def fly(
    env: ManagerBasedRLEnv,
    threshold: float,
    sensor_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    velocity_threshold: float = 1.5,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    linear_norm = torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1)
    is_active = linear_norm < velocity_threshold
    reward = torch.sum(is_contact, dim=-1) < 0.5
    return reward * is_active


"""
stance foot
"""


def feet_slide(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene[asset_cfg.name]

    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward


def foot_force(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float = 500, max_reward: float = 400
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    reward = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2].norm(dim=-1)
    reward[reward < threshold] = 0
    reward[reward > threshold] -= threshold
    reward = reward.clamp(min=0, max=max_reward)
    return reward


def reward_foot_distance(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, ref_dist: float) -> torch.Tensor:
    """
    Calculates the reward based on the distance between the feet. Penalize feet get close to each other or too far away.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :3]
    foot_dist = torch.norm(foot_pos[:, 0, :] - foot_pos[:, 1, :], dim=1)

    reward = torch.clip(ref_dist - foot_dist, min=0.0, max=0.1)

    return reward


def reward_foot_lateral_symmetry(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    ref_dist: float,
) -> torch.Tensor:
    """L2 penalty for lateral foot separation deviating from reference."""
    asset: RigidObject = env.scene[asset_cfg.name]
    body_pos_in_base = torch.zeros_like(asset.data.body_pos_w[:, asset_cfg.body_ids, :3])
    for i in range(len(asset_cfg.body_ids)):
        body_pos_in_base[:, i, :] = quat_apply_inverse(
            asset.data.root_quat_w,
            asset.data.body_pos_w[:, asset_cfg.body_ids, :3][:, i, :] - asset.data.root_pos_w,
        )
    foot_lat_dist = torch.abs(body_pos_in_base[:, 0, 1] - body_pos_in_base[:, 1, 1])
    return torch.square(foot_lat_dist - ref_dist)


def reward_soft_landing(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str,
    command_threshold: float = 0.05,
) -> torch.Tensor:
    """Penalize high impact forces at landing to encourage soft footfalls."""
    contact_sensor: ContactSensor = env.scene[sensor_cfg.name]
    assert contact_sensor.data.net_forces_w is not None

    forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]  # [B, N, 3]
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]  # [B, N]

    force_magnitude = torch.norm(forces, dim=-1)  # [B, N]
    landing_impact = force_magnitude * first_contact.float()  # [B, N]
    cost = torch.sum(landing_impact, dim=1)  # [B]

    num_landings = torch.sum(first_contact.float())
    mean_landing_force = torch.sum(landing_impact) / torch.clamp(num_landings, min=1)
    env.extras["log"]["Metrics/landing_force_mean"] = mean_landing_force

    if command_name is not None:
        command = env.command_manager.get_command(command_name)
        linear_norm = torch.norm(command[:, :2], dim=1)
        angular_norm = torch.abs(command[:, 2])
        total_command = linear_norm + angular_norm
        active = (total_command > command_threshold).float()
        cost = cost * active
    return cost


"""
swing foot
"""


def foot_clearance_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    std: float,
    tanh_mult: float,
    standing_position_foot_z: float = 0.039,
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground, weighted by foot velocity."""
    asset: RigidObject = env.scene[asset_cfg.name]  # type: ignore
    foot_z_target_error = torch.square(
        asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - (target_height + standing_position_foot_z)
    )
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2))
    reward = foot_z_target_error * foot_velocity_tanh
    reward = torch.exp(-torch.sum(reward, dim=1) / std)
    return reward


@torch.compile
def bezier_curve(control_points, t):
    """
    Computes Bézier curve for given control points and parameter t.

    control_points: shape (n, num_envs, dim)
    t: shape (num_envs,)

    curve_point: (num_envs, dim)
    """
    n = len(control_points) - 1  # Degree of the Bézier curve
    dim = control_points.shape[2]  # Dimension of control points
    curve_points = torch.zeros((t.shape[0], dim), dtype=control_points.dtype, device=t.device)  # (num_envs, dim)

    # Calculate the Bézier curve points
    for k in range(n + 1):
        binomial_coeff = math.comb(n, k)
        bernstein_poly = binomial_coeff * (t**k) * ((1 - t) ** (n - k))  # (num_envs, )
        curve_points += (
            bernstein_poly.unsqueeze(1) * control_points[k]
        )  # (num_envs, 1) * (num_envs, dim) = (num_envs, dim)

    return curve_points


def track_foot_height_bezier(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    std: float,
    tanh_mult: float,
    standing_position_foot_z: float = 0.039,
    gait_command_name: str = "locomotion_gait",
    target_height_command_name: str = "foot_height",
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground, weighted by foot velocity."""
    asset: RigidObject = env.scene[asset_cfg.name]  # type: ignore

    gait_generator = env.command_manager.get_term(gait_command_name)

    phase_legs = gait_generator.swing_phase  # (num_envs, 2)

    peak_height = env.command_manager.get_term(target_height_command_name).command  # type: ignore
    height_scale = torch.tanh(tanh_mult * torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2))
    # height_scale = torch.ones_like(torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2))
    velocity_adaptive_target_height = height_scale * peak_height.unsqueeze(-1)  # (num_envs, 2)
    bezier_target_height = torch.zeros_like(velocity_adaptive_target_height)

    for i in range(2):
        # cubic bezier curve points where apex height is achieved at t=0.5
        control_points = torch.stack(
            [
                torch.zeros_like(velocity_adaptive_target_height[:, i].unsqueeze(-1)),
                8 / 6 * velocity_adaptive_target_height[:, i].unsqueeze(-1),
                8 / 6 * velocity_adaptive_target_height[:, i].unsqueeze(-1),
                torch.zeros_like(velocity_adaptive_target_height[:, i].unsqueeze(-1)),
            ],
            dim=0,
        )  # (4, num_envs, 1)
        t = phase_legs[:, i]  # (num_envs, )
        bezier_target_height[:, i] = bezier_curve(control_points, t)[:, 0]

    foot_z_target_error = torch.square(
        asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - (bezier_target_height + standing_position_foot_z)
    )
    return torch.exp(-torch.sum(foot_z_target_error, dim=1) / std)


"""
base tracking
"""


def track_lin_vel_xy_yaw_frame_exp(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned
    robot frame using an exponential kernel.
    """
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
    )
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_world_exp(
    env, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2])
    return torch.exp(-ang_vel_error / std**2)


def track_heading_world_exp(
    env, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of heading commands in world frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    heading_error = math_utils.quat_error_magnitude(
        math_utils.yaw_quat(asset.data.root_quat_w), env.command_manager.get_command(command_name)[:, 3:]
    )
    return torch.exp(-torch.square(heading_error) / std**2)


"""
base orientation
"""


def body_orientation_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    body_orientation = quat_apply_inverse(asset.data.body_quat_w[:, asset_cfg.body_ids[0], :], asset.data.GRAVITY_VEC_W)
    return torch.sum(torch.square(body_orientation[:, :2]), dim=1)


"""
foot orientation
"""


def penalty_feet_ori(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]  # type: ignore
    left_foot_quat = asset.data.body_quat_w[:, asset_cfg.body_ids[0], :]
    right_foot_quat = asset.data.body_quat_w[:, asset_cfg.body_ids[1], :]
    left_foot_gravity = quat_apply_inverse(left_foot_quat, asset.data.GRAVITY_VEC_W)
    right_foot_gravity = quat_apply_inverse(right_foot_quat, asset.data.GRAVITY_VEC_W)

    return torch.norm(left_foot_gravity[:, :2], dim=1) + torch.norm(right_foot_gravity[:, :2], dim=1)


def _feet_rpy(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Compute the yaw angles of feet.

    Args:
    env: The environment.
    asset_cfg: Configuration for the asset.
    feet_index: Optional list of indices specifying which feet to consider.
            If None, all bodies specified in asset_cfg.body_ids are used.

    Returns:
    torch.Tensor: Yaw angles of feet in radians.
    """
    # Get the entity
    entity = env.scene[asset_cfg.name]

    # Get the body IDs to use
    feet_quat = entity.data.body_quat_w[:, asset_cfg.body_ids, :]
    # feet_quat = entity.data.body_quat_w[:, feet_index, :]
    original_shape = feet_quat.shape
    roll, pitch, yaw = euler_xyz_from_quat(feet_quat.reshape(-1, 4))

    roll = (roll + torch.pi) % (2 * torch.pi) - torch.pi
    pitch = (pitch + torch.pi) % (2 * torch.pi) - torch.pi
    # yaw = (yaw + torch.pi) % (2*torch.pi) - torch.pi

    return roll.reshape(original_shape[0], -1), pitch.reshape(original_shape[0], -1), yaw.reshape(original_shape[0], -1)


def _base_rpy(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), base_index: list[int] = [0]):
    """Compute the yaw angles of feet.

    Args:
    env: The environment.
    asset_cfg: Configuration for the asset.
    feet_index: Optional list of indices specifying which feet to consider.
            If None, all bodies specified in asset_cfg.body_ids are used.

    Returns:
    torch.Tensor: Yaw angles of feet in radians.
    """
    # Get the entity
    entity = env.scene[asset_cfg.name]

    # Get the body IDs to use
    body_quat = entity.data.body_quat_w[:, base_index, :]
    original_shape = body_quat.shape
    roll, pitch, yaw = euler_xyz_from_quat(body_quat.reshape(-1, 4))

    return roll.reshape(original_shape[0]), pitch.reshape(original_shape[0]), yaw.reshape(original_shape[0])


def reward_feet_roll(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    # feet_index: list[int] = [22, 23]
) -> torch.Tensor:

    # Calculate roll angles from quaternions for the feet
    feet_roll, _, _ = _feet_rpy(
        env,
        asset_cfg=asset_cfg,
    )

    return torch.sum(torch.square(feet_roll), dim=-1)


def reward_feet_roll_diff(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    # feet_index: list[int] = [22, 23]):
) -> torch.Tensor:

    # Calculate pitch angles from quaternions for the feet
    feet_roll, _, _ = _feet_rpy(
        env,
        asset_cfg=asset_cfg,
    )
    roll_rel_diff = torch.abs((feet_roll[:, 1] - feet_roll[:, 0] + torch.pi) % (2 * torch.pi) - torch.pi)
    return roll_rel_diff


def reward_feet_pitch(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    # feet_index: list[int] = [22, 23]
) -> torch.Tensor:

    # Calculate roll angles from quaternions for the feet
    _, feet_pitch, _ = _feet_rpy(
        env,
        asset_cfg=asset_cfg,
    )
    return torch.sum(torch.square(feet_pitch), dim=-1)


def reward_feet_pitch_contact(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize foot pitch angle only at the moment of first contact.

    During swing the penalty is zero, so knee flexion is not penalized.
    At touchdown, pitch² is penalized to encourage flat foot landings.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]  # [B, N_feet] bool

    _, feet_pitch, _ = _feet_rpy(env, asset_cfg=asset_cfg)  # [B, N_feet]

    # penalize pitch² only on the landing step
    return torch.sum(torch.square(feet_pitch) * first_contact.float(), dim=-1)


def reward_feet_pitch_diff(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:

    # Calculate pitch angles from quaternions for the feet
    _, feet_pitch, _ = _feet_rpy(
        env,
        asset_cfg=asset_cfg,
    )
    pitch_rel_diff = torch.abs((feet_pitch[:, 1] - feet_pitch[:, 0] + torch.pi) % (2 * torch.pi) - torch.pi)
    return pitch_rel_diff


def reward_feet_yaw_diff(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward minimizing the difference between feet yaw angles.

    This function rewards the agent for having similar yaw angles for all feet,
    which encourages a more stable and coordinated gait.

    Args:
        env: The environment.
        std: Standard deviation parameter for the exponential kernel.
        asset_cfg: Configuration for the asset.

    Returns:
        torch.Tensor: Reward based on similarity of feet yaw angles.
    """

    # Calculate yaw angles from quaternions for the feet
    _, _, feet_yaw = _feet_rpy(
        env,
        asset_cfg=asset_cfg,
    )
    yaw_rel_diff = torch.abs((feet_yaw[:, 1] - feet_yaw[:, 0] + torch.pi) % (2 * torch.pi) - torch.pi)
    return yaw_rel_diff


def reward_feet_yaw_mean(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:

    # Calculate yaw angles from quaternions for the feet
    _, _, feet_yaw = _feet_rpy(
        env,
        asset_cfg=asset_cfg,
    )

    _, _, base_yaw = _base_rpy(env, asset_cfg=asset_cfg, base_index=[0])
    mean_yaw = feet_yaw.mean(dim=-1) + torch.pi * (torch.abs(feet_yaw[:, 1] - feet_yaw[:, 0]) > torch.pi)

    yaw_diff = torch.abs((base_yaw - mean_yaw + torch.pi) % (2 * torch.pi) - torch.pi)

    return yaw_diff


"""
joint regularization
"""


def stand_still_joint_deviation_l1(
    env, command_name: str, command_threshold: float = 0.06, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize offsets from the default joint positions when the command is very small."""
    command = env.command_manager.get_command(command_name)
    # Penalize motion when command is nearly zero.
    return mdp.joint_deviation_l1(env, asset_cfg) * (torch.norm(command[:, :2], dim=1) < command_threshold)


# L1 penalty
class variable_posture_l1(ManagerTermBase):
    """
    compute gaussian kernel reward to regularize robot's whole body posture for each gait.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        asset = env.scene[cfg.params["asset_cfg"].name]
        self.default_joint_pos = asset.data.default_joint_pos

        _, joint_names = asset.find_joints(cfg.params["asset_cfg"].joint_names)

        _, _, weight_standing = resolve_matching_names_values(
            data=cfg.params["weight_standing"],  # type: ignore
            list_of_strings=joint_names,
        )
        self.weight_standing = torch.tensor(weight_standing, device=env.device, dtype=torch.float32)

        _, _, weight_walking = resolve_matching_names_values(
            data=cfg.params["weight_walking"],  # type: ignore
            list_of_strings=joint_names,
        )
        self.weight_walking = torch.tensor(weight_walking, device=env.device, dtype=torch.float32)

        _, _, weight_running = resolve_matching_names_values(
            data=cfg.params["weight_running"],  # type: ignore
            list_of_strings=joint_names,
        )
        self.weight_running = torch.tensor(weight_running, device=env.device, dtype=torch.float32)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
        command_name: str,
        weight_standing: dict,
        weight_walking: dict,
        weight_running: dict,
        walking_threshold: float = 0.5,
        running_threshold: float = 1.5,
    ) -> torch.Tensor:

        asset = env.scene[asset_cfg.name]
        command = env.command_manager.get_command(command_name)

        linear_speed = torch.norm(command[:, :2], dim=-1)
        angular_speed = torch.abs(command[:, 2])
        total_speed = linear_speed + angular_speed

        standing_mask = (total_speed < walking_threshold).float()
        walking_mask = ((total_speed >= walking_threshold) & (total_speed < running_threshold)).float()
        running_mask = (total_speed >= running_threshold).float()

        weight = (
            self.weight_standing * standing_mask.unsqueeze(1)
            + self.weight_walking * walking_mask.unsqueeze(1)
            + self.weight_running * running_mask.unsqueeze(1)
        )

        current_joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
        desired_joint_pos = self.default_joint_pos[:, asset_cfg.joint_ids]
        error = torch.abs(current_joint_pos - desired_joint_pos)
        return (weight * error).sum(dim=1)


class variable_posture_l2(ManagerTermBase):
    """
    compute gaussian kernel reward to regularize robot's whole body posture for each gait.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        asset = env.scene[cfg.params["asset_cfg"].name]
        self.default_joint_pos = asset.data.default_joint_pos

        _, joint_names = asset.find_joints(cfg.params["asset_cfg"].joint_names)

        _, _, weight_standing = resolve_matching_names_values(
            data=cfg.params["weight_standing"],  # type: ignore
            list_of_strings=joint_names,
        )
        self.weight_standing = torch.tensor(weight_standing, device=env.device, dtype=torch.float32)

        _, _, weight_walking = resolve_matching_names_values(
            data=cfg.params["weight_walking"],  # type: ignore
            list_of_strings=joint_names,
        )
        self.weight_walking = torch.tensor(weight_walking, device=env.device, dtype=torch.float32)

        _, _, weight_running = resolve_matching_names_values(
            data=cfg.params["weight_running"],  # type: ignore
            list_of_strings=joint_names,
        )
        self.weight_running = torch.tensor(weight_running, device=env.device, dtype=torch.float32)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
        command_name: str,
        weight_standing: dict,
        weight_walking: dict,
        weight_running: dict,
        walking_threshold: float = 0.5,
        running_threshold: float = 1.5,
    ) -> torch.Tensor:

        asset = env.scene[asset_cfg.name]
        command = env.command_manager.get_command(command_name)

        linear_speed = torch.norm(command[:, :2], dim=-1)
        angular_speed = torch.abs(command[:, 2])
        total_speed = linear_speed + angular_speed

        standing_mask = (total_speed < walking_threshold).float()
        walking_mask = ((total_speed >= walking_threshold) & (total_speed < running_threshold)).float()
        running_mask = (total_speed >= running_threshold).float()

        weight = (
            self.weight_standing * standing_mask.unsqueeze(1)
            + self.weight_walking * walking_mask.unsqueeze(1)
            + self.weight_running * running_mask.unsqueeze(1)
        )

        current_joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
        desired_joint_pos = self.default_joint_pos[:, asset_cfg.joint_ids]
        error = torch.square(current_joint_pos - desired_joint_pos)
        return (weight * error).sum(dim=1)


def energy(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    reward = torch.norm(torch.abs(asset.data.applied_torque * asset.data.joint_vel), dim=-1)
    return reward


"""
action
"""


def action_rate_l2(env: ManagerBasedRLEnv, joint_idx: list[int]) -> torch.Tensor:
    """Penalize the rate of change of the actions using L2 squared kernel."""
    return torch.sum(
        torch.square(env.action_manager.action[:, joint_idx] - env.action_manager.prev_action[:, joint_idx]), dim=1
    )


"""
reimplementation of contact rewards to handle soft contact.
"""


def feet_air_time_positive_biped_soft(
    env,
    command_name: str,
    threshold: float,
    action_term_name: str = "physics_callback",
    command_threshold: float = 0.05,
) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_solver = env.action_manager.get_term(action_term_name).contact_solver
    # compute the reward
    air_time = contact_solver.data.current_air_time  # (num_envs, num_bodies)
    contact_time = contact_solver.data.current_contact_time  # (num_envs, num_bodies)
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    # reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    linear_norm = torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1)
    angular_norm = torch.abs(env.command_manager.get_command(command_name)[:, 2])
    total_norm = linear_norm + angular_norm
    reward *= total_norm > command_threshold
    return reward


def fly_soft(
    env: ManagerBasedRLEnv,
    threshold: float,
    action_term_name: str = "physics_callback",
    command_name: str = "base_velocity",
    velocity_threshold: float = 1.5,
) -> torch.Tensor:
    contact_solver = env.action_manager.get_term(action_term_name).contact_solver
    net_contact_forces = contact_solver.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, :, :], dim=-1), dim=1)[0] > threshold
    linear_norm = torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1)
    is_active = linear_norm < velocity_threshold
    reward = torch.sum(is_contact, dim=-1) < 0.5
    return reward * is_active


def feet_slide_soft(
    env,
    action_term_name: str = "physics_callback",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    contact_threshold: float = 5.0,
) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_solver = env.action_manager.get_term(action_term_name).contact_solver
    contacts = contact_solver.data.net_forces_w_history[:, :, :, :].norm(dim=-1).max(dim=1)[0] > contact_threshold
    asset = env.scene[asset_cfg.name]

    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward


def foot_force_soft(
    env: ManagerBasedRLEnv, action_term_name: str = "physics_callback", threshold: float = 500, max_reward: float = 400
) -> torch.Tensor:
    contact_solver = env.action_manager.get_term(action_term_name).contact_solver
    reward = contact_solver.data.net_forces_w[:, :, 2].norm(dim=-1)
    reward[reward < threshold] = 0
    reward[reward > threshold] -= threshold
    reward = reward.clamp(min=0, max=max_reward)
    return reward


def reward_soft_landing_soft(
    env: ManagerBasedRLEnv,
    action_term_name: str = "physics_callback",
    command_name: str = "base_velocity",
    command_threshold: float = 0.05,
) -> torch.Tensor:
    """Penalize high impact forces at landing to encourage soft footfalls."""
    contact_solver = env.action_manager.get_term(action_term_name).contact_solver
    assert contact_solver.data.net_forces_w is not None

    forces = contact_solver.data.net_forces_w  # [B, N, 3]
    first_contact = contact_solver.compute_first_contact(env.step_dt)  # [B, N]

    force_magnitude = torch.norm(forces, dim=-1)  # [B, N]
    landing_impact = force_magnitude * first_contact.float()  # [B, N]
    cost = torch.sum(landing_impact, dim=1)  # [B]

    num_landings = torch.sum(first_contact.float())
    mean_landing_force = torch.sum(landing_impact) / torch.clamp(num_landings, min=1)
    env.extras["log"]["Metrics/soft_landing_force_mean"] = mean_landing_force

    if command_name is not None:
        command = env.command_manager.get_command(command_name)
        linear_norm = torch.norm(command[:, :2], dim=1)
        angular_norm = torch.abs(command[:, 2])
        total_command = linear_norm + angular_norm
        active = (total_command > command_threshold).float()
        cost = cost * active
    return cost
