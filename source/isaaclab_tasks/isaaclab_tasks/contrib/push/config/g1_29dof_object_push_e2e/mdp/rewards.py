# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Contact-gated box progress, contact penalties, and posture bands for whole-body pushing."""

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply, quat_apply_inverse, yaw_quat


def box_contact_forces(env, entity) -> torch.Tensor:
    """Return the box contact force magnitude on every body of a contact source [N].

    Args:
        env: The environment.
        entity: Either a contact sensor filtered against the box, so that ground and self
            contacts are excluded, or the robot articulation of an environment that
            provides ``object_contact_force`` (the coupled-sand environments, where the
            contact sensor is not available).

    Returns:
        Force magnitudes, shape (num_envs, num_bodies), in the body order of ``entity``.
    """
    if not hasattr(entity.data, "force_matrix_w"):
        return torch.linalg.vector_norm(env.object_contact_force, dim=-1)
    matrix = entity.data.force_matrix_w
    if matrix is None:
        raise ValueError("Box-contact terms require a contact sensor filtered against the box.")
    return torch.linalg.vector_norm(matrix.torch, dim=-1).flatten(2).sum(dim=-1)


def box_contact_force(env, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Sum box contact force magnitudes over the selected robot bodies [N]; see :func:`box_contact_forces`."""
    return box_contact_forces(env, env.scene[sensor_cfg.name])[:, sensor_cfg.body_ids].flatten(1).sum(dim=-1)


def hand_contact_fraction(
    env, left_sensor_cfg: SceneEntityCfg, right_sensor_cfg: SceneEntityCfg, force_threshold: float
) -> torch.Tensor:
    """Fraction of the two wrists touching the box: 0, 0.5, or 1."""
    left = box_contact_force(env, left_sensor_cfg) > force_threshold
    right = box_contact_force(env, right_sensor_cfg) > force_threshold
    return 0.5 * (left.float() + right.float())


def progress_ratio(
    command_xy: torch.Tensor, robot_quat_w: torch.Tensor, box_lin_vel_w: torch.Tensor, command_threshold: float
) -> torch.Tensor:
    """Box speed along the commanded direction divided by the commanded speed, clamped to [-1, 1].

    The command is in the robot yaw frame, into which the world-frame box velocity is rotated.
    Standing commands return zero.
    """
    speed = torch.linalg.vector_norm(command_xy, dim=-1)
    velocity = quat_apply_inverse(yaw_quat(robot_quat_w), box_lin_vel_w)[:, :2]
    ratio = (velocity * command_xy).sum(dim=-1) / speed.clamp(min=command_threshold).square()
    return ratio.clamp(-1.0, 1.0) * (speed > command_threshold)


def box_progress_ratio(env, command_name: str, object_name: str, command_threshold: float) -> torch.Tensor:
    """Progress ratio of the named object against the named velocity command; see :func:`progress_ratio`."""
    return progress_ratio(
        env.command_manager.get_command(command_name)[:, :2],
        env.scene["robot"].data.root_quat_w.torch,
        env.scene[object_name].data.root_lin_vel_w.torch,
        command_threshold,
    )


def body_clear_gate(env, body_sensor_cfg: SceneEntityCfg, force_scale: float) -> torch.Tensor:
    """Soft gate in (0, 1] that closes as non-wrist bodies press on the box: exp(-force / force_scale)."""
    return torch.exp(-box_contact_force(env, body_sensor_cfg) / force_scale)


def object_progress(
    env,
    command_name: str,
    object_name: str,
    left_sensor_cfg: SceneEntityCfg,
    right_sensor_cfg: SceneEntityCfg,
    body_sensor_cfg: SceneEntityCfg,
    force_threshold: float,
    body_force_scale: float,
    command_threshold: float,
) -> torch.Tensor:
    """Reward box progress toward the commanded speed, only for pushing with the wrists alone.

    Linear in the box speed up to the command, so a stationary box earns nothing. The credit
    scales with the fraction of wrists touching the box and decays with the force that any
    other body applies to it, so leaning on the box or driving it with the legs earns nothing.
    """
    contact = hand_contact_fraction(env, left_sensor_cfg, right_sensor_cfg, force_threshold)
    gate = body_clear_gate(env, body_sensor_cfg, body_force_scale)
    return box_progress_ratio(env, command_name, object_name, command_threshold) * contact * gate


def hand_contact(
    env,
    command_name: str,
    left_sensor_cfg: SceneEntityCfg,
    right_sensor_cfg: SceneEntityCfg,
    body_sensor_cfg: SceneEntityCfg,
    force_threshold: float,
    body_force_scale: float,
    command_threshold: float,
) -> torch.Tensor:
    """Reward wrist-box contact during moving commands, decaying with any other body's box contact.

    Wrist force above the threshold earns no extra credit.
    """
    command = env.command_manager.get_command(command_name)[:, :2]
    moving = torch.linalg.vector_norm(command, dim=-1) > command_threshold
    contact = hand_contact_fraction(env, left_sensor_cfg, right_sensor_cfg, force_threshold)
    return contact * body_clear_gate(env, body_sensor_cfg, body_force_scale) * moving


def non_hand_contact(env, sensor_cfg: SceneEntityCfg, force_scale: float) -> torch.Tensor:
    """Penalize box contact on legs, pelvis, torso, and upper arms, saturating at one above ``force_scale`` [N]."""
    return (box_contact_force(env, sensor_cfg) / force_scale).clamp(max=1.0)


def object_tilt_l2(env, object_name: str) -> torch.Tensor:
    """Squared sine of the box tilt from upright."""
    box = env.scene[object_name]
    up = torch.zeros_like(box.data.root_pos_w.torch)
    up[:, 2] = 1.0
    return 1.0 - quat_apply(box.data.root_quat_w.torch, up)[:, 2].square()


def base_height_band_l2(env, min_height: float, max_height: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Squared root-height excursion outside [min_height, max_height] above flat ground [m^2]."""
    height = env.scene[asset_cfg.name].data.root_pos_w.torch[:, 2] - env.scene.env_origins[:, 2]
    return ((min_height - height).clamp(min=0.0) + (height - max_height).clamp(min=0.0)).square()


def forward_lean_band_l2(env, asset_cfg: SceneEntityCfg, max_forward_lean: float) -> torch.Tensor:
    """Penalize body roll, backward lean, and forward lean beyond ``max_forward_lean`` [rad].

    Uses gravity projected into the selected body frame: x is sin(forward pitch), y is roll.
    Forward lean inside the band is free, which pushing against a load requires.
    """
    asset = env.scene[asset_cfg.name]
    quaternion = asset.data.body_link_quat_w.torch[:, asset_cfg.body_ids[0]]
    gravity = torch.nn.functional.normalize(asset.data.GRAVITY_VEC_W.torch, dim=-1)
    projected = quat_apply_inverse(quaternion, gravity)
    forward = projected[:, 0]
    limit = torch.sin(torch.tensor(max_forward_lean, device=forward.device))
    pitch_excess = (forward - limit).clamp(min=0.0) + (-forward).clamp(min=0.0)
    return projected[:, 1].square() + pitch_excess.square()


def feet_trailing_distance_l2(env, max_trailing_distance: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize feet farther behind the pelvis than the allowed distance, along the robot heading [m^2].

    Bounds how far the robot can lean its weight onto the box: a self-balanced push keeps the
    feet close under the pelvis, while a plank against the box trails them far behind.
    """
    asset = env.scene[asset_cfg.name]
    offset_w = asset.data.body_pos_w.torch[:, asset_cfg.body_ids] - asset.data.root_pos_w.torch[:, None, :]
    yaw = yaw_quat(asset.data.root_quat_w.torch)[:, None, :].expand(-1, offset_w.shape[1], -1)
    behind = -quat_apply_inverse(yaw, offset_w)[..., 0]
    return (behind - max_trailing_distance).clamp(min=0.0).square().sum(dim=-1)
