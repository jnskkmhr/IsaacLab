# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Wrist tracking errors against the same frames used by pose commands."""

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse


def elbow_inward_l2(
    env,
    wrist_cfg: SceneEntityCfg,
    elbow_cfg: SceneEntityCfg,
    lateral_signs: tuple[float, ...],
    reference_frame_link: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize elbows lying inward of their wrists, measured in a configured link frame.

    Wrist and elbow body selections must have matching order. Use lateral signs
    +1 for the left arm and -1 for the right arm in a reference frame with +y
    pointing left. ``reference_frame_link`` must select exactly one body. The
    result sums squared inward offsets [m²]; outward offsets cost zero.
    """
    robot = env.scene[wrist_cfg.name]
    wrist_pos = robot.data.body_link_pos_w.torch[:, wrist_cfg.body_ids]
    elbow_pos = env.scene[elbow_cfg.name].data.body_link_pos_w.torch[:, elbow_cfg.body_ids]
    reference = env.scene[reference_frame_link.name]
    reference_quat = reference.data.body_link_quat_w.torch[:, reference_frame_link.body_ids]
    if reference_quat.shape[1] != 1:
        raise ValueError("reference_frame_link must select exactly one body.")
    reference_quat = reference_quat.expand(-1, wrist_pos.shape[1], -1)
    relative_y = quat_apply_inverse(reference_quat, wrist_pos - elbow_pos)[..., 1]
    inward_offset = relative_y * relative_y.new_tensor(lateral_signs)
    return inward_offset.clamp_min(0.0).square().sum(dim=-1)


def wrist_tracking(env, command_name: str, std: float, orientation: bool = False) -> torch.Tensor:
    """Exponentially reward position distance [m] or rotation angle [rad]."""
    command = env.command_manager.get_term(command_name)
    position_error, orientation_error = command.tracking_error()
    error = orientation_error if orientation else position_error
    return torch.exp(-error.square() / std**2)


def end_effector_velocity_l2(
    env, asset_cfg: SceneEntityCfg, reference_frame_link: SceneEntityCfg
) -> torch.Tensor:
    """Sum squared link-origin speeds [m²/s²] relative to one moving reference link.

    Remove the reference link's translation and rotation-induced velocity at each
    end effector. A hand fixed relative to the torso therefore incurs zero cost
    even while the waist rotates. Squared speed is invariant to frame rotation.
    """
    asset = env.scene[asset_cfg.name]
    reference = env.scene[reference_frame_link.name]
    position = asset.data.body_link_pos_w.torch[:, asset_cfg.body_ids]
    velocity = asset.data.body_link_lin_vel_w.torch[:, asset_cfg.body_ids]
    reference_position = reference.data.body_link_pos_w.torch[:, reference_frame_link.body_ids]
    if reference_position.shape[1] != 1:
        raise ValueError("reference_frame_link must select exactly one body.")
    reference_velocity = reference.data.body_link_lin_vel_w.torch[:, reference_frame_link.body_ids]
    reference_angular_velocity = reference.data.body_link_ang_vel_w.torch[:, reference_frame_link.body_ids]
    offset = position - reference_position
    relative_velocity = velocity - reference_velocity - torch.linalg.cross(
        reference_angular_velocity.expand_as(offset), offset, dim=-1
    )
    return relative_velocity.square().sum(dim=(1, 2))
