# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sagittal reflection of paired wrist SE(3) observations."""

import torch

from isaaclab_contrib.mdp import mirror_quat, mirror_vec3


def mirror_hand_poses(data: torch.Tensor) -> torch.Tensor:
    """Swap left/right poses and reflect position and XYZW orientation."""
    if data.shape[-1] != 14:
        raise ValueError("Expected left and right poses, each [x, y, z, qx, qy, qz, qw].")
    poses = data.reshape(*data.shape[:-1], 2, 7).flip(-2)
    return torch.cat((mirror_vec3(poses[..., :3]), mirror_quat(poses[..., 3:])), dim=-1).reshape_as(data)


def mirror_direction(data):
    return data * data.new_tensor([1, -1])


def mirror_pose(data):
    return torch.cat((mirror_vec3(data[..., :3]), mirror_quat(data[..., 3:])), dim=-1)


def mirror_velocity(data):
    return torch.cat((mirror_vec3(data[..., :3]), mirror_vec3(data[..., 3:], axial=True)), dim=-1)


def mirror_teacher_command(data):
    from isaaclab_tasks.contrib.velocity.config.g1_29dof_rigid.mdp.symmetry import mirror_velocity_heading

    return torch.cat((mirror_velocity_heading(data[..., :7]), data[..., 7:]), dim=-1)
