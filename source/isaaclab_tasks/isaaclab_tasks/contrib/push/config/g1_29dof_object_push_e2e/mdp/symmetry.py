# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sagittal reflection rules for the box and contact observations."""

import torch

from isaaclab_contrib.mdp import mirror_quat, mirror_vec3


def mirror_pose(data: torch.Tensor) -> torch.Tensor:
    """Reflect a base-frame pose ``[x, y, z, qx, qy, qz, qw]``."""
    if data.shape[-1] != 7:
        raise ValueError("Expected a position followed by an XYZW quaternion.")
    return torch.cat((mirror_vec3(data[..., :3]), mirror_quat(data[..., 3:])), dim=-1)


def mirror_twist(data: torch.Tensor) -> torch.Tensor:
    """Reflect a base-frame linear velocity followed by an angular velocity."""
    if data.shape[-1] != 6:
        raise ValueError("Expected a linear velocity followed by an angular velocity.")
    return torch.cat((mirror_vec3(data[..., :3]), mirror_vec3(data[..., 3:], axial=True)), dim=-1)


def mirror_velocity_command(data: torch.Tensor) -> torch.Tensor:
    """Reflect ``[vx, vy, wz]``: lateral and yaw velocity change sign."""
    if data.shape[-1] != 3:
        raise ValueError("Expected forward, lateral, and yaw velocity commands.")
    return data * data.new_tensor((1, -1, -1))


def mirror_box_contact_state(data: torch.Tensor) -> torch.Tensor:
    """Swap the left- and right-wrist force channels of ``[left, right, other]``."""
    if data.shape[-1] != 3:
        raise ValueError("Expected left-wrist, right-wrist, and other-body contact forces.")
    return data[..., [1, 0, 2]]
