# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reflection of one relative box pose."""

import torch

from isaaclab_contrib.mdp import mirror_quat, mirror_vec3


def mirror_pose(data: torch.Tensor) -> torch.Tensor:
    """Reflect position and XYZW orientation across the robot's sagittal plane."""
    return torch.cat((mirror_vec3(data[..., :3]), mirror_quat(data[..., 3:])), dim=-1)


def mirror_velocity(data: torch.Tensor) -> torch.Tensor:
    """Reflect local linear and axial angular velocities."""
    return torch.cat((mirror_vec3(data[..., :3]), mirror_vec3(data[..., 3:], axial=True)), dim=-1)
