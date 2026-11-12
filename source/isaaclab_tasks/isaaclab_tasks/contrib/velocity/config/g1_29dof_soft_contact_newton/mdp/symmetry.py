# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Mirror rules for the reference task's scalar-first quaternion observations."""

import torch


def mirror_quat_wxyz(data: torch.Tensor) -> torch.Tensor:
    """Reflect WXYZ orientations across the XZ plane, preserving quaternion sign."""
    if data.shape[-1] != 4:
        raise ValueError("Expected a WXYZ quaternion.")
    return data * data.new_tensor((1, -1, 1, -1))


def mirror_velocity_heading_wxyz(data: torch.Tensor) -> torch.Tensor:
    """Reflect [vx, vy, wz, qw, qx, qy, qz] commands across the XZ plane."""
    if data.shape[-1] != 7:
        raise ValueError("Expected three velocity commands followed by a WXYZ heading.")
    return torch.cat((data[..., :3] * data.new_tensor((1, -1, -1)), mirror_quat_wxyz(data[..., 3:])), dim=-1)
