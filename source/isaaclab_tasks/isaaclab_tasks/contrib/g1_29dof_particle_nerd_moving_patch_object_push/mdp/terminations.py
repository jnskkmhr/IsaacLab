# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Keep the entire object footprint inside the simulated (non-boundary) patch."""

import torch
import warp as wp

from isaaclab.utils.math import matrix_from_quat


def object_outside_patch(env, object_name, half_extents, margin):
    obj = env.scene[object_name]
    # Exact world-axis projected half extents of the rotated cube.
    radius = (matrix_from_quat(obj.data.root_quat_w.torch).abs() @ torch.tensor(half_extents, device=env.device))[:, :2]
    half = torch.tensor(env.cfg.scene.terrain.moving_patch_terrain.simulated_terrain_size, device=env.device) / 2
    relative = (obj.data.root_pos_w.torch[:, :2] - wp.to_torch(env.moving_patch_particle.patch_centers)).abs()
    return (relative + radius + margin > half).any(-1)


def invalid_state(env, asset_name):
    data = env.scene[asset_name].data
    return ~(torch.isfinite(data.root_link_pose_w.torch).all(-1) & torch.isfinite(data.root_com_vel_w.torch).all(-1))
