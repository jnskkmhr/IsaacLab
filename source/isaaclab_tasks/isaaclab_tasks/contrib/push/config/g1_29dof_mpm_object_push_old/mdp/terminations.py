# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Object-pushing terminations."""

import torch
import warp as wp

from isaaclab.utils.math import quat_apply


def box_outside_patch(env, margin=0.3):
    """Terminate before any box corner reaches recycled/boundary particles.

    Uses the world-aligned particle patch, not the robot's rotating local axes.
    The margin reserves room for patch quantization and motion before the next check.
    """
    box = env.scene["box"].data
    center = wp.to_torch(env.moving_patch_particle.patch_centers)
    half_size = torch.tensor(env.cfg.scene.box.spawn.size, device=env.device) / 2
    signs = torch.tensor([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], device=env.device)
    offsets = (signs * half_size).expand(env.num_envs, -1, -1)
    corners = (
        quat_apply(box.root_quat_w.torch[:, None, :].expand(-1, 8, -1), offsets) + box.root_pos_w.torch[:, None, :]
    )
    allowed = (
        torch.tensor(env.cfg.scene.terrain.moving_patch_terrain.simulated_terrain_size, device=env.device) / 2 - margin
    )
    return ((corners[:, :, :2] - center[:, None, :]).abs() > allowed).any(dim=(1, 2))
