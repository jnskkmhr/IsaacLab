# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Object orientation and optional footprint termination helpers."""

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import matrix_from_quat

from isaaclab_tasks.contrib.push.mdp.terminations import box_tilted


def outside_bed(
    env, asset_cfg: SceneEntityCfg, bed_size: tuple[float, float], half_size: tuple[float, float, float], margin: float
) -> torch.Tensor:
    """Test the body's rotated bounding box against the tray's inner XY edges."""
    asset = env.scene[asset_cfg.name]
    position = asset.data.root_pos_w.torch - env.scene.env_origins
    rotation = matrix_from_quat(asset.data.root_quat_w.torch)
    extent = (rotation.abs() @ position.new_tensor(half_size))[:, :2]
    limit = position.new_tensor(bed_size) / 2 - margin
    return (position[:, :2].abs() + extent > limit).any(dim=-1)


def object_tilted(env, command_name: str, limit: float) -> torch.Tensor:
    """Keep boxes and cylinders upright; a sphere's orientation is unrestricted."""
    command = env.command_manager.get_term(command_name)
    return box_tilted(env, limit=limit, object_name=command.cfg.object_name) & ~command.is_sphere
