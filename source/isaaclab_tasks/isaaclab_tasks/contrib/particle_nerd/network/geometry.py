# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch


def boundary_boxes(lower, upper, device):
    """Build 5 boundary box descriptors (1 floor + 4 walls) for intrinsic bed bounds [lower, upper]."""
    x_lo, y_lo, z_lo = float(lower[0]), float(lower[1]), float(lower[2])
    x_hi, y_hi, _z_hi = float(upper[0]), float(upper[1]), float(upper[2])
    lx, ly = x_hi - x_lo, y_hi - y_lo
    cx, cy = 0.5 * (x_lo + x_hi), 0.5 * (y_lo + y_hi)
    wall = 0.1
    wall_h = 0.4
    floor_h = 0.1
    poses = torch.tensor(
        [
            [cx, cy, z_lo - 0.5 * floor_h, 0.0, 0.0, 0.0, 1.0],
            [x_hi + 0.5 * wall, cy, z_lo + 0.5 * wall_h, 0.0, 0.0, 0.0, 1.0],
            [x_lo - 0.5 * wall, cy, z_lo + 0.5 * wall_h, 0.0, 0.0, 0.0, 1.0],
            [cx, y_hi + 0.5 * wall, z_lo + 0.5 * wall_h, 0.0, 0.0, 0.0, 1.0],
            [cx, y_lo - 0.5 * wall, z_lo + 0.5 * wall_h, 0.0, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
        device=device,
    )
    sizes = torch.tensor(
        [
            [0.5 * lx + wall, 0.5 * ly + wall, 0.5 * floor_h],
            [0.5 * wall, 0.5 * ly + wall, 0.5 * wall_h],
            [0.5 * wall, 0.5 * ly + wall, 0.5 * wall_h],
            [0.5 * lx + wall, 0.5 * wall, 0.5 * wall_h],
            [0.5 * lx + wall, 0.5 * wall, 0.5 * wall_h],
        ],
        dtype=torch.float32,
        device=device,
    )
    friction = torch.full((5,), 0.85, dtype=torch.float32, device=device)
    return poses, sizes, friction
