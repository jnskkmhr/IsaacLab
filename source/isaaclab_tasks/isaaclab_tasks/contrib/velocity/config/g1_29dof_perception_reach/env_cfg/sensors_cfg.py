# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch

from isaaclab.sensors import patterns


def elevation_grid_pattern(cfg: patterns.GridPatternCfg, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Cast from 20 m above the torso along world-down, independent of torso tilt."""
    starts, directions = patterns.grid_pattern(cfg, device)
    starts[:, 2] = 20.0
    return starts, directions
