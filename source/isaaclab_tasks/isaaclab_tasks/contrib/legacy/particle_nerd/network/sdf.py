# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""GPU signed-distance sampling shared by learned motion and contact estimation."""

from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import warp as wp

wp.set_module_options({"enable_backward": True})


@wp.func
def sample_distance(p: wp.vec3, values: wp.array3d(dtype=float), origin: wp.vec3, spacing: float):
    grid = (p - origin) / spacing
    bounded = wp.vec3(0.0)
    base = wp.vec3i(0)
    fraction = wp.vec3(0.0)
    for a in range(3):
        bounded[a] = wp.clamp(grid[a], 0.0, float(values.shape[a] - 1))
        base[a] = wp.min(int(wp.floor(bounded[a])), values.shape[a] - 2)
        fraction[a] = bounded[a] - float(base[a])
    result = float(0.0)
    for i in range(2):
        for j in range(2):
            for k in range(2):
                w = wp.vec3(float(i), float(j), float(k))
                weight = float(1.0)
                for a in range(3):
                    weight *= (1.0 - w[a]) * (1.0 - fraction[a]) + w[a] * fraction[a]
                result += weight * values[base[0] + i, base[1] + j, base[2] + k]
    # Extend outside the padded volume conservatively; nearby contacts lie inside it.
    return result + spacing * wp.length(grid - bounded)


@wp.func
def surface(p: wp.vec3, values: wp.array3d(dtype=float), origin: wp.vec3, spacing: float):
    d = sample_distance(p, values, origin, spacing)
    gradient = wp.vec3(0.0)
    for a in range(3):
        offset = wp.vec3(0.0)
        offset[a] = spacing * 0.5
        gradient[a] = (
            sample_distance(p + offset, values, origin, spacing) - sample_distance(p - offset, values, origin, spacing)
        ) / spacing
    normal = wp.vec3(0.0, 0.0, 1.0)
    if wp.length(gradient) > 1.0e-8:
        normal = wp.normalize(gradient)
    return d, normal


@lru_cache(maxsize=16)
def tensors(device, geometry="shovel"):
    if geometry == "shovel":
        with np.load(Path(__file__).parent / "assets/shovel_sdf.npz") as data:
            return (
                torch.as_tensor(data["values"], device=device),
                tuple(map(float, data["origin"])),
                float(data["spacing"]),
            )
    return (torch.zeros((2, 2, 2), device=device), (0.0, 0.0, 0.0), 1.0)
