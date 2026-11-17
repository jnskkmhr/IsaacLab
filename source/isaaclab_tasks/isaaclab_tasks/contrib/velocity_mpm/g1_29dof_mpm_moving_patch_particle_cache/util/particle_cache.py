# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Bounded GPU particle history with independent world-XY tile maps per environment.

Particle fields stay on the GPU. Python receives patch centers, outgoing integer
indices and tile coordinates, and counts. Chunk changes are not CUDA-graph captured.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import warp as wp

from .kernel import (
    ParticleState,
)

if TYPE_CHECKING:
    pass


PARTICLE_FIELDS = {
    "q": wp.vec3,
    "qd": wp.vec3,
    "grad": wp.mat33,
    "elastic": wp.mat33,
    "plastic": float,
    "stress": wp.mat33,
    "transform": wp.mat33,
    "mass": float,
    "radius": float,
    "flags": int,
}


def allocate_particle_state(count: int, device: wp.Device) -> ParticleState:
    """Allocate the particle fields needed to resume the material solve."""
    data = ParticleState()
    for name, dtype in PARTICLE_FIELDS.items():
        setattr(data, name, wp.empty(count, dtype=dtype, device=device))
    return data
