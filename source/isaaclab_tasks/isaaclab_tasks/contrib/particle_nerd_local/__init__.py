# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ParticleNeRD-Local: Translation-invariant local-to-global Transolver with unified SDF colliders & stress head."""

from . import config
from .features import LocalNeighborhoodAggregator, build_collider_sdf_grid, features_particle_nerd_local
from .manager import NewtonParticleNeRDLocalCouplerManager, NewtonParticleNeRDLocalManager
from .model import ConsistentStressContactHead, ParticleNeRDLocalNetwork
from .solver import ParticleNeRDLocalSolver
from .solver_cfg import DEFAULT_PARTICLE_NERD_LOCAL_CHECKPOINT, ParticleNeRDLocalSolverCfg

__all__ = [
    "ConsistentStressContactHead",
    "DEFAULT_PARTICLE_NERD_LOCAL_CHECKPOINT",
    "LocalNeighborhoodAggregator",
    "NewtonParticleNeRDLocalCouplerManager",
    "NewtonParticleNeRDLocalManager",
    "ParticleNeRDLocalNetwork",
    "ParticleNeRDLocalSolver",
    "ParticleNeRDLocalSolverCfg",
    "build_collider_sdf_grid",
    "config",
    "features_particle_nerd_local",
]
