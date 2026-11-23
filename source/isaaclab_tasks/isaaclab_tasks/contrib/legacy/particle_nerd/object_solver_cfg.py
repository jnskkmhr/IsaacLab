# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Object Transolver and analytical contact settings for the interactive examples."""

from pathlib import Path

from isaaclab.utils import configclass

from .solver_cfg import ParticleNeRDSolverCfg


@configclass
class ParticleNeRDObjectSolverCfg(ParticleNeRDSolverCfg):
    """Current G1 compact NeRD method with object geometry and a retrained checkpoint."""

    class_type: str = "isaaclab_tasks.contrib.legacy.particle_nerd.object_manager:NewtonParticleNeRDObjectManager"
    checkpoint: str = str(Path(__file__).resolve().parent / "checkpoint/object_topk.pt")
    collider_body_patterns: list[str] = [r".*/Object"]
    batch_inference: bool = True
    active_k: int | str | None = "auto"
    anchor_particles: int = 512
    selection_interval: int = 4
    batch_contacts: bool = False
    effective_mass_scale: float = 100.0
    tangent_mass_scale: float = 40.0
    horizontal_mass_scale: float = 8.0
    upward_mass_scale: float = 2.0
    normal_power: float = 3.0
    contact_iterations: int = 200
    contact_relaxation: float = 0.02
    gradient_radius: float = 0.065
    gradient_contact_band: float = 0.08
    contact_stabilization: float = 0.01
    contact_offset: float = 0.0
    bed_depth: float = 0.3
    """Initial bed depth above the tray floor, used to locate each environment's origin (m)."""
