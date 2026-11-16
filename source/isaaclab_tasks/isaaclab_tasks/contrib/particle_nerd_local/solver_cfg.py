# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for ParticleNeRDLocalSolver (translation-invariant CConv + packing rate + SDF + stress head)."""

from pathlib import Path

from isaaclab_newton.physics import NewtonSolverCfg

from isaaclab.utils import configclass

DEFAULT_PARTICLE_NERD_LOCAL_CHECKPOINT = str(Path(__file__).resolve().parent / "checkpoint/best.pt")


@configclass
class ParticleNeRDLocalSolverCfg(NewtonSolverCfg):
    """Translation-invariant ParticleNeRD-Local solver with unified SDF colliders and stress-to-contact integration."""

    class_type: str = "isaaclab_tasks.contrib.particle_nerd_local.manager:NewtonParticleNeRDLocalManager"
    checkpoint: str = DEFAULT_PARTICLE_NERD_LOCAL_CHECKPOINT
    collider_body_patterns: list[str] = [r".*/Object"]
    active_k: int | str | None = None
    selection_interval: int = 4
    settle_steps: int = 0
    settle_cache_dir: str = "/tmp/particle_nerd_local_settled_cache"
    compile_model: bool = True
    capture_solver: bool = True
    batch_inference: bool = True
    use_unified_sdf_colliders: bool = True
    use_stress_contact_head: bool = False
    effective_mass_scale: float = 10.0
    tangent_mass_scale: float = 10.0
    horizontal_mass_scale: float = 1.0
    upward_mass_scale: float = 1.0
    normal_power: float = 1.0
    contact_iterations: int = 200
    contact_relaxation: float = 0.02
    gradient_radius: float = 0.065
    gradient_contact_band: float = 0.08
    contact_stabilization: float = 0.01
    contact_offset: float = 0.0
    bed_depth: float = 0.3
    far_support_scale: float = 5.0
    far_gap_weight: float = 0.3
    far_influence_weight: float = 0.15
