# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for WorldParticle neural particle dynamics and proxy contact reconstruction."""

from pathlib import Path

from isaaclab_newton.physics import NewtonSolverCfg

from isaaclab.utils import configclass

DEFAULT_WORLD_PARTICLE_CHECKPOINT = str(Path(__file__).resolve().parent / "checkpoint/best.pt")


@configclass
class ParticleWorldSolverCfg(NewtonSolverCfg):
    """WorldParticle transformer with Warp continuous convolution and proxy contact reconstruction."""

    class_type: str = "isaaclab_tasks.contrib.legacy.particle_world.manager:NewtonParticleWorldManager"
    checkpoint: str = DEFAULT_WORLD_PARTICLE_CHECKPOINT
    """Path to a trained WorldParticle checkpoint (`world_particle_v1` schema)."""
    collider_body_patterns: list[str] = [r".*/Object"]
    active_k: int | str | None = None
    selection_interval: int = 1
    anchor_particles: int = 512
    compile_model: bool = True
    batch_inference: bool = True
    batch_contacts: bool = True
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
    far_support_scale: float | None = None
    far_gap_weight: float | None = None
    far_influence_weight: float | None = None
    use_actual_velocity: bool | None = None
