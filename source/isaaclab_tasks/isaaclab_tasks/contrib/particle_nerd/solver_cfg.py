# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Configuration for neural particle dynamics and reconstructed collider reactions."""

from isaaclab_newton.physics import NewtonSolverCfg

from isaaclab.utils import configclass


@configclass
class FixedBedCfg:
    """Bed-frame inputs used for fixed-terrain diagnostics against the G1 example.

    Generated terrain uses a vertical height chart. A powered tray uses its body
    frame; this does not add gravity conditioning to the trained network.
    """

    kind: str = "flat"
    lower: tuple[float, float, float] = (-1.5, -1.5, -0.3)
    upper: tuple[float, float, float] = (1.5, 1.5, 0.0)
    slope: float = 0.0
    wave_height: float = 0.0
    wave_count: int = 1
    sliding_window: bool | None = None
    """None follows the example: canonicalize beds wider or longer than 3.1 m."""


@configclass
class ParticleNeRDSolverCfg(NewtonSolverCfg):
    """Particle Transolver with independent rigid proxy contact reconstruction.

    The current checkpoint schema supports two box colliders per world. There is
    no native particle solve, settling phase, or MPM fallback.
    """

    class_type: str = "isaaclab_tasks.contrib.particle_nerd.manager:NewtonParticleNeRDManager"
    checkpoint: str = ""
    """Path to a particle_transolver_v1 checkpoint with g1_particle_nerd_v1 schema."""
    collider_body_patterns: list[str] = [r".*/left_ankle_roll_link", r".*/right_ankle_roll_link"]
    active_k: int | str | None = "auto"
    selection_interval: int = 1
    """Physics steps between Top-K refreshes; intermediate steps reuse the selected IDs and blend weights."""
    anchor_particles: int = 512
    compile_model: bool = True
    batch_inference: bool = True
    """Batch compatible sand-world inputs in one network call; False retains sequential inference."""
    batch_contacts: bool = True
    """Solve independent collider contacts in parallel; False retains per-collider launches."""
    effective_mass_scale: float = 50.0
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
    """Depth below the undeformed terrain used by the canonical boundary features [m]."""
