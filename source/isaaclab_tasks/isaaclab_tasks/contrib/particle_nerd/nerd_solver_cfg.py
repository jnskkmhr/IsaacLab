# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration of :class:`~isaaclab_tasks.contrib.particle_nerd.nerd_solver.ParticleNeRDSolver`."""

from pathlib import Path

from isaaclab_newton.physics import NewtonSolverCfg

from isaaclab.utils import configclass

DEFAULT_PARTICLE_NERD_CHECKPOINT = str(Path(__file__).resolve().parent / "checkpoint/best.pt")
"""Trained checkpoint shipped with the package."""


@configclass
class ParticleNeRDSolverCfg(NewtonSolverCfg):
    """Neural particle solver: a trained network steps the particles, a wrench solver returns the contact wrenches."""

    class_type: str = "isaaclab_tasks.contrib.particle_nerd.nerd_manager:NewtonParticleNeRDManager"

    checkpoint: str = DEFAULT_PARTICLE_NERD_CHECKPOINT
    """Path of the trained checkpoint."""

    collider_body_patterns: list[str] = [r".*/Object"]
    """Regular expressions of the body labels that interact with the particles. Every pattern must match at
    least one body in every environment."""

    bed_depth: float = 0.3
    """Depth of the particle bed [m]. Used to place the tray walls and floor when the scene has no static boxes
    that collide with particles."""

    active_k: int | str | None = None
    """Number of particles per environment that the network steps. The other particles stay where they are.

    ``None`` steps every particle. ``"auto"`` chooses the number from the particle count and the material. The
    particles are the ones nearest to the colliders and the fastest ones; up to 256 evenly spaced other particles
    are added as input only, weighted so that they stand for all particles that are not stepped."""

    selection_interval: int = 4
    """Number of steps between two selections of the stepped particles (only with :attr:`active_k`)."""

    sleep_speed: float = 0.02
    """Speed below which a particle sleeps [m/s]. A particle whose predicted move in a step is shorter than
    ``sleep_speed * dt`` keeps its position and gets zero velocity. 0 disables sleeping. Sleeping stops the slow
    sinking of a resting bed."""

    compile_model: bool = True
    """Compile the network with :func:`torch.compile`."""

    capture_solver: bool = True
    """Record the coupled step as a CUDA graph and replay it. Used when the physics configuration does not set
    ``capture_solver`` itself."""

    effective_mass_scale: float = 10.0
    """Effective mass of a contact particle, as a multiple of its own mass, for a contact normal that points
    down (the particle is pushed into the bed)."""

    horizontal_mass_scale: float = 1.0
    """Effective mass multiple for a horizontal contact normal."""

    upward_mass_scale: float = 1.0
    """Effective mass multiple for a contact normal that points up (the particle is lifted)."""

    tangent_mass_scale: float = 10.0
    """Effective mass multiple for friction."""

    normal_power: float = 1.0
    """Exponent of the blend between the three normal-direction multiples."""

    contact_iterations: int = 200
    """Maximum number of iterations of the wrench solver."""

    contact_relaxation: float = 0.02
    """Relaxation factor of the wrench solver."""

    gradient_contact_band: float = 0.08
    """Distance from a collider surface within which a particle is a contact candidate [m]."""

    contact_stabilization: float = 0.01
    """Fraction of the penetration that the wrench solver removes per step."""

    contact_offset: float = 0.0
    """Extra margin added to the collider surface in the wrench solver [m]."""
