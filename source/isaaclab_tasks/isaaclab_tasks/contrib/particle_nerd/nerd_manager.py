# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Isaac Lab Newton managers for :class:`~.nerd_solver.ParticleNeRDSolver`.

- :class:`NewtonParticleNeRDManager` builds the solver for one coupler entry.
- :class:`NewtonParticleNeRDCouplerManager` steps the coupled rigid and neural solvers, optionally as a recorded
  CUDA graph.
"""

from __future__ import annotations

import gc
from typing import ClassVar

import newton
import torch
import warp as wp
from isaaclab_newton.physics import NewtonManager

from isaaclab.physics import PhysicsManager

from isaaclab_contrib.coupling.coupler import NewtonCouplerManager

from .nerd_solver_cfg import ParticleNeRDSolverCfg

_MATERIAL_ATTRIBUTES = (
    ("young_modulus", 1.0e15),
    ("poisson_ratio", 0.3),
    ("damping", 0.0),
    ("hardening", 0.0),
    ("friction", 0.5),
    ("yield_pressure", 1.0e15),
    ("tensile_yield_ratio", 0.0),
    ("yield_stress", 0.0),
    ("hardening_rate", 1.0),
    ("softening_rate", 1.0),
    ("dilatancy", 0.0),
    ("viscosity", 0.0),
)
"""Per-particle material parameters of the particle asset (``mpm`` namespace) and their defaults."""

_MPM_STATE_ATTRIBUTES = (
    "mpm:particle_qd_grad",
    "mpm:particle_elastic_strain",
    "mpm:particle_Jp",
    "mpm:particle_stress",
    "mpm:particle_transform",
)
"""Per-particle state of the MPM solver, which the neural solver does not use."""

_BUILDER_PARTICLE_LISTS = (
    "particle_q",
    "particle_qd",
    "particle_mass",
    "particle_radius",
    "particle_flags",
    "particle_world",
    "particle_mu",
    "particle_cohesion",
    "particle_adhesion",
)


class NewtonParticleNeRDManager(NewtonManager):
    """Build a :class:`~.nerd_solver.ParticleNeRDSolver` for a coupler entry."""

    @classmethod
    def _register_builder_attributes(cls, builder):
        # Register only the material parameters that the particle asset writes. The MPM solver would also
        # register its per-particle state (148 bytes per particle in every model and state copy).
        if builder.has_custom_attribute("mpm:young_modulus"):
            return
        for name, default in _MATERIAL_ATTRIBUTES:
            builder.add_custom_attribute(
                newton.ModelBuilder.CustomAttribute(
                    name=name,
                    frequency=newton.Model.AttributeFrequency.PARTICLE,
                    assignment=newton.Model.AttributeAssignment.MODEL,
                    dtype=wp.float32,
                    default=default,
                    namespace="mpm",
                )
            )

    @classmethod
    def _prepare_builder_for_finalize(cls, builder):
        for name in _MPM_STATE_ATTRIBUTES:
            builder.custom_attributes.pop(name, None)
        # Kinematic bodies get zero mass, so the wrench solver treats them as immovable
        kinematic = int(newton.BodyFlags.KINEMATIC)
        for body, flags in enumerate(builder.body_flags):
            if int(flags) & kinematic:
                builder.body_mass[body] = 0.0
                builder.body_inv_mass[body] = 0.0
                builder.body_inertia[body] = wp.mat33()
                builder.body_inv_inertia[body] = wp.mat33()

    @classmethod
    def _create_solver(cls, model, solver_cfg):
        from .nerd_solver import ParticleNeRDSolver

        return ParticleNeRDSolver(model, solver_cfg)


class NewtonParticleNeRDCouplerManager(NewtonCouplerManager):
    """Step the coupled rigid and neural solvers, replaying a recorded CUDA graph when capture is enabled."""

    _solver_graphs: ClassVar[dict[tuple, torch.cuda.CUDAGraph]] = {}
    _solver_graph_pool: ClassVar[tuple | None] = None
    _solver_graph_signature: ClassVar[tuple | None] = None

    @classmethod
    def _solver_specific_clear(cls):
        cls._solver_graphs = {}
        cls._solver_graph_pool = None
        cls._solver_graph_signature = None
        super()._solver_specific_clear()

    @classmethod
    def _build_solver(cls, model, solver_cfg):
        # Free the host-side particle lists of the finalized builders before the coupled solver creates its
        # per-entry views; with many environments they otherwise exhaust the host memory.
        from isaaclab.sim import SimulationContext

        sim = SimulationContext.instance()
        if sim is not None:
            for _, resource in getattr(sim, "_backend_registry", ()):
                if not isinstance(getattr(resource, "particle_q", None), list):
                    continue
                for name in _BUILDER_PARTICLE_LISTS:
                    values = getattr(resource, name, None)
                    if isinstance(values, list):
                        values.clear()
                for attribute in getattr(resource, "custom_attributes", {}).values():
                    if getattr(getattr(attribute, "frequency", None), "name", None) == "PARTICLE":
                        if isinstance(getattr(attribute, "values", None), list):
                            attribute.values.clear()
            gc.collect()
        super()._build_solver(model, solver_cfg)

    @classmethod
    def _initialize_contacts(cls) -> None:
        # Without soft contacts the collision pipeline needs no particle buffers
        model = cls.backend.model
        particle_count = int(model.particle_count)
        skip_particles = cls._collision_cfg is not None and getattr(cls._collision_cfg, "soft_contact_max", None) == 0
        try:
            if skip_particles:
                model.particle_count = 0
            super()._initialize_contacts()
        finally:
            if skip_particles:
                model.particle_count = particle_count

    @staticmethod
    def _requires_external_contacts(solver_cfg):
        if isinstance(solver_cfg, ParticleNeRDSolverCfg):
            return False
        return NewtonCouplerManager._requires_external_contacts(solver_cfg)

    @classmethod
    def _get_cuda_graph_model_signature(cls):
        return (
            tuple((name, getattr(entry.solver, "capture_revision", 0)) for name, entry in cls._solver._entries.items())
            if cls._solver is not None
            else ()
        )

    @classmethod
    def _step_solver(cls, state_0, state_1, control, contacts, substep_dt):
        from .nerd_solver import ParticleNeRDSolver

        neural_solvers = []
        for entry in cls._solver._entries.values():
            if isinstance(entry.solver, ParticleNeRDSolver):
                entry.solver.capture_collider_state(state_0, entry.body_local_to_global)
                neural_solvers.append(entry.solver)
        capture = getattr(PhysicsManager._cfg, "capture_solver", None)
        if capture is None:
            capture = bool(neural_solvers) and all(solver.cfg.capture_solver for solver in neural_solvers)
        if not capture:
            cls._solver.step(state_0, state_1, control, contacts, substep_dt)
            if cls._scene_data_backend is not None:
                cls._scene_data_backend.geometry_timestamp += 1
            return

        signature = (id(cls._solver), cls._get_cuda_graph_model_signature())
        if signature != cls._solver_graph_signature:
            cls._solver_graphs = {}
            cls._solver_graph_pool = None
            cls._solver_graph_signature = signature
        # A step that selects the active particles runs different work, so it gets its own graph
        selects = tuple(solver.selection_phase == 0 for solver in neural_solvers)
        key = (id(state_0), id(state_1), id(control), id(contacts), substep_dt, selects)
        graph = cls._solver_graphs.get(key)
        stream = wp.get_stream(cls.backend.model.device)
        if graph is None:
            # Run the step once outside a graph (it compiles and allocates), then record the same step
            phases = [solver.selection_phase for solver in neural_solvers]
            cls._solver.step(state_0, state_1, control, contacts, substep_dt)
            for solver, phase in zip(neural_solvers, phases, strict=True):
                solver.selection_phase = phase
                solver._has_neural_state = True
            if cls._solver_graph_pool is None:
                cls._solver_graph_pool = torch.cuda.graph_pool_handle()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=cls._solver_graph_pool, stream=wp.stream_to_torch(stream)):
                with wp.ScopedCapture(device=cls.backend.model.device, stream=stream, external=True):
                    cls._solver.step(state_0, state_1, control, contacts, substep_dt)
            cls._solver_graphs[key] = graph
        else:
            with torch.cuda.stream(wp.stream_to_torch(stream)):
                graph.replay()
            for solver in neural_solvers:
                solver.selection_phase = (solver.selection_phase + 1) % solver.cfg.selection_interval
        if cls._scene_data_backend is not None:
            cls._scene_data_backend.geometry_timestamp += 1
