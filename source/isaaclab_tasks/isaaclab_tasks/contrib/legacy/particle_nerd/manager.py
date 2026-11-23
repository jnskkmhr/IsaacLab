# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Newton construction and graph lifecycle for configurable Particle NeRD entries."""

from typing import ClassVar

import torch
import warp as wp
from isaaclab_newton.physics import NewtonManager

from isaaclab.physics import PhysicsManager

from isaaclab_contrib.coupling.coupler import NewtonCouplerManager

from .solver_cfg import ParticleNeRDSolverCfg


class NewtonParticleNeRDManager(NewtonManager):
    """Construct a neural solver through the standard CouplerEntryCfg factory."""

    @classmethod
    def _register_builder_attributes(cls, builder):
        # Register only the particle asset's scalar material parameter schema.
        # Omit SolverImplicitMPM's 5 per-particle STATE tensor attributes
        # (particle_qd_grad, particle_elastic_strain, particle_Jp, particle_stress,
        # particle_transform), which are unused by Particle NeRD and would otherwise
        # allocate 148 B/particle across every Model/State/ModelView copy.
        if builder.has_custom_attribute("mpm:young_modulus"):
            return
        import newton

        defaults = (
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
        for name, default in defaults:
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
        for state_attr in (
            "mpm:particle_qd_grad",
            "mpm:particle_elastic_strain",
            "mpm:particle_Jp",
            "mpm:particle_stress",
            "mpm:particle_transform",
        ):
            builder.custom_attributes.pop(state_attr, None)
        import newton

        kinematic_flag = int(newton.BodyFlags.KINEMATIC)
        for body_id, flags in enumerate(builder.body_flags):
            if int(flags) & kinematic_flag:
                builder.body_mass[body_id] = 0.0
                builder.body_inv_mass[body_id] = 0.0
                builder.body_inertia[body_id] = wp.mat33()
                builder.body_inv_inertia[body_id] = wp.mat33()

    @classmethod
    def _create_solver(cls, model, solver_cfg):
        from .solver import ParticleNeRDSolver

        return ParticleNeRDSolver(model, solver_cfg)


class NewtonParticleNeRDCouplerManager(NewtonCouplerManager):
    """Capture neural/rigid steps while fixed-size particle recycling updates GPU buffers."""

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
        # Release finalized host-side builder particle lists before SolverCoupled
        # constructs its per-entry views so 512+ environment runs do not swap.
        import gc

        from isaaclab.sim import SimulationContext

        sim = SimulationContext.instance()
        if sim is not None:
            for _, resource in getattr(sim, "_backend_registry", ()):
                if hasattr(resource, "particle_q") and isinstance(getattr(resource, "particle_q", None), list):
                    for attr_name in (
                        "particle_q",
                        "particle_qd",
                        "particle_mass",
                        "particle_radius",
                        "particle_flags",
                        "particle_world",
                        "particle_mu",
                        "particle_cohesion",
                        "particle_adhesion",
                    ):
                        lst = getattr(resource, attr_name, None)
                        if isinstance(lst, list):
                            lst.clear()
                    for attr in getattr(resource, "custom_attributes", {}).values():
                        if getattr(getattr(attr, "frequency", None), "name", None) == "PARTICLE":
                            if isinstance(getattr(attr, "values", None), list):
                                attr.values.clear()
            gc.collect()
        super()._build_solver(model, solver_cfg)

    @classmethod
    def _initialize_contacts(cls) -> None:
        model = cls.backend.model
        saved_particle_count = int(model.particle_count)
        skip_soft_pairs = cls._collision_cfg is not None and getattr(cls._collision_cfg, "soft_contact_max", None) == 0
        try:
            if skip_soft_pairs:
                model.particle_count = 0
            super()._initialize_contacts()
        finally:
            if skip_soft_pairs:
                model.particle_count = saved_particle_count

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
        from .solver import ParticleNeRDSolver

        neural_solvers = []
        for entry in cls._solver._entries.values():
            if isinstance(entry.solver, ParticleNeRDSolver):
                entry.solver.capture_collider_state(state_0, entry.body_local_to_global)
                neural_solvers.append(entry.solver)
        if not PhysicsManager._cfg.capture_solver:
            cls._solver.step(state_0, state_1, control, contacts, substep_dt)
            return
        phases = tuple(solver.selection_phase for solver in neural_solvers)
        signature = (id(cls._solver), cls._get_cuda_graph_model_signature())
        if signature != cls._solver_graph_signature:
            cls._solver_graphs = {}
            cls._solver_graph_pool = None
            cls._solver_graph_signature = signature
        key = (id(state_0), id(state_1), id(control), id(contacts), substep_dt, phases)
        graph = cls._solver_graphs.get(key)
        if graph is None:
            # Execute once, then record the same selection phase without advancing simulation time.
            cls._solver.step(state_0, state_1, control, contacts, substep_dt)
            for solver, phase in zip(neural_solvers, phases, strict=True):
                solver.selection_phase = phase
            stream = wp.get_stream(cls.backend.model.device)
            if cls._solver_graph_pool is None:
                cls._solver_graph_pool = torch.cuda.graph_pool_handle()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=cls._solver_graph_pool, stream=wp.stream_to_torch(stream)):
                with wp.ScopedCapture(device=cls.backend.model.device, stream=stream, external=True):
                    cls._solver.step(state_0, state_1, control, contacts, substep_dt)
            cls._solver_graphs[key] = graph
        else:
            graph.replay()
            for solver in neural_solvers:
                solver.selection_phase = (solver.selection_phase + 1) % solver.cfg.selection_interval
