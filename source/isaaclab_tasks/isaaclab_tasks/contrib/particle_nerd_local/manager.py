# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Isaac Lab Newton solver manager for ParticleNeRDLocalSolver."""

import torch
import warp as wp

from isaaclab.physics import PhysicsManager

from ..particle_nerd.manager import NewtonParticleNeRDCouplerManager, NewtonParticleNeRDManager
from .solver_cfg import ParticleNeRDLocalSolverCfg


class NewtonParticleNeRDLocalManager(NewtonParticleNeRDManager):
    """Construct a ParticleNeRDLocalSolver through the standard CouplerEntryCfg factory."""

    @classmethod
    def _create_solver(cls, model, solver_cfg):
        from .solver import ParticleNeRDLocalSolver

        return ParticleNeRDLocalSolver(model, solver_cfg)


class NewtonParticleNeRDLocalCouplerManager(NewtonParticleNeRDCouplerManager):
    """Coupler manager supporting ParticleNeRDLocalSolverCfg with CUDA Graph capture."""

    @staticmethod
    def _requires_external_contacts(solver_cfg):
        if isinstance(solver_cfg, ParticleNeRDLocalSolverCfg):
            return False
        return NewtonParticleNeRDCouplerManager._requires_external_contacts(solver_cfg)

    @classmethod
    def _step_solver(cls, state_0, state_1, control, contacts, substep_dt):
        from .solver import ParticleNeRDLocalSolver

        neural_solvers = []
        for entry in cls._solver._entries.values():
            if isinstance(entry.solver, ParticleNeRDLocalSolver):
                entry.solver.capture_collider_state(state_0, entry.body_local_to_global)
                neural_solvers.append(entry.solver)
        capture_enabled = getattr(PhysicsManager._cfg, "capture_solver", None)
        if capture_enabled is None:
            capture_enabled = bool(neural_solvers) and all(
                getattr(s.cfg, "capture_solver", True) for s in neural_solvers
            )
        if not capture_enabled:
            cls._solver.step(state_0, state_1, control, contacts, substep_dt)
            if cls._scene_data_backend is not None:
                cls._scene_data_backend.geometry_timestamp += 1
            return
        phases = tuple(solver.selection_phase == 0 for solver in neural_solvers)
        signature = (id(cls._solver), cls._get_cuda_graph_model_signature())
        if signature != cls._solver_graph_signature:
            cls._solver_graphs = {}
            cls._solver_graph_pool = None
            cls._solver_graph_signature = signature
        key = (id(state_0), id(state_1), id(control), id(contacts), substep_dt, phases)
        graph = cls._solver_graphs.get(key)
        if graph is None:
            saved_phases = [solver.selection_phase for solver in neural_solvers]
            cls._solver.step(state_0, state_1, control, contacts, substep_dt)
            for solver, phase in zip(neural_solvers, saved_phases, strict=True):
                solver.selection_phase = phase
                solver._has_neural_state = True
            stream = wp.get_stream(cls.backend.model.device)
            if cls._solver_graph_pool is None:
                cls._solver_graph_pool = torch.cuda.graph_pool_handle()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=cls._solver_graph_pool, stream=wp.stream_to_torch(stream)):
                with wp.ScopedCapture(device=cls.backend.model.device, stream=stream, external=True):
                    cls._solver.step(state_0, state_1, control, contacts, substep_dt)
            cls._solver_graphs[key] = graph
        else:
            stream = wp.get_stream(cls.backend.model.device)
            with torch.cuda.stream(wp.stream_to_torch(stream)):
                graph.replay()
            for solver in neural_solvers:
                solver.selection_phase = (solver.selection_phase + 1) % solver.cfg.selection_interval
        if cls._scene_data_backend is not None:
            cls._scene_data_backend.geometry_timestamp += 1
