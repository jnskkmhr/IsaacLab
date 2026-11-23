# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton construction and coupler manager for WorldParticle entries."""

import torch

from isaaclab.physics import PhysicsManager

from ..particle_nerd.manager import NewtonParticleNeRDCouplerManager, NewtonParticleNeRDManager
from .solver_cfg import ParticleWorldSolverCfg


class NewtonParticleWorldManager(NewtonParticleNeRDManager):
    """Construct a ParticleWorldSolver through the standard CouplerEntryCfg factory."""

    @classmethod
    def _create_solver(cls, model, solver_cfg):
        from .solver import ParticleWorldSolver

        return ParticleWorldSolver(model, solver_cfg)


class NewtonParticleWorldCouplerManager(NewtonParticleNeRDCouplerManager):
    """Coupler manager supporting ParticleWorldSolverCfg."""

    @staticmethod
    def _requires_external_contacts(solver_cfg):
        if isinstance(solver_cfg, ParticleWorldSolverCfg):
            return False
        return NewtonParticleNeRDCouplerManager._requires_external_contacts(solver_cfg)

    @classmethod
    def _step_solver(cls, state_0, state_1, control, contacts, substep_dt):
        from .solver import ParticleWorldSolver

        neural_solvers = []
        for entry in cls._solver._entries.values():
            if isinstance(entry.solver, ParticleWorldSolver):
                entry.solver.capture_collider_state(state_0, entry.body_local_to_global)
                neural_solvers.append(entry.solver)
        if not getattr(PhysicsManager._cfg, "capture_solver", False):
            cls._solver.step(state_0, state_1, control, contacts, substep_dt)
            if cls._scene_data_backend is not None:
                cls._scene_data_backend.geometry_timestamp += 1
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
            if cls._solver_graph_pool is None:
                cls._solver_graph_pool = torch.cuda.graph_pool_handle()
            for _ in range(2):
                cls._solver.step(state_0, state_1, control, contacts, substep_dt)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=cls._solver_graph_pool):
                cls._solver.step(state_0, state_1, control, contacts, substep_dt)
            cls._solver_graphs[key] = graph
        graph.replay()
        if cls._scene_data_backend is not None:
            cls._scene_data_backend.geometry_timestamp += 1
