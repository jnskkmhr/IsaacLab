# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton factory and two-way coupling for the object NeRD examples."""

from isaaclab.physics import PhysicsManager

from .manager import NewtonParticleNeRDCouplerManager, NewtonParticleNeRDManager
from .object_solver_cfg import ParticleNeRDObjectSolverCfg


class NewtonParticleNeRDObjectManager(NewtonParticleNeRDManager):
    """Register material features and construct NeRD without a native MPM solver."""

    @classmethod
    def _create_solver(cls, model, solver_cfg):
        from .object_solver import ParticleNeRDObjectSolver

        return ParticleNeRDObjectSolver(model, solver_cfg)


class NewtonParticleNeRDObjectCouplerManager(NewtonParticleNeRDCouplerManager):
    """Snapshot pre-rigid neural inputs, then run lagged two-way object coupling."""

    @staticmethod
    def _requires_external_contacts(solver_cfg):
        if isinstance(solver_cfg, ParticleNeRDObjectSolverCfg):
            return False
        return NewtonParticleNeRDCouplerManager._requires_external_contacts(solver_cfg)

    @classmethod
    def _step_solver(cls, state_0, state_1, control, contacts, substep_dt):
        if PhysicsManager._cfg.use_cuda_graph:
            raise ValueError("The interactive object NeRD examples require use_cuda_graph=False")
        from .object_solver import ParticleNeRDObjectSolver

        for entry in cls._solver._entries.values():
            if isinstance(entry.solver, ParticleNeRDObjectSolver):
                entry.solver.capture_collider_state(state_0, entry.body_local_to_global)
        cls._solver.step(state_0, state_1, control, contacts, substep_dt)
        if cls._scene_data_backend is not None:
            cls._scene_data_backend.geometry_timestamp += 1
