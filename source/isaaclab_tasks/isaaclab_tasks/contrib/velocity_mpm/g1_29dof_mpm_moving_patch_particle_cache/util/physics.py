# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Capture coupled physics while particle paging remains on the CPU timeline."""

from typing import ClassVar

import newton
import warp as wp
from isaaclab_newton.physics import NewtonMPMManager

from isaaclab.physics import PhysicsManager

from isaaclab_contrib.coupling.coupler import NewtonCouplerManager


class NewtonParticleCacheCouplerManager(NewtonCouplerManager):
    """Replay the solver between the manager's eager particle-cache callbacks.

    State-force callbacks page particles before the solver, and post-step callbacks
    restore the fixed boundary afterwards. Neither callback is recorded. Actions,
    material randomizers and cache loads update the solver's existing GPU buffers.
    Changes to material-dependent branches or buffer addresses trigger recapture.
    """

    _solver_graphs: ClassVar[dict[tuple, wp.Graph]] = {}
    _solver_graph_signature: ClassVar[tuple | None] = None

    @classmethod
    def _step_solver(
        cls,
        state_0: newton.State,
        state_1: newton.State,
        control: newton.Control,
        contacts: newton.Contacts | None,
        substep_dt: float,
    ) -> None:
        if not PhysicsManager._cfg.capture_solver or not cls.backend.model.device.is_cuda:
            super()._step_solver(state_0, state_1, control, contacts, substep_dt)
            return
        if not cls._supports_cuda_graph_capture():
            raise ValueError("Particle-cache solver capture requires fixed-capacity, unpadded sparse MPM storage")

        signature = (id(cls._solver), cls._get_cuda_graph_model_signature())
        if signature != cls._solver_graph_signature:
            cls._solver_graphs = {}
            cls._solver_graph_signature = signature
        key = (id(state_0), id(state_1), id(control), id(contacts), substep_dt)
        graph = cls._solver_graphs.get(key)
        if graph is None:
            # Execute this requested step once to initialize lazy solver resources.
            # Capture records the following launches without advancing GPU state.
            super()._step_solver(state_0, state_1, control, contacts, substep_dt)
            cls._solver_graphs[key] = cls._capture_graph(
                lambda: cls._solver.step(state_0, state_1, control, contacts, substep_dt)
            )
        else:
            wp.capture_launch(graph)

    @classmethod
    def _check_solver_status(cls) -> None:
        super()._check_solver_status()
        # The base MPM check only checks the manager's outer graph. Our graph
        # contains the solver alone, so its asynchronous capacity checks live here.
        if cls._solver_graphs:
            for solver in NewtonMPMManager._implicit_mpm_solvers():
                solver.check_sparse_grid_rebuild_status()

    @classmethod
    def _solver_specific_clear(cls) -> None:
        cls._solver_graphs = {}
        cls._solver_graph_signature = None
        super()._solver_specific_clear()
