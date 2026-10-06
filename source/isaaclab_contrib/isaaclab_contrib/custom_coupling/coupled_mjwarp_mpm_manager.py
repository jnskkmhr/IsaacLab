# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Direct MJWarp/MPM wrench exchange following Newton's two-way MPM example."""

from __future__ import annotations

import warp as wp
from isaaclab_newton.physics import MJWarpSolverCfg, MPMSolverCfg, NewtonManager
from newton import Contacts, Control, Model, ModelFlags, ShapeFlags, State, StateFlags
from newton.solvers.experimental.coupled import ModelView, SolverCoupled

from isaaclab_contrib.coupling.coupler import NewtonCouplerManager

from .kernels import clear_mpm_body_wrenches, collect_mpm_body_wrenches, prepare_mpm_collider_state
from .newton_manager_cfg import CoupledMJWarpMPMSolverCfg


class DirectSolverCoupler(SolverCoupled):
    """Exchange lagged grid-contact wrenches without Newton's proxy or ADMM algorithms.

    Solver entries provide ownership, rigid contact filtering, and reset handling.
    MPM receives the selected rigid bodies as external colliders. Its predicted collider
    velocity excludes the previous MPM wrench, as in ``example_mpm_twoway_coupling.py``.
    """

    def __init__(
        self,
        model: Model,
        entries: list[SolverCoupled.Entry],
        rigid_entry: str,
        mpm_entry: str,
        collider_bodies: list[int],
        terrain_shapes: list[int],
    ) -> None:
        super().__init__(model, entries)
        self.rigid_entry = rigid_entry
        self.mpm_entry = mpm_entry
        self.coupling_forces = wp.zeros(model.body_count, dtype=wp.spatial_vector, device=model.device)
        self._collider_body_q = wp.clone(model.body_q)
        self._collider_body_qd = wp.clone(model.body_qd)
        self._collider_body_f = wp.zeros_like(self.coupling_forces)
        self._collider_model = ModelView(model, "direct_mpm_colliders")
        self._collider_bodies = set(collider_bodies)
        self._terrain_shapes = set(terrain_shapes)
        self._configure_colliders()

    def _configure_colliders(self) -> None:
        """Present only selected robot shapes and the MPM supporting terrain to MPM."""
        shape_flags = self.model.shape_flags.numpy().copy()
        particle_collision = int(ShapeFlags.COLLIDE_PARTICLES)
        for shape, body in enumerate(self.model.shape_body.numpy()):
            if int(body) in self._collider_bodies or shape in self._terrain_shapes:
                shape_flags[shape] |= particle_collision
            else:
                shape_flags[shape] &= ~particle_collision
        self._collider_model.shape_flags = wp.array(shape_flags, dtype=int, device=self.model.device)
        self.solver(self.mpm_entry).setup_collider(model=self._collider_model)

    def notify_model_changed(self, flags: ModelFlags | int) -> None:
        super().notify_model_changed(flags)
        if flags & (ModelFlags.BODY_INERTIAL_PROPERTIES | ModelFlags.BODY_PROPERTIES | ModelFlags.SHAPE_PROPERTIES):
            self._configure_colliders()

    def _step_coupled(
        self, state_in: State, state_out: State, control: Control | None, contacts: Contacts | None, dt: float
    ) -> None:
        del state_out
        rigid = self._entries[self.rigid_entry]
        sand = self._entries[self.mpm_entry]
        # Preserve external forces and hold the previous MPM wrench throughout the rigid substeps.
        self._add_body_force_input(rigid, rigid.body_local_to_global, self.coupling_forces)
        self._step_entry(rigid, control, contacts, dt)
        wp.copy(self._collider_body_q, state_in.body_q)
        wp.copy(self._collider_body_qd, state_in.body_qd)
        wp.launch(
            prepare_mpm_collider_state,
            dim=rigid.body_local_to_global.shape[0],
            inputs=[
                dt,
                rigid.body_local_to_global,
                rigid.state_1.body_q,
                rigid.state_1.body_qd,
                self.coupling_forces,
                self.model.body_inv_mass,
                self.model.body_inv_inertia,
            ],
            outputs=[self._collider_body_q, self._collider_body_qd],
            device=self.model.device,
        )
        # External collider body arrays belong to the rigid model, not the particle entry's model.
        # Attach them only during MPM evaluation so generic entry reset/reconciliation retains its layout.
        state = sand.state_0
        body_arrays = state.body_q, state.body_qd, state.body_f
        state.body_q = self._collider_body_q
        state.body_qd = self._collider_body_qd
        state.body_f = self._collider_body_f
        self._collider_body_f.zero_()
        try:
            sand.solver.step(state, state, None, None, dt)
            impulses, positions, collider_ids = sand.solver.collect_collider_impulses(state)
            self.coupling_forces.zero_()
            if collider_ids.shape[0]:
                wp.launch(
                    collect_mpm_body_wrenches,
                    dim=collider_ids.shape[0],
                    inputs=[
                        dt,
                        collider_ids,
                        impulses,
                        positions,
                        sand.solver.collider_body_index,
                        self._collider_body_q,
                        self.model.body_com,
                    ],
                    outputs=[self.coupling_forces],
                    device=self.model.device,
                )
        finally:
            state.body_q, state.body_qd, state.body_f = body_arrays

    def _reset_coupling_state(
        self, state: State, *, world_mask: wp.array[wp.bool] | None = None, flags: StateFlags | int | None = None
    ) -> None:
        del state, flags
        if world_mask is None:
            self.coupling_forces.zero_()
        else:
            wp.launch(
                clear_mpm_body_wrenches,
                dim=self.model.body_count,
                inputs=[self.model.body_world, world_mask],
                outputs=[self.coupling_forces],
                device=self.model.device,
            )


class NewtonCoupledMJWarpMPMManager(NewtonCouplerManager):
    """Build the direct MJWarp/MPM coupler using the standard solver-entry lifecycle."""

    _solver_config_types = (CoupledMJWarpMPMSolverCfg,)

    @classmethod
    def _build_solver(cls, model: Model, solver_cfg: CoupledMJWarpMPMSolverCfg) -> None:
        cls._validate_config(solver_cfg)
        if NewtonManager._report_contacts:
            raise NotImplementedError("Direct MJWarp/MPM coupling does not support Newton contact sensors")
        configs = {entry.name: entry for entry in solver_cfg.entries}
        if solver_cfg.rigid_entry == solver_cfg.mpm_entry or set(configs) != {
            solver_cfg.rigid_entry,
            solver_cfg.mpm_entry,
        }:
            raise ValueError("Direct coupling requires exactly one named MJWarp entry and one named MPM entry")
        rigid_cfg = configs[solver_cfg.rigid_entry]
        sand_cfg = configs[solver_cfg.mpm_entry]
        if not isinstance(rigid_cfg.solver_cfg, MJWarpSolverCfg) or not isinstance(sand_cfg.solver_cfg, MPMSolverCfg):
            raise TypeError("Direct coupling requires MJWarpSolverCfg and MPMSolverCfg entries")
        if sand_cfg.substeps != 1 or sand_cfg.solver_cfg.collider_velocity_mode != "forward":
            raise ValueError("Direct MPM coupling requires one MPM step and forward collider velocities")
        resolved = [cls._resolve_entry(model, entry) for entry in solver_cfg.entries]
        cls._validate_resolved_entries(model, resolved, solver_cfg, set())
        rigid = next(entry for entry in resolved if entry.config.name == solver_cfg.rigid_entry)
        sand = next(entry for entry in resolved if entry.config.name == solver_cfg.mpm_entry)
        if sand.bodies or rigid.particles:
            raise ValueError(
                "The MPM entry must own only particles and terrain shapes; MJWarp must own only rigid bodies"
            )
        bodies = cls._resolve_entities_to_body_ids(model, solver_cfg.collider_bodies, "direct MPM colliders")
        if not bodies or not set(bodies).issubset(rigid.bodies):
            raise ValueError("Direct MPM collider bodies must be owned by the MJWarp entry")
        NewtonManager._solver = DirectSolverCoupler(
            model,
            [cls._build_entry(entry) for entry in resolved],
            solver_cfg.rigid_entry,
            solver_cfg.mpm_entry,
            bodies,
            sand.shapes,
        )
        NewtonManager._use_single_state = False
        NewtonManager._supports_contact_sensors = False
        NewtonManager._needs_collision_pipeline = cls._requires_external_contacts(rigid_cfg.solver_cfg)
        NewtonManager._supports_rigid_body_force_input = True
