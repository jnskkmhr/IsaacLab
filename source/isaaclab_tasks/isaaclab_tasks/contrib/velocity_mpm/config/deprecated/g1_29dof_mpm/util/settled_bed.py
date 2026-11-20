# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task lifecycle for one-time bed settling; Newton owns particle snapshots and sleep state."""

from __future__ import annotations

import logging
import math

import newton
import torch
import warp as wp
from isaaclab_newton.physics import reset_mpm_solver_state

logger = logging.getLogger(__name__)


class SettledBed:
    """Prepare the actual sand bed once and restore selected environments on reset.

    Robot dynamics, RL counters and rendering do not advance during preparation. The initial
    material snapshot is reused as an approximation after material randomization. Particles
    still moving at the end of the settling budget retain their velocity and awake state.
    """

    def __init__(self, env):
        self.backend = env.sim.backend_manager.backend
        self.entry = self.backend.solver._entries["sand"]
        self.solver = self.entry.solver
        self.dt = env.physics_dt
        self.selected = torch.zeros(self.backend.model.world_count + 1, dtype=torch.bool, device=env.device)
        self.world_mask = wp.from_torch(self.selected, dtype=wp.bool)
        if not math.isfinite(env.cfg.mpm_settle_time) or env.cfg.mpm_settle_time <= 0:
            raise ValueError("mpm_settle_time must be finite and positive when caching the initial state.")
        if env.cfg.reset_particle_jitter != 0.0:
            raise ValueError("Cached settled beds require reset_particle_jitter=0; jitter invalidates equilibrium.")
        self.steps = math.ceil(env.cfg.mpm_settle_time / self.dt)

        # Synchronize startup properties and entry state before advancing only the sand solver.
        self.backend.forward()
        # Exclude JOINT flags: MJWarp otherwise replaces the authored robot pose/velocity
        # with model defaults while leaving the pre-reset body transforms inconsistent.
        reset_mpm_solver_state(self.backend, flags=newton.StateFlags.BODY | newton.StateFlags.PARTICLE)
        state = self.entry.state_0
        body_velocity = wp.clone(state.body_qd)

        def settle_step():
            state.body_qd.zero_()
            if state.particle_f is not None:
                state.particle_f.zero_()
            self.solver.step(state, state, None, None, self.dt)

        logger.info(
            "Preparing initial sand snapshot: %d steps (%.3f s simulated time).", self.steps, self.steps * self.dt
        )
        settle_step()  # Load kernels outside capture.
        if self.steps > 1:
            with wp.ScopedCapture(device=self.solver.model.device) as capture:
                settle_step()
            for _ in range(self.steps - 1):
                wp.capture_launch(capture.graph)
        self.solver.check_sparse_grid_rebuild_status()
        wp.copy(state.body_qd, body_velocity)
        arrays = [state.particle_q, state.particle_qd] + [
            getattr(state.mpm, name)
            for name in (
                "particle_elastic_strain",
                "particle_transform",
                "particle_qd_grad",
                "particle_stress",
                "particle_Jp",
            )
        ]
        if any(not torch.isfinite(wp.to_torch(array)).all().item() for array in arrays):
            raise RuntimeError("Initial sand settling produced non-finite particle state; snapshot rejected.")
        self.snapshot = self.solver.save_particle_state(state)
        if self.solver.particle_activity is not None:
            counts = torch.bincount(wp.to_torch(self.solver.particle_activity).long(), minlength=3).tolist()
            logger.info("Initial sand snapshot: sleeping=%d, support=%d, awake=%d particles.", *counts)
        self.restore(torch.arange(env.num_envs, device=env.device))

    def restore(self, env_ids):
        """Restore GPU buffers and coupling state without settling or stepping physics."""
        self.selected.zero_()
        self.selected[env_ids] = True
        # Consume authored robot poses/material changes before restoring quiet history. A later
        # ordinary forward() must not clear the restored activity again.
        self.backend.forward()
        reset_mpm_solver_state(
            self.backend, world_mask=self.world_mask, flags=newton.StateFlags.BODY | newton.StateFlags.PARTICLE
        )
        self.solver.restore_particle_state(self.entry.state_0, self.snapshot, world_mask=self.world_mask)
        coupler = self.backend.solver
        # Entry ownership maps, rather than a 1:1 particle ordering assumption, synchronize both
        # output and canonical buffers. No task code accesses Newton's private activity arrays.
        coupler._sync_entry_reset_state(self.entry, self.world_mask, newton.StateFlags.PARTICLE)
        for state in {id(s): s for s in (self.backend.state_0, self.backend.state_1)}.values():
            coupler._reconcile_reset_state(state, self.world_mask, newton.StateFlags.PARTICLE)
