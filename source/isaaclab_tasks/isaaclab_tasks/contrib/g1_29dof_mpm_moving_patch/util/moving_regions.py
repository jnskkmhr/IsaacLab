# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Resolve task body selections and connect them to Newton's GPU region controller."""

import re

import warp as wp
from isaaclab_newton.physics import MPMWorkingSetCfg, configure_mpm_working_set

from ..env_cfg.physics_cfg import MPM_ENTRY


class MovingRegions:
    """Keep a stable pose input buffer populated inside the captured physics step."""

    def __init__(self, env):
        manager = env.sim.backend_manager
        self.backend = manager.backend
        self.entry = self.backend.solver._entries[MPM_ENTRY]
        solver = self.entry.solver
        cfg = env.cfg.moving_patch
        indices, extents, offsets = [], [], []
        for region in cfg.regions:
            matches = [
                i for i, label in enumerate(self.backend.model.body_label) if re.fullmatch(region.body_path, label)
            ]
            if not matches:
                raise ValueError(f"Moving MPM region has no matching bodies: {region.body_path!r}")
            indices.extend(matches)
            extents.extend([region.half_extents] * len(matches))
            offsets.extend([region.offset] * len(matches))
        self.body_q = wp.clone(self.backend.state_0.body_q)
        solver.set_particle_active_regions(
            body_indices=indices,
            half_extents=extents,
            offsets=offsets,
            body_q=self.body_q,
            body_world=self.backend.model.body_world,
            keep_material_motion=cfg.keep_material_motion,
        )
        # Copy after actuator state swaps and before the coupled solver. The copy is captured,
        # so body tracking requires neither Python callbacks nor CPU pose reads during replay.
        manager.register_post_actuator_callback(self.update_input)
        solver.update_particle_activity(self.entry.state_0, env.physics_dt)
        capacity = solver.model.particle_count if cfg.working_set_capacity is None else cfg.working_set_capacity
        configure_mpm_working_set(
            manager,
            MPMWorkingSetCfg(
                particle_capacity=capacity,
                page_size=cfg.working_set_page_size,
            ),
            solver=solver,
            state=self.entry.state_0,
        )

    def update_input(self):
        """Copy current global body transforms into the persistent region input buffer."""
        wp.copy(self.body_q, self.backend.state_0.body_q)
