# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Connect the reference particle cache to the current coupled backend."""

from types import SimpleNamespace

import newton
import numpy as np
import torch
import warp as wp

from .cache.particles import MovingPatchParticles
from .terrain_mesh import WarpTerrainMesh


class PagedPatch:
    def __init__(self, env):
        self.env = env
        self.backend = env.sim.backend_manager.backend
        self.entry = self.backend.solver._entries["sand"]
        ids = self.entry.particle_indices.numpy()
        if not np.array_equal(ids, np.arange(self.backend.model.particle_count)):
            raise ValueError("Paged sand requires the sand entry to own all particle slots in canonical order")
        cfg = env.cfg.scene.sand.spawn
        generator = cfg.terrain_generator.class_type(cfg.terrain_generator.copy(), device="cpu")
        origins = env.scene.env_origins.cpu().numpy()
        if env.num_envs != 1:
            raise ValueError("Generated particle-cache playback currently supports num_envs=1")
        mesh = generator.terrain_mesh.copy()
        mesh.apply_translation(origins[0])
        background = WarpTerrainMesh(mesh.vertices, mesh.faces, device=env.device)
        background.env_origins = origins
        self.patch = MovingPatchParticles(
            self.backend.model,
            SimpleNamespace(
                particle_q=self.entry.state_0.particle_q,
                particle_qd=self.entry.state_0.particle_qd,
                body_q=self.backend.state_0.body_q,
            ),
            self.backend.solver,
            cfg.paged_patch,
            "sand",
            background,
        )
        self.world_mask = wp.ones(env.num_envs + 1, dtype=wp.bool, device=env.device)
        self.sync()
        env.sim.backend_manager.register_post_step_callback(self.restore_boundaries)

    def sync(self):
        coupler = self.backend.solver
        coupler._sync_entry_reset_state(self.entry, self.world_mask, newton.StateFlags.PARTICLE)
        for state in {id(s): s for s in (self.backend.state_0, self.backend.state_1)}.values():
            coupler._reconcile_reset_state(state, self.world_mask, newton.StateFlags.PARTICLE)
        self.env._sand._invalidate_caches(pos=True, vel=True)

    def update(self):
        self.entry.solver.check_sparse_grid_rebuild_status()
        # Keep captured state objects unchanged; paging receives views of the current buffers.
        state = SimpleNamespace(
            particle_q=self.entry.state_0.particle_q,
            particle_qd=self.entry.state_0.particle_qd,
            body_q=self.backend.state_0.body_q,
        )
        changed = self.patch.update(state)
        if changed:
            self.sync()
            if self.env._sand_surface is not None:
                self.env._sand_surface.invalidate(None)
        self.patch.check_terrain_queries()

    def restore_boundaries(self):
        self.patch.restore_boundary_particles(self.backend.state_0)
        self.patch.restore_boundary_particles(self.entry.state_0)

    def reset(self, env_ids):
        mask = torch.zeros(self.env.num_envs + 1, dtype=torch.bool, device=self.env.device)
        mask[env_ids] = True
        self.patch.reset(wp.from_torch(mask, dtype=wp.bool))
