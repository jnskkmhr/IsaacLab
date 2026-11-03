# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fixed-size GPU particle storage for independent cached patches."""

import re

import newton
import numpy as np
import torch
import warp as wp
from newton.solvers.experimental.coupled import SolverCoupled

from .kernel import mark_resets, refresh_cached_properties, restore_boundary_particles, update_patch_center
from .particle_cache import ParticleStateCache
from .terrain import MovingPatchTerrainCfg, WarpTerrainMesh


class MovingPatchParticles:
    """
    This class manages particle states during the simulation.
    It handles
    - Tracking the positions and states of particles within each patch.
    - Managing particle resets and reinitialization when they leave the patch or when the environment requests it.
    - Updating patch centers based on the tracked robot body positions.
    - Handling dynamic and static particles, including mass assignment and material state restoration.
    """

    def __init__(
        self,
        model: newton.Model,
        state: newton.State,
        coupled_solver: SolverCoupled,
        terrain: MovingPatchTerrainCfg,
        entry_name: str,
        background_mesh: WarpTerrainMesh,
    ) -> None:
        self.model = model
        self.terrain = terrain
        self.background_mesh = background_mesh
        self._setup_solver(coupled_solver, entry_name)
        self._setup_tracked_bodies()
        self._setup_particle_storage()
        self.cache = ParticleStateCache(self)
        self.update(state)
        self.check_terrain_queries()

    def _setup_solver(self, coupled_solver: SolverCoupled, entry_name: str) -> None:
        """Resolve the MPM entry and its particle material state."""
        self.solver = coupled_solver.solver(entry_name)
        self.material_state = coupled_solver.entry_state(entry_name, phase="input")
        if self.solver.model.particle_count != self.model.particle_count:
            raise ValueError("Moving patch requires the MPM entry to own all particles in original order")
        if not hasattr(self.solver, "_mpm_model"):
            raise RuntimeError("Unsupported implicit MPM implementation: missing derived particle properties")

    def _setup_tracked_bodies(self) -> None:
        """Find the primary robot body and every explicitly tracked object in each world."""
        body_worlds = self.model.body_world.numpy()
        bodies = []
        for world in range(self.model.world_count):
            primary = [
                i
                for i, label in enumerate(self.model.body_label)
                if body_worlds[i] == world and label.rsplit("/", 1)[-1] == self.terrain.tracked_body
            ]
            if len(primary) != 1:
                raise ValueError(f"Expected one {self.terrain.tracked_body!r} in world {world}, got {primary}")
            for pattern in self.terrain.tracked_body_patterns:
                if not any(
                    body_worlds[i] == world and re.fullmatch(pattern, label)
                    for i, label in enumerate(self.model.body_label)
                ):
                    raise ValueError(f"Tracked body pattern {pattern!r} matched no body in world {world}")
            extra = [
                i
                for i, label in enumerate(self.model.body_label)
                if body_worlds[i] == world
                and i not in primary
                and any(re.fullmatch(pattern, label) for pattern in self.terrain.tracked_body_patterns)
            ]
            bodies.append(primary + extra)
        self.body_counts = np.array([len(ids) for ids in bodies], dtype=np.int32)
        indices = np.full((self.model.world_count, max(self.body_counts)), -1, dtype=np.int32)
        for world, ids in enumerate(bodies):
            indices[world, : len(ids)] = ids
        self.bodies = wp.array(indices, dtype=int, device=self.model.device)

    def _setup_particle_storage(self) -> None:
        """Allocate patch centers, reference positions, and reusable reset buffers."""
        model = self.model
        particle_env_ids = model.particle_world.numpy()
        if np.any(particle_env_ids < 0) or np.any(particle_env_ids >= model.world_count):
            raise ValueError("Moving-patch particles must belong to valid simulation worlds")
        origins = self.background_mesh.env_origins
        if origins.shape != (model.world_count, 3):  # type: ignore
            raise ValueError("Shared terrain origins must match the number of Newton worlds")
        # The center-update kernel uses these origins to quantize patch movement.
        self.origins = wp.array(origins, dtype=wp.vec3, device=model.device)
        self.patch_centers = wp.array(
            np.repeat(origins[:, None, :2], self.bodies.shape[1], axis=1), dtype=wp.vec2, device=model.device
        )  # type: ignore
        self.particle_reset_positions = wp.clone(model.particle_q)
        self.particle_is_dynamic = wp.ones(model.particle_count, dtype=int, device=model.device)
        self.particle_dynamic_mass = wp.clone(model.particle_mass)
        if np.any(self.particle_dynamic_mass.numpy() <= 0):
            raise ValueError("Initial sand masses must be positive; the wrapper assigns zero boundary mass")
        self.initial_particle_plastic_volume_ratio = wp.clone(self.material_state.mpm.particle_Jp)
        self.env_reset_requested = wp.ones(model.world_count, dtype=int, device=model.device)
        self.terrain_query_miss_count = wp.zeros(1, dtype=int, device=model.device)

    def set_dynamic_particle_density(self, density: torch.Tensor, env_ids: torch.Tensor | slice | None = None) -> None:
        """Update stored dynamic masses after the backend changes the selected environments' density.

        ``density`` contains one nominal material density per selected environment. This updates
        the masses restored when boundary particles become simulated particles; it does not change
        the current kinematic/dynamic classification or write the solver's material parameters.
        """
        env_ids = slice(None) if env_ids is None else env_ids
        device = str(self.model.device)
        selected_envs = torch.zeros(self.model.world_count, dtype=torch.bool, device=device)
        selected_envs[env_ids] = True
        env_density = torch.zeros(self.model.world_count, device=device)
        env_density[env_ids] = density
        particle_env_ids = wp.to_torch(self.model.particle_world).long()
        selected = selected_envs[particle_env_ids]
        radius = wp.to_torch(self.model.particle_radius)[selected]
        wp.to_torch(self.particle_dynamic_mass)[selected] = env_density[particle_env_ids[selected]] * (8.0 * radius**3)

    def update(self, state: newton.State) -> None:
        """Move centers and recycle particles before one coupled physics substep."""
        cfg = self.terrain
        model = self.model
        wp.launch(
            update_patch_center,
            dim=self.bodies.shape,
            inputs=[
                state.body_q,  # body_q
                self.bodies,  # body_indices
                self.origins,  # env_origin
                wp.vec2(
                    *((self.background_mesh.bounds[0, :2] + self.background_mesh.bounds[1, :2]) / 2)
                ),  # background_terrain_center
                wp.vec2(
                    *(self.background_mesh.bounds[1, :2] - self.background_mesh.bounds[0, :2])
                ),  # background_terrain_size
                wp.vec2(*cfg.total_patch_size),  # total_patch_size
                cfg.patch_discretization_step,  # patch_discretization_step
            ],
            outputs=[self.patch_centers],  # patch_center
            device=model.device,
        )
        if self.cache.update(state):
            self._refresh_particle_mass()
        self.env_reset_requested.zero_()

    def check_terrain_queries(self) -> None:
        """Report failed terrain queries outside CUDA capture, once per control step."""
        if self.terrain_query_miss_count.numpy()[0]:
            raise ValueError("Recycled particles left the generated terrain surface")
        self.terrain_query_miss_count.zero_()

    def _refresh_particle_mass(self) -> None:
        """Synchronize restored particle properties and invalidate solver warm starts."""
        for name in ("particle_mass", "particle_inv_mass", "particle_radius", "particle_flags"):
            wp.copy(getattr(self.solver.model, name), getattr(self.model, name))
        # This task owns only ordinary material particles, with no particle-backed colliders.
        derived = self.solver._mpm_model
        wp.launch(
            refresh_cached_properties,
            dim=self.model.particle_count,
            inputs=[self.model.particle_mass, self.model.particle_radius, self.model.particle_flags],
            outputs=[
                derived.particle_volume,
                derived.particle_density,
                derived.material_particle_volume,
                derived.particle_flags,
                derived.material_particle_flags,
            ],
            device=self.model.device,
        )
        # A full solver reset also clears sparse-grid error status and is forbidden
        # during capture. Recycling only invalidates warm starts and collider history.
        self.solver._clear_reset_warmstarts(None, (None, None))
        self.solver._last_step_data.save_collider_current_position(
            self.material_state.body_q,
            body_world=self.solver.model.body_world,
            world_count=self.solver.model.world_count,
        )

    def restore_boundary_particles(self, state: newton.State) -> None:
        """Remove numerical boundary drift after MPM advection."""
        wp.launch(
            restore_boundary_particles,
            dim=self.model.particle_count,
            inputs=[
                self.particle_is_dynamic,  # particle_is_dynamic
                self.particle_reset_positions,  # particle_reset_positions
                state.particle_q,  # particle_q
                state.particle_qd,  # particle_qd
            ],
            device=self.model.device,
        )

    def reset(self, world_mask: wp.array | None) -> None:
        """Defer selected-world particle reset until after IsaacLab updates robot forward kinematics."""
        if world_mask is None:
            self.env_reset_requested.fill_(1)
        else:
            wp.launch(
                mark_resets,
                dim=self.model.world_count,
                inputs=[
                    world_mask,  # selected
                    self.env_reset_requested,  # pending
                ],
                device=self.model.device,
            )
