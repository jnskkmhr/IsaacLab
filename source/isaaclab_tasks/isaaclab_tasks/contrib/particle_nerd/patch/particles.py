# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fixed-size GPU particle storage for independent recycling patches."""

import newton
import numpy as np
import warp as wp
from newton.solvers.experimental.coupled import SolverCoupled

from isaaclab_tasks.contrib.g1_29dof_mpm_moving_patch.util.kernel import (
    mark_resets,
    restore_boundary_particles,
    update_patch_center,
)
from isaaclab_tasks.contrib.g1_29dof_mpm_moving_patch.util.terrain import MovingPatchTerrainCfg, WarpTerrainMesh

from .kernels import update_particles


class MovingPatchParticles:
    """
    This class manages particle states during the simulation.
    It handles
    - Tracking the positions and states of particles within each patch.
    - Managing particle resets and reinitialization when they leave the patch or when the environment requests it.
    - Updating patch centers based on the tracked robot body positions.
    - Handling dynamic and static particles, including mass assignment and neural history reset.
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
        self.update(state)
        self.check_terrain_queries()

    def _setup_solver(self, coupled_solver: SolverCoupled, entry_name: str) -> None:
        """Resolve the neural solver and bind the static terrain chart."""
        self.solver = coupled_solver.solver(entry_name)
        if self.solver.model.particle_count != self.model.particle_count:
            raise ValueError("Moving patch requires the particle entry to own all particles in original order")
        if not hasattr(self.solver, "history_wp"):
            raise TypeError("Moving NeRD patches require a ParticleNeRD solver")
        self.solver.bind_terrain(self.background_mesh, self.terrain.particle_depth)

    def _setup_tracked_bodies(self) -> None:
        """Find one tracked rigid body in each simulation world."""
        body_worlds = self.model.body_world.numpy()
        world_matches: dict[int, list[int]] = {}
        target = self.terrain.tracked_body
        for i, label in enumerate(self.model.body_label):
            if label and label.rsplit("/", 1)[-1] == target:
                world_matches.setdefault(int(body_worlds[i]), []).append(i)
        bodies = []
        for world in range(self.model.world_count):
            matches = world_matches.get(world, [])
            if len(matches) != 1:
                raise ValueError(f"Expected one {self.terrain.tracked_body!r} in world {world}, got {matches}")
            bodies.append(matches[0])
        self.bodies = wp.array(bodies, dtype=int, device=self.model.device)

    def _setup_particle_storage(self) -> None:
        """Allocate patch centers, reference positions, and reusable reset buffers."""
        model = self.model
        particle_env_ids = model.particle_world.numpy()
        # Mixed-contact tasks may leave some worlds particle-free.
        if np.any(particle_env_ids < 0) or np.any(particle_env_ids >= model.world_count):
            raise ValueError("Moving-patch particles must belong to valid simulation worlds")
        origins = self.background_mesh.env_origins
        if origins.shape != (model.world_count, 3):  # type: ignore
            raise ValueError("Shared terrain origins must match the number of Newton worlds")
        # The center-update kernel uses these origins to quantize patch movement.
        self.origins = wp.array(origins, dtype=wp.vec3, device=model.device)
        self.patch_centers = wp.array(origins[:, :2], dtype=wp.vec2, device=model.device)  # type: ignore
        particle_template_positions = model.particle_q.numpy()
        particle_template_positions[:, 2] -= self.background_mesh.env_clone_origins[particle_env_ids, 2]  # type: ignore
        self.particle_template_positions = wp.array(particle_template_positions, dtype=wp.vec3, device=model.device)
        self.particle_reset_positions = wp.clone(model.particle_q)
        self.particle_is_dynamic = wp.ones(model.particle_count, dtype=int, device=model.device)
        self.particle_dynamic_mass = wp.clone(model.particle_mass)
        if np.any(self.particle_dynamic_mass.numpy() <= 0):
            raise ValueError("Initial sand masses must be positive; the wrapper assigns zero boundary mass")
        self.env_reset_requested = wp.ones(model.world_count, dtype=int, device=model.device)
        self.terrain_query_miss_count = wp.zeros(1, dtype=int, device=model.device)

    def update(self, state: newton.State) -> None:
        """Move centers and recycle particles before one coupled physics substep."""
        cfg = self.terrain
        model = self.model
        wp.launch(
            update_patch_center,
            dim=model.world_count,
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
        wp.launch(
            update_particles,
            dim=model.particle_count,
            inputs=[
                model.particle_world,  # particle_env_ids
                self.patch_centers,  # patch_centers
                self.particle_template_positions,  # particle_template_positions
                self.env_reset_requested,  # env_reset_requested
                self.particle_dynamic_mass,  # particle_dynamic_mass
                wp.vec2(*cfg.simulated_terrain_size),  # simulated_terrain_size
                wp.vec2(*cfg.total_patch_size),  # total_patch_size
                self.background_mesh.mesh.id,  # background_mesh
                self.background_mesh.height_query_start_z,  # ray_z
                self.background_mesh.height_query_max_distance,  # ray_length
            ],
            outputs=[
                self.particle_reset_positions,  # particle_reset_positions
                self.particle_is_dynamic,  # particle_is_dynamic
                state.particle_q,  # particle_q
                state.particle_qd,  # particle_qd
                model.particle_mass,  # particle_mass
                model.particle_inv_mass,  # particle_inv_mass
                self.solver.history_wp,  # neural position history
                self.terrain_query_miss_count,  # terrain_query_miss_count
            ],
            device=model.device,
        )
        self.env_reset_requested.zero_()
        # All slots stay valid, so their inference buffers and graph bindings remain fixed.
        self._refresh_particle_mass()

    def check_terrain_queries(self) -> None:
        """Report failed terrain queries outside CUDA capture, once per control step."""
        if self.terrain_query_miss_count.numpy()[0]:
            raise ValueError("Recycled particles left the generated terrain surface")
        self.terrain_query_miss_count.zero_()

    def _refresh_particle_mass(self) -> None:
        """Copy boundary mass changes into the particle entry without rebinding slots."""
        for name in ("particle_mass", "particle_inv_mass"):
            wp.copy(getattr(self.solver.model, name), getattr(self.model, name))

    def restore_boundary_particles(self, state: newton.State) -> None:
        """Remove numerical boundary drift after neural prediction."""
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
