# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fixed-size GPU particle storage for independent alpha patches."""

import newton
import numpy as np
import torch
import warp as wp
from newton.solvers.experimental.coupled import SolverCoupled

from .kernel import mark_resets, refresh_density, restore_boundary_particles, update_centers, update_particles
from .terrain import MovingPatchTerrainCfg, WarpTerrainMesh


class MovingPatchParticles:
    """Keep one bounded particle patch per Newton world; discard departed terrain history.

    Initialization reads topology once. Stepping uses GPU arrays, except for one
    scalar indicating whether masses changed. No background particle cache exists.
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

    def _setup_solver(self, coupled_solver: SolverCoupled, entry_name: str) -> None:
        """Resolve the MPM entry and its particle material state."""
        self.solver = coupled_solver.solver(entry_name)
        self.material_state = coupled_solver.entry_state(entry_name, phase="input")
        if self.solver.model.particle_count != self.model.particle_count:
            raise ValueError("Moving patch requires the MPM entry to own all particles in original order")
        if not hasattr(self.solver, "_mpm_model"):
            raise RuntimeError("Unsupported implicit MPM implementation: missing derived particle properties")

    def _setup_tracked_bodies(self) -> None:
        """Find one tracked robot body in each simulation world."""
        body_worlds = self.model.body_world.numpy()
        bodies = []
        for world in range(self.model.world_count):
            matches = [
                i
                for i, label in enumerate(self.model.body_label)
                if body_worlds[i] == world
                and label
                and "/Robot/" in label
                and label.rsplit("/", 1)[-1] == self.terrain.tracked_body
            ]
            if len(matches) != 1:
                raise ValueError(f"Expected one {self.terrain.tracked_body!r} in world {world}, got {matches}")
            bodies.append(matches[0])
        self.bodies = wp.array(bodies, dtype=int, device=self.model.device)

    def _setup_particle_storage(self) -> None:
        """Allocate patch centers, reference positions, and reusable reset buffers."""
        model = self.model
        worlds = model.particle_world.numpy()
        if np.any(worlds < 0) or not np.all(np.bincount(worlds, minlength=model.world_count)):
            raise ValueError("Each environment must contain sand; global particles are unsupported")
        origins = self.background_mesh.spawn_origins
        if origins.shape != (model.world_count, 3):  # type: ignore
            raise ValueError("Shared terrain origins must match the number of Newton worlds")
        # The center-update kernel uses these origins to quantize patch movement.
        self.origins = wp.array(origins, dtype=wp.vec3, device=model.device)
        self.centers = wp.array(origins[:, :2], dtype=wp.vec2, device=model.device)  # type: ignore
        reference = model.particle_q.numpy()
        reference[:, 2] -= self.background_mesh.initial_origins[worlds, 2]  # type: ignore
        self.reference = wp.array(reference, dtype=wp.vec3, device=model.device)
        self.anchors = wp.clone(model.particle_q)
        self.dynamic = wp.ones(model.particle_count, dtype=int, device=model.device)
        self.dynamic_mass = wp.clone(model.particle_mass)
        if np.any(self.dynamic_mass.numpy() <= 0):
            raise ValueError("Initial sand masses must be positive; the wrapper assigns zero boundary mass")
        self.initial_plastic = wp.clone(self.material_state.mpm.particle_Jp)
        self.pending_reset = wp.ones(model.world_count, dtype=int, device=model.device)
        self.changed = wp.zeros(2, dtype=int, device=model.device)

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
        wp.to_torch(self.dynamic_mass)[selected] = env_density[particle_env_ids[selected]] * (8.0 * radius**3)

    def update(self, state: newton.State) -> None:
        """Move centers and recycle particles before one coupled physics substep."""
        cfg = self.terrain
        model = self.model
        wp.launch(
            update_centers,
            dim=model.world_count,
            inputs=[
                state.body_q,  # body_q
                self.bodies,  # bodies
                self.origins,  # origins
                self.centers,  # centers
                wp.vec2(
                    *((self.background_mesh.bounds[0, :2] + self.background_mesh.bounds[1, :2]) / 2)
                ),  # background_center
                wp.vec2(*(self.background_mesh.bounds[1, :2] - self.background_mesh.bounds[0, :2])),  # background
                wp.vec2(*cfg.patch_size),  # outer
                cfg.shift_step,  # shift
            ],
            device=model.device,
        )
        self.changed.zero_()
        material = self.material_state.mpm
        wp.launch(
            update_particles,
            dim=model.particle_count,
            inputs=[
                model.particle_world,  # worlds
                self.centers,  # centers
                self.reference,  # reference
                self.anchors,  # anchors
                self.dynamic,  # dynamic
                self.pending_reset,  # reset
                wp.vec2(*cfg.moving_terrain_size),  # moving
                wp.vec2(*cfg.patch_size),  # outer
                state.particle_q,  # q
                state.particle_qd,  # qd
                model.particle_mass,  # mass
                model.particle_inv_mass,  # inv_mass
                self.dynamic_mass,  # dynamic_mass
                self.changed,  # changed
                material.particle_qd_grad,  # grad
                material.particle_elastic_strain,  # elastic
                material.particle_Jp,  # plastic
                self.initial_plastic,  # initial_plastic
                material.particle_stress,  # stress
                material.particle_transform,  # transform
                self.background_mesh.mesh.id,  # surface_mesh
                self.background_mesh.ray_z,  # ray_z
                self.background_mesh.ray_length,  # ray_length
            ],
            device=model.device,
        )
        self.pending_reset.zero_()
        changed = self.changed.numpy()
        if changed[1]:
            raise ValueError("Recycled particles left the generated terrain surface")
        if changed[0]:
            for name in ("particle_mass", "particle_inv_mass"):
                wp.copy(getattr(self.solver.model, name), getattr(model, name))
            # Mass is the only changing material property. Radius, volume, ACTIVE flags,
            # collider membership and constitutive parameters stay fixed.
            derived = self.solver._mpm_model
            wp.launch(
                refresh_density,
                dim=model.particle_count,
                inputs=[
                    self.solver.model.particle_mass,  # mass
                    derived.particle_volume,  # volume
                    derived.particle_density,  # density
                ],
                device=model.device,
            )
            self.solver.reset(self.material_state, flags=0)

    def restore_boundary_particles(self, state: newton.State) -> None:
        """Remove numerical boundary drift after MPM advection."""
        wp.launch(
            restore_boundary_particles,
            dim=self.model.particle_count,
            inputs=[
                self.dynamic,  # dynamic
                self.anchors,  # anchors
                state.particle_q,  # q
                state.particle_qd,  # qd
            ],
            device=self.model.device,
        )

    def reset(self, world_mask: wp.array | None) -> None:
        """Defer selected-world particle reset until after IsaacLab updates robot forward kinematics."""
        if world_mask is None:
            self.pending_reset.fill_(1)
        else:
            wp.launch(
                mark_resets,
                dim=self.model.world_count,
                inputs=[
                    world_mask,  # selected
                    self.pending_reset,  # pending
                ],
                device=self.model.device,
            )
