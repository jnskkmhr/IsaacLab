# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Mixed neural sand and rigid ground with one policy/contact interface."""

import mujoco_warp as mjw
import numpy as np
import torch
import warp as wp
from isaaclab_newton.physics import NewtonManager

from isaaclab_tasks.contrib.g1_29dof_mpm_moving_patch.mixed_env import sum_rigid_foot_forces

from .particle_nerd_env import G1ParticleNeRDEnv


class G1MixedTerrainEnv(G1ParticleNeRDEnv):
    """Integrate all robots with MJWarp and only sand worlds with Particle NeRD."""

    def _set_contact_margin(self, payload=None):
        """Assign the neural and rigid groups their configured foot margins."""
        self._set_foot_contact_margins(
            [
                self.cfg.particle_contact_margin if soft else self.cfg.rigid_contact_margin
                for soft in self.scene.terrain.mpm_env_mask.cpu().tolist()
            ]
        )

    def _particle_instance_ids(self, env_ids):
        """Map scene reset IDs to the compact particle asset, excluding rigid worlds."""
        return self._sand.select_mpm_env_ids(env_ids)

    def reset_particle_state(self, env_ids):
        """Reset selected worlds and invalidate their previous rigid contact forces."""
        super().reset_particle_state(env_ids)
        self._rigid_contact_valid[env_ids] = False

    def _setup_contact_state(self) -> None:
        entry = NewtonManager._solver._entries["robot"]
        self._rigid_solver = entry.solver
        foot_names = self.scene["robot"].find_bodies(self.cfg.foot_body_expr, preserve_order=True)[1]
        foot_bodies = self._resolve_newton_body_ids(foot_names).cpu().numpy().flatten()
        body_to_foot = {int(body): index for index, body in enumerate(foot_bodies)}
        local_to_global = entry.body_local_to_global.numpy()
        shape_to_foot = np.full(entry.solver.model.shape_count, -1, dtype=np.int32)
        for shape, body in enumerate(entry.solver.model.shape_body.numpy()):
            if body >= 0:
                shape_to_foot[shape] = body_to_foot.get(int(local_to_global[body]), -1)
        self._shape_to_foot = wp.array(shape_to_foot, dtype=int, device=self.device)
        self._shape_is_ground = wp.array(
            [label.startswith("/World/ground/") for label in entry.solver.model.shape_label],
            dtype=wp.bool,
            device=self.device,
        )
        capacity = self._rigid_solver.mjw_data.contact.geom.shape[0]
        self._contact_indices = wp.array(np.arange(capacity), dtype=int, device=self.device)
        self._rigid_contact_wrench = wp.zeros(capacity, dtype=wp.spatial_vector, device=self.device)
        self._rigid_foot_force = wp.zeros(len(foot_bodies), dtype=wp.vec3, device=self.device)
        self._rigid_contact_valid = torch.zeros((self.num_envs, 1, 1), dtype=torch.bool, device=self.device)
        self._rigid_contact_valid_wp = wp.from_torch(self._rigid_contact_valid)
        super()._setup_contact_state()

    def _restore_boundary_particles(self) -> None:
        super()._restore_boundary_particles()
        self._rigid_contact_valid_wp.fill_(True)

    def _refresh_contact_forces(self) -> None:
        super()._refresh_contact_forces()
        # Coupled solvers do not support ContactSensor yet; read the MJWarp entry.
        solver = self._rigid_solver
        mjw.contact_force(solver.mjw_model, solver.mjw_data, self._contact_indices, True, self._rigid_contact_wrench)
        self._rigid_foot_force.zero_()
        wp.launch(
            sum_rigid_foot_forces,
            dim=self._contact_indices.shape[0],
            inputs=[
                solver.mjw_data.nacon,
                solver.mjw_data.contact.worldid,
                solver.mjw_data.contact.geom,
                solver.mjc_geom_to_newton_shape,
                self._shape_to_foot,
                self._shape_is_ground,
                self._rigid_contact_wrench,
            ],
            outputs=[self._rigid_foot_force],
            device=self.device,
        )
        # Include MJWarp contacts also when a sand foot reaches the supporting floor.
        rigid_force = wp.to_torch(self._rigid_foot_force).reshape(self.num_envs, self.foot_count, 3)
        self._foot_contact_force += torch.where(self._rigid_contact_valid, rigid_force, 0.0)
        torch.nan_to_num_(self._foot_contact_force, nan=0.0, posinf=0.0, neginf=0.0)
        self._foot_contact = self._foot_contact_force.norm(dim=-1) > self.cfg.foot_contact_force_threshold
