# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Moving-patch reset/contact handling plus robot-object forces from the rigid solver entry."""

import re

import mujoco_warp as mjw
import numpy as np
import torch
import warp as wp
from isaaclab_newton.physics import NewtonManager

from isaaclab_tasks.contrib.g1_29dof_mpm.g1_mpm_env import sum_rigid_foot_forces
from isaaclab_tasks.contrib.g1_29dof_particle_nerd_moving_patch.particle_nerd_env import G1ParticleNeRDEnv


class ObjectContactForces:
    """Read object-filtered rigid forces for fixed or cached sand-bed environments."""

    def _setup_contact_state(self):
        super()._setup_contact_state()
        body_ids = self._resolve_newton_body_ids(self._robot.body_names).cpu().numpy().flatten()
        body_to_robot = {int(body): i for i, body in enumerate(body_ids)}
        entry = NewtonManager._solver._entries["robot"]
        mapping = entry.body_local_to_global.numpy()
        shape_to_robot = np.full(entry.solver.model.shape_count, -1, dtype=np.int32)
        for shape, body in enumerate(entry.solver.model.shape_body.numpy()):
            if body >= 0:
                shape_to_robot[shape] = body_to_robot.get(int(mapping[body]), -1)
        self._shape_to_robot = wp.array(shape_to_robot, device=self.device)
        self._shape_is_object = wp.array(
            [
                bool(re.match(f"^(?:{self.scene[self.cfg.push_object_name].cfg.prim_path})(/|$)", label))
                for label in entry.solver.model.shape_label
            ],
            dtype=wp.bool,
            device=self.device,
        )
        if not self._shape_is_object.numpy().any():
            raise RuntimeError("No object collision shapes found in the rigid solver entry.")
        self._object_contact_force = wp.zeros(len(body_ids), dtype=wp.vec3, device=self.device)
        self._object_contact_step = -1
        self._object_rigid_solver = entry.solver
        capacity = entry.solver.mjw_data.contact.geom.shape[0]
        self._object_contact_indices = wp.array(np.arange(capacity, dtype=np.int32), device=self.device)
        self._object_contact_wrench = wp.zeros(capacity, dtype=wp.spatial_vector, device=self.device)
        self._object_contact_valid = torch.zeros((self.num_envs, 1, 1), dtype=torch.bool, device=self.device)

    @property
    def object_contact_force(self) -> torch.Tensor:
        """World-frame object force on each robot body [N], shaped (worlds, bodies, 3)."""
        if self._object_contact_step != self.common_step_counter:
            self._object_contact_force.zero_()
            solver = self._object_rigid_solver
            mjw.contact_force(
                solver.mjw_model, solver.mjw_data, self._object_contact_indices, True, self._object_contact_wrench
            )
            wp.launch(
                sum_rigid_foot_forces,
                dim=self._object_contact_indices.shape[0],
                inputs=[
                    solver.mjw_data.nacon,
                    solver.mjw_data.contact.worldid,
                    solver.mjw_data.contact.geom,
                    solver.mjc_geom_to_newton_shape,
                    self._shape_to_robot,
                    self._shape_is_object,
                    self._object_contact_wrench,
                ],
                outputs=[self._object_contact_force],
                device=self.device,
            )
            self._object_contact_step = self.common_step_counter
        forces = wp.to_torch(self._object_contact_force).view(self.num_envs, self._robot.num_bodies, 3)
        return torch.where(self._object_contact_valid, forces, 0.0)

    def step(self, action):
        self._object_contact_valid.fill_(True)
        return super().step(action)

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        self._object_contact_valid[env_ids] = False
        self._object_contact_step = -1


class ObjectPushEnv(ObjectContactForces, G1ParticleNeRDEnv):
    """Keep all bed particles active and expose object-filtered rigid contact forces."""
