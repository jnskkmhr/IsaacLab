# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Moving-patch environment exposing physics-rate coupled forces to Agile rewards."""

from __future__ import annotations

from collections.abc import Sequence

import mujoco_warp as mjw
import numpy as np
import torch
import warp as wp
from isaaclab_newton.physics import NewtonManager

from isaaclab_tasks.contrib.g1_29dof_mpm_moving_patch.mixed_env import sum_rigid_foot_forces
from isaaclab_tasks.contrib.g1_29dof_mpm_moving_patch.mpm_env import G1MovingPatchEnv


@wp.kernel
def _record_body_forces(
    body_ids: wp.array2d[wp.int64],
    rigid_force: wp.array[wp.vec3],
    coupling_force: wp.array[wp.spatial_vector],
    history: wp.array3d[wp.vec3],
):
    env, body = wp.tid()
    for age in range(4, 0, -1):
        history[env, age, body] = history[env, age - 1, body]
    force = rigid_force[env * body_ids.shape[1] + body] + wp.spatial_top(coupling_force[body_ids[env, body]])
    for axis in range(3):
        if not wp.isfinite(force[axis]):
            force[axis] = 0.0
    history[env, 0, body] = force


class G1MovingPatchAgileEnv(G1MovingPatchEnv):
    """Reuse particle recycling/reset and collect rigid plus MPM body forces at 200 Hz."""

    def _setup_contact_state(self) -> None:
        super()._setup_contact_state()
        entry = NewtonManager._solver._entries["robot"]
        self._rigid_solver = entry.solver
        body_ids = self._resolve_newton_body_ids(self._robot.body_names)
        self._contact_body_ids = wp.from_torch(body_ids)
        body_to_index = {int(body): index for index, body in enumerate(body_ids.cpu().numpy().flatten())}
        local_to_global = entry.body_local_to_global.numpy()
        shape_to_body = np.full(entry.solver.model.shape_count, -1, dtype=np.int32)
        for shape, body in enumerate(entry.solver.model.shape_body.numpy()):
            if body >= 0:
                shape_to_body[shape] = body_to_index.get(int(local_to_global[body]), -1)
        self._shape_to_body = wp.array(shape_to_body, dtype=int, device=self.device)
        # Agile's rigid contact sensors include every partner, including robot self-contact.
        self._contact_partner = wp.ones(entry.solver.model.shape_count, dtype=wp.bool, device=self.device)
        capacity = self._rigid_solver.mjw_data.contact.geom.shape[0]
        self._contact_indices = wp.array(np.arange(capacity), dtype=int, device=self.device)
        self._contact_wrench = wp.zeros(capacity, dtype=wp.spatial_vector, device=self.device)
        self._rigid_body_force = wp.zeros(body_ids.numel(), dtype=wp.vec3, device=self.device)
        self._body_force_history = wp.zeros(
            (self.num_envs, 5, self._robot.num_bodies), dtype=wp.vec3, device=self.device
        )

    @property
    def body_contact_force_history(self) -> torch.Tensor:
        """World-frame forces [N], newest first, in articulation body order: (N, 5, B, 3)."""
        return wp.to_torch(self._body_force_history)

    def _restore_boundary_particles(self) -> None:
        super()._restore_boundary_particles()
        solver = self._rigid_solver
        mjw.contact_force(solver.mjw_model, solver.mjw_data, self._contact_indices, True, self._contact_wrench)
        self._rigid_body_force.zero_()
        wp.launch(
            sum_rigid_foot_forces,
            dim=self._contact_indices.shape[0],
            inputs=[
                solver.mjw_data.nacon,
                solver.mjw_data.contact.worldid,
                solver.mjw_data.contact.geom,
                solver.mjc_geom_to_newton_shape,
                self._shape_to_body,
                self._contact_partner,
                self._contact_wrench,
            ],
            outputs=[self._rigid_body_force],
            device=self.device,
        )
        wp.launch(
            _record_body_forces,
            dim=(self.num_envs, self._robot.num_bodies),
            inputs=[self._contact_body_ids, self._rigid_body_force, self._coupling_forces],
            outputs=[self._body_force_history],
            device=self.device,
        )

    def reset_mpm_state(self, env_ids: Sequence[int] | torch.Tensor | slice) -> None:
        super().reset_mpm_state(env_ids)
        self.body_contact_force_history[env_ids] = 0.0
