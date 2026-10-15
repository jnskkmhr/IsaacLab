# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton-specific state and impulse access, isolated from the manager terms.

Private coupler entries provide the MPM submodel and proxy feedback. The
collector records both generated MPM impulses and lagged impulses applied to MJWarp.
"""

import re

import mujoco_warp as mjw
import torch
import warp as wp
from isaaclab_newton.physics import NewtonManager
from scene_cfg import BODY_SIZES


@wp.kernel
def sum_rigid_contact_impulses(
    count: wp.array[int],
    worlds: wp.array[int],
    geoms: wp.array[wp.vec2i],
    positions: wp.array[wp.vec3],
    geom_to_shape: wp.array2d[int],
    shape_to_body: wp.array[int],
    poses: wp.array[wp.transform],
    contact_wrench: wp.array[wp.spatial_vector],
    dt: float,
    output: wp.array[wp.spatial_vector],
):
    """Sum rigid contact impulses about each body's center of mass (zero local offset)."""
    contact = wp.tid()
    if contact < count[0]:
        world = worlds[contact]
        pair = geoms[contact]
        for side in range(2):
            shape = geom_to_shape[world, pair[side]]
            if shape >= 0:
                body = shape_to_body[shape]
                if body >= 0:
                    sign = float(2 * side - 1)
                    impulse = sign * dt * wp.spatial_top(contact_wrench[contact])
                    torque = sign * dt * wp.spatial_bottom(contact_wrench[contact])
                    arm = positions[contact] - wp.transform_get_translation(poses[body])
                    wp.atomic_add(output, body, wp.spatial_vector(impulse, torque + wp.cross(arm, impulse)))


class NewtonData:
    """Map Newton indices to persistent per-environment particle and body IDs."""

    def __init__(self, env):
        self.env = env
        self.model = NewtonManager.get_model()
        coupler = NewtonManager._solver
        self.entry = coupler._entries["sand"]
        self.solver = self.entry.solver
        rigid_entry = coupler._entries["rigid"]
        self.rigid_solver = rigid_entry.solver
        shape_bodies = self.rigid_solver.model.shape_body.numpy()
        body_map = rigid_entry.body_local_to_global.numpy()
        self.rigid_shape_to_body = wp.array(
            [int(body_map[b]) if b >= 0 else -1 for b in shape_bodies], dtype=wp.int32, device=env.device
        )
        capacity = self.rigid_solver.mjw_data.naconmax
        self.contact_indices = wp.array(list(range(capacity)), dtype=wp.int32, device=env.device)
        self.contact_wrench = wp.zeros(capacity, dtype=wp.spatial_vector, device=env.device)
        self.rigid_impulses = wp.zeros(self.model.body_count, dtype=wp.spatial_vector, device=env.device)
        self.rigid_contact_poses = wp.clone(NewtonManager.get_state_0().body_q)
        self.feedback = next(
            mapping.coupling_forces
            for mapping in coupler._proxy_mappings
            if mapping.src_name == "rigid" and mapping.dst_name == "sand"
        )
        self.local_to_global = wp.to_torch(self.entry.body_local_to_global).long()
        body_indices = [[-1] * len(BODY_SIZES) for _ in range(env.num_envs)]
        for index, label in enumerate(self.model.body_label):
            match = re.fullmatch(r"/World/envs/env_(\d+)/Body(\d+)", label)
            if match:
                world, object_id = map(int, match.groups())
                body_indices[world][object_id] = index
        self.body_indices = torch.tensor(body_indices, device=env.device)
        if (self.body_indices < 0).any():
            raise RuntimeError(f"Missing body IDs in Newton model: {body_indices}")
        # Material accessor and model mass order must agree; this scene has one particle object per world.
        world = wp.to_torch(self.model.particle_world)
        self.particle_indices = torch.stack(
            [torch.nonzero(world == i, as_tuple=False).flatten() for i in range(env.num_envs)]
        )
        if self.particle_indices.numel() != self.model.particle_count:
            raise RuntimeError("Expected all material particles to belong to a recorded environment.")

    def state(self):
        """Positions use environment origins; all vector components use world-aligned axes."""
        state = NewtonManager.get_state_0()
        pose = wp.to_torch(state.body_q)[self.body_indices].clone()
        pose[..., :3] -= self.env.scene.env_origins[:, None, :]
        return {
            "particle_position": wp.to_torch(state.particle_q)[self.particle_indices]
            - self.env.scene.env_origins[:, None, :],
            "particle_velocity": wp.to_torch(state.particle_qd)[self.particle_indices],
            "body_pose": pose,
            "body_velocity": wp.to_torch(state.body_qd)[self.body_indices],
        }

    def body_impulse(self):
        """Impulse exerted by material on each body [N s, N m s], about box centers."""
        state = self.entry.state_0
        impulses, positions, collider_ids = self.solver.collect_collider_impulses(state)
        impulse = wp.to_torch(impulses)
        position = wp.to_torch(positions)
        collider_id = wp.to_torch(collider_ids).long()
        valid = collider_id >= 0
        ids = wp.to_torch(self.solver.collider_body_index)[collider_id[valid]].long()
        impulse, position = impulse[valid], position[valid]
        valid = ids >= 0
        ids, impulse, position = ids[valid], impulse[valid], position[valid]
        center = wp.to_torch(state.body_q)[ids, :3]  # All boxes have COM at their link origin.
        wrench = torch.cat((impulse, torch.cross(position - center, impulse, dim=-1)), dim=-1)
        output = torch.zeros((self.model.body_count, 6), device=self.env.device)
        output.index_add_(0, self.local_to_global[ids], wrench)
        return output[self.body_indices]

    def cache_rigid_contact_poses(self):
        """Store the COM frame at which the single rigid substep applies contact forces."""
        wp.copy(self.rigid_contact_poses, NewtonManager.get_state_0().body_q)

    def feedback_impulse(self):
        """Coupler feedback [N s, N m s]; sample BEFORE stepping for the applied impulse."""
        return wp.to_torch(self.feedback)[self.body_indices].clone() * self.env.physics_dt

    def rigid_contact_impulse(self):
        """Direct rigid-body contacts; exact force integration requires one rigid substep."""
        solver = self.rigid_solver
        data = solver.mjw_data
        mjw.contact_force(solver.mjw_model, data, self.contact_indices, True, self.contact_wrench)
        self.rigid_impulses.zero_()
        wp.launch(
            sum_rigid_contact_impulses,
            dim=data.naconmax,
            inputs=[
                data.nacon,
                data.contact.worldid,
                data.contact.geom,
                data.contact.pos,
                solver.mjc_geom_to_newton_shape,
                self.rigid_shape_to_body,
                self.rigid_contact_poses,
                self.contact_wrench,
                self.env.physics_dt,
            ],
            outputs=[self.rigid_impulses],
            device=self.env.device,
        )
        return wp.to_torch(self.rigid_impulses)[self.body_indices].clone()
