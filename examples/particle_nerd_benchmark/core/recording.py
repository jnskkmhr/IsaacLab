# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Streaming, step-aligned native MPM states and collider impulses."""

import h5py
import numpy as np
import torch
import warp as wp
from isaaclab_newton.physics import NewtonManager


class Recorder:
    """Save post-step states and MPM impulse/dt about each rigid body's COM.

    Every force is the MPM material-on-body force in world axes. Direct rigid
    floor/wall contacts are not included. The initial state has no force sample.
    """

    def __init__(self, env, path, steps, *, check_tray_contacts=False):
        self.env = env
        self.model = NewtonManager.get_model()
        self.entry = NewtonManager._solver._entries["sand"]
        self.solver = self.entry.solver
        self.mapping = wp.to_torch(self.entry.body_local_to_global).long()
        self.file = h5py.File(path, "w")
        self.file.attrs["quaternion_order"] = "xyzw"
        self.file.attrs["vector_frame"] = "world; origin at initial tray centre"
        self.file.attrs["force_definition"] = "MPM material-on-body impulse/dt; rigid contacts excluded"
        self.file.create_dataset("body_names", data=np.array(self.model.body_label, dtype=h5py.string_dtype()))
        self.file.create_dataset("particle_mass", data=self.model.particle_mass.numpy())
        self.file.create_dataset("particle_radius", data=self.model.particle_radius.numpy())
        self.file.create_dataset("body_mass", data=self.model.body_mass.numpy())
        self.file.create_dataset("body_com", data=self.model.body_com.numpy())
        self.tray_contact_shapes = None
        if check_tray_contacts:
            self.tray_contact_shapes = np.array(
                ["/RigidWall" in label or "/RigidFloor" in label for label in self.model.shape_label]
            )
            self.file.create_dataset("rigid_tray_contact_count", (steps + 1,), dtype="i4")
            self.file["rigid_tray_contact_count"].attrs["definition"] = (
                "Dynamic-body versus rigid tray contacts with nonpositive gap, including contact margins"
            )
        self.datasets = {}
        self.file.create_dataset("applied_action", (steps, env.action_manager.total_action_dim), dtype="f4")
        for name in ("shape_body", "shape_type", "shape_transform", "shape_scale"):
            self.file.create_dataset(name, data=getattr(self.model, name).numpy())
        for name, shape in (
            ("particle_position", (self.model.particle_count, 3)),
            ("particle_velocity", (self.model.particle_count, 3)),
            ("body_pose", (self.model.body_count, 7)),
            ("body_velocity", (self.model.body_count, 6)),
            ("material_force", (self.model.body_count, 3)),
            ("material_torque", (self.model.body_count, 3)),
        ):
            count = steps if name.startswith("material_") else steps + 1
            self.datasets[name] = self.file.create_dataset(
                name, (count, *shape), dtype="f4", chunks=(1, *shape), compression="lzf"
            )
        self.file.create_dataset("state_time", data=np.arange(steps + 1) * env.physics_dt)
        self.file.create_dataset("force_time", data=np.arange(1, steps + 1) * env.physics_dt)
        if "robot" in env.scene.articulations:
            joints = env.scene["robot"].num_joints
            self.file.create_dataset(
                "joint_names", data=np.array(env.scene["robot"].joint_names, dtype=h5py.string_dtype())
            )
            self.datasets["joint_position"] = self.file.create_dataset(
                "joint_position", (steps + 1, joints), dtype="f4"
            )
            self.datasets["joint_velocity"] = self.file.create_dataset(
                "joint_velocity", (steps + 1, joints), dtype="f4"
            )
        self.min_z = []

    def record(self, index):
        state = NewtonManager.get_state_0()
        if self.tray_contact_shapes is not None:
            contacts = NewtonManager.get_contacts()
            count = int(contacts.rigid_contact_count.numpy()[0])
            a = contacts.rigid_contact_shape0.numpy()[:count]
            b = contacts.rigid_contact_shape1.numpy()[:count]
            valid = (a >= 0) & (b >= 0)
            a, b = a[valid], b[valid]
            bodies = self.model.shape_body.numpy()
            touching = (self.tray_contact_shapes[a] & (bodies[b] >= 0)) | (
                self.tray_contact_shapes[b] & (bodies[a] >= 0)
            )
            points = [
                contacts.rigid_contact_point0.numpy()[:count][valid],
                contacts.rigid_contact_point1.numpy()[:count][valid],
            ]
            poses = state.body_q.numpy()
            for point, body in zip(points, (bodies[a], bodies[b]), strict=True):
                moving = body >= 0
                pose = poses[body[moving]]
                local = point[moving]
                point[moving] = (
                    pose[:, :3]
                    + local
                    + 2 * np.cross(pose[:, 3:6], np.cross(pose[:, 3:6], local) + pose[:, 6:7] * local)
                )
            gap = np.sum((points[1] - points[0]) * contacts.rigid_contact_normal.numpy()[:count][valid], axis=1)
            gap -= contacts.rigid_contact_margin0.numpy()[:count][valid]
            gap -= contacts.rigid_contact_margin1.numpy()[:count][valid]
            touching &= gap <= 0.0
            self.file["rigid_tray_contact_count"][index] = np.count_nonzero(touching)
        for name, array in (
            ("particle_position", state.particle_q),
            ("particle_velocity", state.particle_qd),
            ("body_pose", state.body_q),
            ("body_velocity", state.body_qd),
        ):
            values = array.numpy()
            if not np.isfinite(values).all():
                raise RuntimeError(f"Nonfinite {name} at sample {index}")
            self.datasets[name][index] = values
        if "joint_position" in self.datasets:
            self.datasets["joint_position"][index] = self.env.scene["robot"].data.joint_pos.torch[0].cpu().numpy()
            self.datasets["joint_velocity"][index] = self.env.scene["robot"].data.joint_vel.torch[0].cpu().numpy()
        if index:
            self.file["applied_action"][index - 1] = self.env.action_manager.action[0].cpu().numpy()
            impulses, positions, colliders = self.solver.collect_collider_impulses(self.entry.state_0)
            ids = wp.to_torch(colliders).long()
            keep = ids >= 0
            local = wp.to_torch(self.solver.collider_body_index)[ids[keep]].long()
            impulse = wp.to_torch(impulses)[keep]
            position = wp.to_torch(positions)[keep]
            keep = local >= 0
            local, impulse, position = local[keep], impulse[keep], position[keep]
            q = wp.to_torch(self.entry.state_0.body_q)[local]
            # Body COM offsets matter for articulated robot links.
            from isaaclab.utils.math import quat_apply

            centre = q[:, :3] + quat_apply(q[:, 3:], wp.to_torch(self.solver.model.body_com)[local])
            wrench = torch.cat((impulse, torch.cross(position - centre, impulse, dim=-1)), dim=-1) / self.env.physics_dt
            output = torch.zeros((self.model.body_count, 6), device=self.env.device)
            output.index_add_(0, self.mapping[local], wrench)
            values = output.cpu().numpy()
            if not np.isfinite(values).all():
                raise RuntimeError(f"Nonfinite MPM reaction at sample {index}")
            self.datasets["material_force"][index - 1] = values[:, :3]
            self.datasets["material_torque"][index - 1] = values[:, 3:]

    def close(self):
        self.file.close()
