# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton MJWarp rigid-body inference alongside externally predicted particles.

Only rigid geometry is inserted into Newton. Material particles remain in the
learned model, so this adapter never constructs or steps an MPM solver.
"""

import newton
import numpy as np
import warp as wp
from newton.solvers import SolverMuJoCo


class NewtonRigidStepper:
    """Own the rigid solver and apply material-on-body wrenches once per step."""

    def __init__(self, episode, device="cuda:0"):
        e = episode
        if np.any(abs(e["body_com"]) > 1e-8):
            raise ValueError("The current force estimator requires each local body COM at its geometry origin.")
        builder = newton.ModelBuilder(gravity=wp.vec3(*e["gravity"]))
        for i, (pose, size) in enumerate(zip(e["body_pose"][0], e["body_size"], strict=True)):
            mass = float(e["body_mass"][i])
            body = builder.add_body(
                xform=wp.transform(wp.vec3(*pose[:3]), wp.quat(*pose[3:])),
                mass=mass,
                inertia=wp.mat33(*e["body_inertia"][i].flatten()),
                is_kinematic=mass == 0,
                lock_inertia=True,
                label=f"body_{i}",
            )
            cfg = newton.ModelBuilder.ShapeConfig(density=0, mu=float(e["body_friction"][i]), margin=0, gap=0)
            shape = e["body_shapes"][i]
            if shape == "box":
                builder.add_shape_box(
                    body, hx=float(size[0] / 2), hy=float(size[1] / 2), hz=float(size[2] / 2), cfg=cfg
                )
            elif shape == "sphere":
                builder.add_shape_sphere(body, radius=float(size[0] / 2), cfg=cfg)
            elif shape == "cylinder":
                builder.add_shape_cylinder(body, radius=float(size[0] / 2), half_height=float(size[2] / 2), cfg=cfg)
            else:
                raise ValueError(f"Unsupported rigid geometry: {shape}")
        self.model = builder.finalize(device=device)
        self.state = self.model.state()
        self.next_state = self.model.state()
        self.control = self.model.control()
        self.state.body_qd.assign(e["body_velocity"][0])
        newton.eval_ik(self.model, self.state, self.state.joint_q, self.state.joint_qd)
        # Match the Isaac Lab collector; Newton's standalone default integrator
        # is implicitfast, whereas the collector explicitly uses Euler.
        self.solver = SolverMuJoCo(
            self.model,
            use_mujoco_contacts=False,
            njmax=128,
            nconmax=128,
            integrator="euler",
            iterations=100,
            ls_iterations=50,
        )
        self.pipeline = newton.CollisionPipeline(self.model)
        self.contacts = self.pipeline.contacts()
        self.dt = e["dt"]
        self.gravity = np.asarray(e["gravity"], dtype=float)
        self.mass = np.asarray(e["body_mass"], dtype=float)
        self.direct_contact_force = np.zeros((len(self.mass), 3))

    def step(self, material_wrench):
        """Advance rigid bodies; wrench order is world force then world COM torque.

        direct_contact_force is inferred from each dynamic body's linear momentum
        balance after subtracting gravity and the supplied material force. It
        includes rigid-tray contacts, which must not be labeled sand reactions.
        """
        wrench = np.asarray(material_wrench, dtype=np.float32)
        if wrench.shape != (self.model.body_count, 6) or not np.isfinite(wrench).all():
            raise ValueError("Material wrenches must be finite with shape (body_count, 6).")
        before = self.state.body_qd.numpy()
        self.state.clear_forces()
        self.state.body_f.assign(wrench)
        self.pipeline.collide(self.state, self.contacts)
        self.solver.step(self.state, self.next_state, self.control, self.contacts, self.dt)
        self.state, self.next_state = self.next_state, self.state
        pose, velocity = self.state.body_q.numpy(), self.state.body_qd.numpy()
        self.direct_contact_force = (
            self.mass[:, None] * ((velocity[:, :3] - before[:, :3]) / self.dt - self.gravity) - wrench[:, :3]
        )
        self.direct_contact_force[self.mass == 0] = 0
        return pose, velocity
