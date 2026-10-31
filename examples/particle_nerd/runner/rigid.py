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


def rigid_scene_description(manifest, body_pose, body_velocity, dt=0.005):
    """Describe the rigid model used by both interactive play renderers."""
    size = np.asarray(manifest["body_size_m"], dtype=np.float32)
    mass = float(manifest["body_mass_kg"])
    friction = float(manifest.get("body_contact_friction", 0.6))
    inertia = (
        np.asarray(manifest["body_inertia"])
        if "body_inertia" in manifest
        else np.diag(mass / 12 * (np.sum(size**2) - size**2))
    )
    poses = np.asarray(body_pose)
    velocities = np.asarray(body_velocity)
    if poses.ndim == 1:
        poses = poses[None]
    if velocities.ndim == 1:
        velocities = velocities[None]
    count = len(poses)
    return dict(
        body_pose=poses[None],
        body_velocity=velocities[None],
        body_size=np.tile(size, (count, 1)),
        body_mass=np.full(count, mass),
        body_inertia=np.tile(inertia, (count, 1, 1)),
        body_com=np.zeros((count, 3)),
        body_shapes=[manifest.get("geometry", "box")] * count,
        body_friction=np.full(count, friction),
        body_color=np.tile([0.32, 0.38, 0.42], (count, 1)),
        gravity=[0, 0, -9.81],
        dt=dt,
        ground_plane=True,
        ground_friction=0.6,
    )


def build_rigid_model(episode, device="cuda:0"):
    """Build the rigid scene shared by learned physics and reference visualization."""
    e = episode
    if np.any(abs(e["body_com"]) > 1e-8):
        raise ValueError("The current force estimator requires each local body COM at its geometry origin.")
    builder = newton.ModelBuilder(gravity=wp.vec3(*e["gravity"]))
    if e.get("ground_plane", False):
        builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=e.get("ground_friction", 0.6)))
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
        visual = {"color": tuple(e["body_color"][i])} if "body_color" in e else {}
        shape = e["body_shapes"][i]
        if shape == "shovel":
            import json

            from particle_nerd.shovel.asset import ASSET_DIR

            meta = json.loads((ASSET_DIR / "metadata.json").read_text())
            with np.load(ASSET_DIR / "shovel_meshes.npz") as meshes:
                for index, item in enumerate(meta["shapes"]):
                    mesh_cfg = newton.ModelBuilder.ShapeConfig(
                        density=0,
                        mu=0.5,
                        margin=0,
                        gap=0,
                        has_shape_collision=not item["visual"],
                        has_particle_collision=not item["visual"],
                        is_visible="collision_" not in item["name"],
                    )
                    builder.add_shape_mesh(
                        body,
                        mesh=newton.Mesh(meshes[f"vertices_{index}"], meshes[f"faces_{index}"].ravel()),
                        cfg=mesh_cfg,
                        color=tuple(item["color"]),
                        label=item["name"],
                    )
        elif shape == "box":
            builder.add_shape_box(
                body, hx=float(size[0] / 2), hy=float(size[1] / 2), hz=float(size[2] / 2), cfg=cfg, **visual
            )
        elif shape == "sphere":
            builder.add_shape_sphere(body, radius=float(size[0] / 2), cfg=cfg, **visual)
        elif shape == "cylinder":
            builder.add_shape_cylinder(
                body, radius=float(size[0] / 2), half_height=float(size[2] / 2), cfg=cfg, **visual
            )
        else:
            raise ValueError(f"Unsupported rigid geometry: {shape}")
    return builder.finalize(device=device)


class NewtonRigidStepper:
    """Own the rigid solver and apply material-on-body wrenches once per step."""

    def __init__(self, episode, device="cuda:0"):
        e = episode
        self.model = build_rigid_model(episode, device)
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
            # Let Newton size contact and constraint storage from the geometry.
            # The compound shovel can exceed 128 contacts with the ground.
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

    def step(self, material_wrench, external_wrench=None):
        """Advance rigid bodies; wrench order is world force then world COM torque.

        direct_contact_force is inferred from each dynamic body's linear momentum
        balance after subtracting gravity, material force, and external force. It
        includes rigid-tray contacts, which must not be labeled sand reactions.
        """
        wrench = np.asarray(material_wrench, dtype=np.float32)
        if wrench.shape != (self.model.body_count, 6) or not np.isfinite(wrench).all():
            raise ValueError("Material wrenches must be finite with shape (body_count, 6).")
        if external_wrench is not None:
            external = np.asarray(external_wrench, dtype=np.float32)
            if external.shape != wrench.shape or not np.isfinite(external).all():
                raise ValueError("External wrenches must be finite with shape (body_count, 6).")
            wrench = wrench + external
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
