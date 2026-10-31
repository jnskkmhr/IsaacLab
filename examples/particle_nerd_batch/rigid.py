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
    return dict(
        body_pose=np.asarray(body_pose)[None],
        body_velocity=np.asarray(body_velocity)[None],
        body_size=np.tile(size, (len(body_pose), 1)),
        body_mass=np.full(len(body_pose), mass),
        body_inertia=np.tile(inertia, (len(body_pose), 1, 1)),
        body_com=np.zeros((len(body_pose), 3)),
        body_shapes=[manifest.get("geometry", "box")] * len(body_pose),
        body_friction=np.full(len(body_pose), friction),
        body_color=np.tile([0.32, 0.38, 0.42], (len(body_pose), 1)),
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
        builder.begin_world()
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
            from pathlib import Path

            ASSET_DIR = Path(__file__).parent / "assets"

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
        builder.end_world()
    return builder.finalize(device=device)


class RigidBatch:
    """One MJWarp model and collision pipeline for all independent worlds."""

    def __init__(self, scene, device):
        self.model = build_rigid_model(scene, device)
        self.states = (self.model.state(), self.model.state())
        self.control = self.model.control()
        for state in self.states:
            state.body_qd.assign(scene["body_velocity"][0])
            newton.eval_ik(self.model, state, state.joint_q, state.joint_qd)
        self.solver = SolverMuJoCo(
            self.model, use_mujoco_contacts=False, integrator="euler", iterations=100, ls_iterations=50
        )
        self.pipeline = newton.CollisionPipeline(self.model)
        self.contacts = self.pipeline.contacts()
        if self.solver.mjw_data.nworld != len(scene["body_mass"]):
            raise RuntimeError("MJWarp did not construct the requested independent worlds")

    def step(self, source, target, dt):
        self.pipeline.collide(source, self.contacts)
        self.solver.step(source, target, self.control, self.contacts, dt)
