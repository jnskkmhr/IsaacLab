# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate the existing object contact law with recorded pre-MPM body velocity."""

import numpy as np
import warp as wp
from particle_nerd.force_estimation import kernels
from particle_nerd.force_estimation.estimator import ContactEstimator


@wp.kernel
def prepare_recorded_body(
    pose: wp.array(dtype=wp.transform),
    velocity: wp.array(dtype=wp.spatial_vector),
    inertia: wp.mat33d,
    prescribed: bool,
    free: wp.array(dtype=wp.vec3d),
    inverse: wp.array(dtype=wp.mat33d),
):
    free[0] = wp.vec3d(wp.spatial_top(velocity[0]))
    free[1] = wp.vec3d(wp.spatial_bottom(velocity[0]))
    inverse[0] = wp.mat33d(0.0)
    if not prescribed:
        rotation = wp.quat_to_matrix(wp.quatd(wp.transform_get_rotation(pose[0])))
        inverse[0] = wp.inverse(rotation * inertia * wp.transpose(rotation))


class RecordedForceEstimator:
    """Same finite-material-mobility contact law, with explicit body constraints.

    Dynamic bodies use the recorded proxy free velocity after feedback rewind;
    no additional gravity integration is applied. Prescribed bodies have zero
    rigid mobility, the infinite-body-mass limit, while material mobility stays
    finite. This is distinct from fixing the material velocity.

    The isotropic effective particle mass factor is 10, as in the Box example.
    No learned force targets, force filtering, or force-parameter fitting is used.
    """

    def __init__(self, particle_mass, body_mass, body_inertia, size, shape, dt, friction, device="cuda:0"):
        self.prescribed = body_mass == 0
        self.estimator = ContactEstimator(
            particle_mass,
            1.0 if self.prescribed else body_mass,
            np.eye(3) if self.prescribed else body_inertia,
            size,
            dt,
            shape=shape,
            friction=friction,
            effective_mass_scale=10.0,
            stabilization=0.01,
            contact_solver="jacobi",
            device=device,
        )
        if self.prescribed:
            self.estimator.mass = float("inf")
        self.pose = wp.empty(1, dtype=wp.transform, device=device)
        self.velocity = wp.empty(1, dtype=wp.spatial_vector, device=device)

    def estimate(self, positions, next_velocity, collider_state):
        """Return raw material-on-body force and solver diagnostics, without stepping bodies."""
        e = self.estimator
        position = wp.array(positions, dtype=wp.vec3, device=e.device)
        velocity = wp.array(next_velocity, dtype=wp.vec3, device=e.device)
        self.pose.assign(np.asarray(collider_state[:7], dtype=np.float32)[None])
        self.velocity.assign(np.asarray(collider_state[7:], dtype=np.float32)[None])
        e.grid.build(position, e.radius)
        wp.launch(
            kernels.fit_gradients,
            e.n,
            [e.grid.id, position, velocity, e.radius, e.gradients, self.pose, e.sizes, e.shapes, 0, -1.0],
            device=e.device,
        )
        wp.launch(
            prepare_recorded_body,
            1,
            [self.pose, self.velocity, e.inertia, self.prescribed, e.free, e.inverse],
            device=e.device,
        )
        wp.launch(
            kernels.prepare_contacts,
            e.n,
            [
                position,
                velocity,
                e.gradients,
                self.pose,
                e.sizes,
                e.shapes,
                0,
                e.free,
                e.inverse,
                e.mass,
                e.particle_mass,
                e.dt,
                e.scale,
                e.horiz_scale,
                e.up_scale,
                e.tangent_scale,
                e.normal_power,
                e.stabilization,
                e.flags,
                e.frames,
                e.angular,
                e.response,
                e.compliance,
                e.contact_velocity,
                e.gap,
                False,
                *e.sdf,
            ],
            device=e.device,
        )
        wp.utils.array_scan(e.flags, e.offsets, inclusive=True)
        wp.launch(kernels.compact_contacts, e.n, [e.flags, e.offsets, e.ids], device=e.device)
        e.impulses.zero_()
        e.wrench.zero_()
        wp.launch(
            kernels.solve_contacts_parallel,
            128,
            [
                e.offsets,
                e.ids,
                e.frames,
                e.angular,
                e.response,
                e.compliance,
                e.contact_velocity,
                e.gap,
                e.particle_mass,
                e.free,
                e.mass,
                e.scale,
                e.horiz_scale,
                e.up_scale,
                e.tangent_scale,
                e.normal_power,
                e.friction,
                e.dt,
                e.max_iterations,
                wp.float64(e.tolerance),
                e.impulses,
                e.wrench,
                0,
                e.stats,
                False,
                e.jacobi_relaxation,
            ],
            block_dim=128,
            device=e.device,
        )
        return e.wrench.numpy()[0, :3], e.stats.numpy()


class NeRDRecordedForceEstimator:
    """Current G1 proxy force reconstruction, adapted only to one primitive object.

    The caller supplies the same selected particle IDs as compact inference.
    Body inputs contain the rewound free velocity, separately from neural inputs.
    The original RecordedForceEstimator remains available for historical/WorldParticle evaluations.
    """

    def __init__(self, particle_mass, body_mass, body_inertia, size, shape, dt, friction, device="cuda:0"):
        import newton

        from .network import CONTACT_SETTINGS, ProxyEstimator

        builder = newton.ModelBuilder()
        body = builder.add_body(mass=body_mass, inertia=wp.mat33(*np.asarray(body_inertia).ravel()), lock_inertia=True)
        cfg = newton.ModelBuilder.ShapeConfig(density=0, mu=friction, margin=0, gap=0)
        if shape == "box":
            builder.add_shape_box(body, hx=size[0] / 2, hy=size[1] / 2, hz=size[2] / 2, cfg=cfg)
        elif shape == "sphere":
            builder.add_shape_sphere(body, radius=size[0] / 2, cfg=cfg)
        elif shape == "cylinder":
            builder.add_shape_cylinder(body, radius=size[0] / 2, half_height=size[2] / 2, cfg=cfg)
        else:
            raise ValueError(shape)
        n = len(particle_mass)
        builder.add_particles(pos=np.zeros((n, 3)), vel=np.zeros((n, 3)), mass=particle_mass, radius=np.zeros(n))
        self.model = builder.finalize(device=device)
        self.input = self.model.state()
        self.output = self.model.state()
        self.dt = dt
        self.estimator = ProxyEstimator(
            self.model,
            [body],
            particles_per_env=n,
            colliders_per_world=1,
            primitive_colliders=True,
            batch_contacts=False,
            contact_solver="rft_body_velocity",
            iterations=CONTACT_SETTINGS["contact_iterations"],
            effective_mass_scale=CONTACT_SETTINGS["effective_mass_scale"],
            tangent_mass_scale=CONTACT_SETTINGS["tangent_mass_scale"],
            horiz_mass_scale=CONTACT_SETTINGS["horizontal_mass_scale"],
            up_mass_scale=CONTACT_SETTINGS["upward_mass_scale"],
            normal_power=CONTACT_SETTINGS["normal_power"],
            relaxation=CONTACT_SETTINGS["contact_relaxation"],
            gradient_radius=CONTACT_SETTINGS["gradient_radius"],
            gradient_contact_band=CONTACT_SETTINGS["gradient_contact_band"],
            stabilization=CONTACT_SETTINGS["contact_stabilization"],
            contact_offset=CONTACT_SETTINGS["contact_offset"],
        )
        self.estimator.tolerance = 1.0e-9
        self.estimator.stats = self.estimator._all_stats

    def estimate(self, positions, next_velocity, collider_state, *, active_idx):
        self.input.particle_q.assign(positions)
        self.input.body_q.assign(np.asarray(collider_state[:7], dtype=np.float32)[None])
        self.input.body_qd.assign(np.asarray(collider_state[7:], dtype=np.float32)[None])
        self.output.particle_qd.assign(next_velocity)
        self.estimator.capture(self.input, self.dt)
        self.estimator.evaluate(self.output, active_idx=[active_idx])
        return self.estimator.wrench.numpy()[0, :3], self.estimator.stats.numpy()
