# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Analytical GPU contact estimation, independent of the particle predictor."""

import numpy as np
import warp as wp

from particle_nerd.force_estimation import kernels


class ContactEstimator:
    """Estimate material-on-body world force and COM torque from next particle velocity.

    Geometry and mass are fixed at construction. All step arguments are Warp GPU
    arrays. Positions are at t; particle velocities are the prediction at t+dt.
    The external wrench contains known forces, excluding material contact and gravity.
    The effective particle mass scale and stabilization are approximation parameters,
    not identified material laws. No MPM state or future rigid state is used.
    """

    def __init__(
        self,
        particle_mass,
        body_mass,
        body_inertia,
        body_size,
        dt,
        shape="box",
        friction=0.48,
        gravity=(0, 0, -9.81),
        effective_mass_scale=10.0,
        stabilization=0.01,
        gradient_radius=0.065,
        max_iterations=2000,
        tolerance=1e-9,
        device="cuda:0",
    ):
        if dt <= 0 or body_mass <= 0 or max_iterations < 1:
            raise ValueError("dt, body mass, and iteration limit must be positive")
        if effective_mass_scale <= 0 or gradient_radius <= 0 or tolerance <= 0 or friction < 0:
            raise ValueError("Invalid force-estimator parameters")
        if len(particle_mass) == 0 or np.any(np.asarray(particle_mass) <= 0):
            raise ValueError("Particle masses must be positive")
        self.n, self.device = len(particle_mass), device
        self.mass, self.dt = float(body_mass), float(dt)
        self.inertia = wp.mat33d(*np.asarray(body_inertia).flatten())
        self.gravity = wp.vec3(*gravity)
        self.scale, self.stabilization = float(effective_mass_scale), float(stabilization)
        self.radius, self.friction = float(gradient_radius), float(friction)
        self.max_iterations, self.tolerance = max_iterations, tolerance
        self.particle_mass = wp.array(particle_mass, dtype=float, device=device)
        self.sizes = wp.array([body_size], dtype=wp.vec3, device=device)
        self.shapes = wp.array([{"box": 0, "sphere": 1, "cylinder": 2}[shape]], dtype=int, device=device)

        def zeros(n, dtype):
            return wp.zeros(n, dtype=dtype, device=device)

        self.grid = wp.HashGrid(64, 64, 64, device=device)
        self.grid.reserve(self.n)
        self.gradients = zeros(self.n, wp.mat33d)
        self.free, self.inverse = zeros(2, wp.vec3d), zeros(1, wp.mat33d)
        self.flags, self.offsets, self.ids = [zeros(self.n, int) for _ in range(3)]
        self.frames, self.angular, self.response, self.compliance = [zeros(self.n, wp.mat33d) for _ in range(4)]
        self.contact_velocity, self.impulses = [zeros(self.n, wp.vec3d) for _ in range(2)]
        self.gap, self.stats = zeros(self.n, wp.float64), zeros(3, wp.float64)
        self.wrench = zeros(1, wp.spatial_vector)

    def estimate(self, position, next_velocity, body_pose, body_velocity, external_wrench):
        """Enqueue the solve on the active Warp stream; return a device wrench array.

        Shapes: particle arrays [N] vec3, body arrays [1] transform/spatial_vector.
        The caller selects the stream (use the Torch stream for Torch/Warp sharing).
        This method does not update particles or rigid bodies and never reads GPU data.
        """
        self.grid.build(position, self.radius)
        wp.launch(
            kernels.fit_gradients,
            self.n,
            [
                self.grid.id,
                position,
                next_velocity,
                self.radius,
                self.gradients,
                body_pose,
                self.sizes,
                self.shapes,
                0,
                -1.0,
            ],
            device=self.device,
        )
        wp.launch(
            kernels.prepare_body,
            1,
            [
                body_pose,
                body_velocity,
                external_wrench,
                0,
                self.mass,
                self.inertia,
                self.gravity,
                self.dt,
                self.free,
                self.inverse,
            ],
            device=self.device,
        )
        wp.launch(
            kernels.prepare_contacts,
            self.n,
            [
                position,
                next_velocity,
                self.gradients,
                body_pose,
                self.sizes,
                self.shapes,
                0,
                self.free,
                self.inverse,
                self.mass,
                self.particle_mass,
                self.dt,
                self.scale,
                self.stabilization,
                self.flags,
                self.frames,
                self.angular,
                self.response,
                self.compliance,
                self.contact_velocity,
                self.gap,
            ],
            device=self.device,
        )
        wp.utils.array_scan(self.flags, self.offsets, inclusive=True)
        wp.launch(kernels.compact_contacts, self.n, [self.flags, self.offsets, self.ids], device=self.device)
        self.impulses.zero_()
        self.wrench.zero_()
        wp.launch(
            kernels.solve_contacts_parallel,
            128,
            [
                self.offsets,
                self.ids,
                self.frames,
                self.angular,
                self.response,
                self.compliance,
                self.contact_velocity,
                self.gap,
                self.particle_mass,
                self.free,
                self.mass,
                self.scale,
                self.friction,
                self.dt,
                self.max_iterations,
                wp.float64(self.tolerance),
                self.impulses,
                self.wrench,
                0,
                self.stats,
            ],
            block_dim=128,
            device=self.device,
        )
        return self.wrench

    def diagnostics(self):
        """Explicit host read for logging, outside the simulation step."""
        stats, wrench = self.stats.numpy(), self.wrench.numpy()[0]
        return dict(
            contacts=int(stats[0]),
            iterations=int(stats[1]),
            residual=float(stats[2]),
            converged=bool(stats[2] <= self.tolerance),
            finite=bool(np.isfinite(wrench).all()),
            wrench=wrench.tolist(),
        )
