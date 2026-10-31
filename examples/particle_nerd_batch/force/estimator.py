# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared GPU buffers and launches for independent single-intruder contact solves."""

import numpy as np
import warp as wp

from . import kernels
from .body_velocity import initialize_body_velocity_solve
from .grid import Grid


class ContactEstimator:
    """Return [E] material-on-intruder wrenches, world force then COM torque.

    All worlds share geometry, body mass/inertia and contact-law settings. Their
    states, particle masses, contacts and solver convergence are independent.
    Particle velocities are predicted at n+1; particle positions remain at n.
    """

    def __init__(
        self,
        particle_mass,
        body_mass,
        body_inertia,
        body_size,
        shape,
        sdf,
        dt,
        effective_mass_scale=180.0,
        relaxation=0.075,
        max_iterations=2000,
        tolerance=1e-9,
        gradient_radius=0.065,
        friction=0.6,
        gravity=(0, 0, -9.81),
        device="cuda:0",
        rft=False,
        horiz_mass_scale=None,
        up_mass_scale=None,
        tangent_mass_scale=None,
        normal_power=None,
    ):
        masses = np.asarray(particle_mass, dtype=np.float32)
        if masses.ndim != 2 or not np.isfinite(masses).all() or np.any(masses <= 0):
            raise ValueError("Particle masses must be positive [E,N]")
        if min(body_mass, dt, effective_mass_scale, tolerance, gradient_radius) <= 0 or max_iterations < 1:
            raise ValueError("Mass, timestep, scale, radius, tolerance and iteration limit must be positive")
        if not 0 < relaxation <= 1 or friction < 0:
            raise ValueError("Invalid contact relaxation or friction")
        self.environments, self.particles = masses.shape
        self.total = masses.size
        self.device, self.dt, self.mass = device, float(dt), float(body_mass)
        self.scale, self.relaxation = float(effective_mass_scale), float(relaxation)
        self.horiz_scale = float(horiz_mass_scale if horiz_mass_scale is not None else (8.0 if rft else self.scale))
        self.up_scale = float(up_mass_scale if up_mass_scale is not None else (2.0 if rft else self.scale))
        self.tangent_scale = float(tangent_mass_scale if tangent_mass_scale is not None else (4.0 if rft else self.scale))
        self.normal_power = float(normal_power if normal_power is not None else (3.0 if rft else 1.0))
        self.max_iterations, self.tolerance = max_iterations, tolerance
        self.radius, self.friction = gradient_radius, friction
        self.gravity = wp.vec3(*gravity)
        inertia = np.asarray(body_inertia, dtype=np.float64)
        self.inertia = wp.mat33d(*((inertia + inertia.T) * 0.5).flatten())
        self.sdf = (wp.from_torch(sdf[0]), wp.vec3(*sdf[1]), sdf[2])
        self.particle_mass = wp.array(masses.ravel(), device=device)
        self.sizes = wp.array(np.tile(body_size, (self.environments, 1)), dtype=wp.vec3, device=device)
        self.shapes = wp.full(
            self.environments, {"box": 0, "sphere": 1, "cylinder": 2, "shovel": 3}[shape], dtype=int, device=device
        )

        def zeros(count, dtype):
            return wp.zeros(count, dtype=dtype, device=device)

        self.grid = Grid(self.environments, self.particles, self.radius, device)
        self.gradients = zeros(self.total, wp.mat33d)
        self.free, self.inverse = zeros(2 * self.environments, wp.vec3d), zeros(self.environments, wp.mat33d)
        self.flags, self.offsets, self.ids = [zeros(self.total, int) for _ in range(3)]
        self.frames, self.angular, self.response, self.compliance = [zeros(self.total, wp.mat33d) for _ in range(4)]
        self.material, self.impulses = [zeros(self.total, wp.vec3d) for _ in range(2)]
        self.gap = zeros(self.total, wp.float64)
        self.stats, self.initializer_stats = [zeros(3 * self.environments, wp.float64) for _ in range(2)]
        self.initial_velocity = zeros(2 * self.environments, wp.vec3d)
        self.wrench = zeros(self.environments, wp.spatial_vector)

    def estimate(self, position, next_velocity, body_pose, body_velocity, external):
        self.grid.build(position)
        wp.launch(
            kernels.fit_gradients,
            self.total,
            [
                self.grid.starts,
                self.grid.ends,
                self.grid.ids,
                self.grid.dim,
                position,
                next_velocity,
                self.radius,
                self.gradients,
                body_pose,
                self.sizes,
                self.shapes,
                self.particles,
                -1.0,
            ],
            device=self.device,
        )
        wp.launch(
            kernels.prepare_body,
            self.environments,
            [
                body_pose,
                body_velocity,
                external,
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
            self.total,
            [
                position,
                next_velocity,
                self.gradients,
                body_pose,
                self.sizes,
                self.shapes,
                self.particles,
                self.free,
                self.inverse,
                self.mass,
                self.particle_mass,
                self.dt,
                self.scale,
                self.horiz_scale,
                self.up_scale,
                self.tangent_scale,
                self.normal_power,
                0.01,
                self.flags,
                self.frames,
                self.angular,
                self.response,
                self.compliance,
                self.material,
                self.gap,
                False,
                *self.sdf,
            ],
            device=self.device,
        )
        wp.utils.array_scan(self.flags, self.offsets, inclusive=True)
        wp.launch(kernels.compact_contacts, self.total, [self.flags, self.offsets, self.ids], device=self.device)
        wp.launch(
            initialize_body_velocity_solve,
            self.environments * 128,
            [
                self.particles,
                self.offsets,
                self.ids,
                self.frames,
                self.angular,
                self.response,
                self.material,
                self.gap,
                self.particle_mass,
                self.free,
                self.mass,
                self.scale,
                self.horiz_scale,
                self.up_scale,
                self.tangent_scale,
                self.normal_power,
                self.friction,
                self.impulses,
                self.initial_velocity,
                self.initializer_stats,
            ],
            block_dim=128,
            device=self.device,
        )
        wp.launch(
            kernels.solve_contacts_parallel,
            self.environments * 128,
            [
                self.offsets,
                self.ids,
                self.frames,
                self.angular,
                self.response,
                self.compliance,
                self.material,
                self.gap,
                self.particle_mass,
                self.initial_velocity,
                self.mass,
                self.scale,
                self.horiz_scale,
                self.up_scale,
                self.tangent_scale,
                self.normal_power,
                self.friction,
                self.dt,
                self.max_iterations,
                wp.float64(self.tolerance),
                self.impulses,
                self.wrench,
                self.particles,
                self.stats,
                False,
                self.relaxation,
            ],
            block_dim=128,
            device=self.device,
        )
        return self.wrench
