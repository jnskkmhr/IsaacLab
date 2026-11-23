# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reconstruct 200 Hz material-on-body contact forces from particle rollouts using Warp contact kernels."""

import importlib
import importlib.machinery
import sys
import types
from pathlib import Path

import numpy as np
import warp as wp

_NERD_PKG = "_showcase_particle_nerd"
_NERD_ROOT = Path(__file__).resolve().parents[2] / "particle_nerd"
if _NERD_PKG not in sys.modules:
    pkg = types.ModuleType(_NERD_PKG)
    pkg.__path__ = [str(_NERD_ROOT)]
    pkg.__spec__ = importlib.machinery.ModuleSpec(_NERD_PKG, loader=None, is_package=True)
    pkg.__spec__.submodule_search_locations = pkg.__path__
    sys.modules[_NERD_PKG] = pkg

contact_kernels = importlib.import_module(_NERD_PKG + ".contact.kernels")


@wp.kernel
def _prepare_kinematic_body(
    velocity: wp.array(dtype=wp.spatial_vector),
    free: wp.array(dtype=wp.vec3d),
    inverse: wp.array(dtype=wp.mat33d),
):
    free[0] = wp.vec3d(wp.spatial_top(velocity[0]))
    free[1] = wp.vec3d(wp.spatial_bottom(velocity[0]))
    inverse[0] = wp.mat33d(0.0)


class BenchmarkForceReconstructor:
    """Evaluate unilateral Coulomb contact impulses from particle positions/velocities onto the benchmark collider."""

    def __init__(
        self,
        particle_mass: np.ndarray,
        size_m: tuple[float, float, float],
        shape: str,
        dt: float = 0.005,
        particle_radius: float = 0.016,
        friction: float = 0.48,
        effective_mass_scale: float = 7.0,
        gradient_radius: float = 0.065,
        contact_band: float = 0.08,
        stabilization: float = 0.01,
        max_iterations: int = 120,
        tolerance: float = 1.0e-8,
        relaxation: float = 0.15,
        device: str = "cuda:0",
    ):
        self.n = len(particle_mass)
        self.dt = float(dt)
        self.friction = float(friction)
        self.scale = float(effective_mass_scale)
        self.radius = float(gradient_radius)
        self.contact_band = float(contact_band)
        self.stabilization = float(stabilization)
        self.max_iterations = int(max_iterations)
        self.tolerance = float(tolerance)
        self.relaxation = float(relaxation)
        self.device = device
        self.mass = float("inf")

        shape_code = {"box": 0, "sphere": 1, "cylinder": 2}[shape]
        inflated_size = tuple(float(s) + 2.0 * float(particle_radius) for s in size_m)
        self.grid = wp.HashGrid(64, 64, 64, device=device)
        self.grid.reserve(self.n)
        self.particle_mass = wp.array(particle_mass.astype(np.float32), dtype=float, device=device)
        self.sizes = wp.array([inflated_size], dtype=wp.vec3, device=device)
        self.shapes = wp.array([shape_code], dtype=int, device=device)
        self.pose = wp.empty(1, dtype=wp.transform, device=device)
        self.velocity = wp.empty(1, dtype=wp.spatial_vector, device=device)
        self.free = wp.empty(2, dtype=wp.vec3d, device=device)
        self.inverse = wp.empty(1, dtype=wp.mat33d, device=device)

        self.gradients = wp.empty(self.n, dtype=wp.mat33d, device=device)
        self.flags = wp.empty(self.n, dtype=int, device=device)
        self.offsets = wp.empty(self.n, dtype=int, device=device)
        self.ids = wp.empty(self.n, dtype=int, device=device)
        self.frames = wp.empty(self.n, dtype=wp.mat33d, device=device)
        self.angular = wp.empty(self.n, dtype=wp.mat33d, device=device)
        self.response = wp.empty(self.n, dtype=wp.mat33d, device=device)
        self.compliance = wp.empty(self.n, dtype=wp.mat33d, device=device)
        self.contact_velocity = wp.empty(self.n, dtype=wp.vec3d, device=device)
        self.gap = wp.empty(self.n, dtype=wp.float64, device=device)
        self.impulses = wp.empty(self.n, dtype=wp.vec3d, device=device)
        self.wrench = wp.empty(1, dtype=wp.spatial_vector, device=device)
        self.stats = wp.empty(3, dtype=wp.float64, device=device)
        self.dummy_sdf = wp.zeros((2, 2, 2), dtype=float, device=device)

    def estimate_step(
        self,
        pos_t: np.ndarray,
        vel_next: np.ndarray,
        body_pose_t: np.ndarray,
        body_vel_t: np.ndarray,
    ) -> np.ndarray:
        """Compute world-frame force [Fx, Fy, Fz] exerted by sand particles on the collider at step t."""
        position = wp.array(pos_t.astype(np.float32), dtype=wp.vec3, device=self.device)
        velocity = wp.array(vel_next.astype(np.float32), dtype=wp.vec3, device=self.device)
        self.pose.assign(np.asarray(body_pose_t[:7], dtype=np.float32)[None])
        self.velocity.assign(np.asarray(body_vel_t[:6], dtype=np.float32)[None])
        self.grid.build(position, self.radius)

        wp.launch(
            contact_kernels.fit_gradients,
            self.n,
            [
                self.grid.id,
                position,
                velocity,
                self.radius,
                self.gradients,
                self.pose,
                self.sizes,
                self.shapes,
                0,
                self.contact_band,
            ],
            device=self.device,
        )
        wp.launch(
            _prepare_kinematic_body,
            1,
            [self.velocity, self.free, self.inverse],
            device=self.device,
        )
        wp.launch(
            contact_kernels.prepare_contacts,
            self.n,
            [
                position,
                velocity,
                self.gradients,
                self.pose,
                self.sizes,
                self.shapes,
                0,
                self.free,
                self.inverse,
                self.mass,
                self.particle_mass,
                self.dt,
                self.scale,
                self.scale,
                self.scale,
                self.scale,
                1.0,
                self.stabilization,
                self.flags,
                self.frames,
                self.angular,
                self.response,
                self.compliance,
                self.contact_velocity,
                self.gap,
                False,
                self.dummy_sdf,
                wp.vec3(0.0, 0.0, 0.0),
                1.0,
            ],
            device=self.device,
        )
        wp.utils.array_scan(self.flags, self.offsets, inclusive=True)
        wp.launch(
            contact_kernels.compact_contacts,
            self.n,
            [self.flags, self.offsets, self.ids],
            device=self.device,
        )
        self.impulses.zero_()
        self.wrench.zero_()
        wp.launch(
            contact_kernels.solve_contacts_parallel,
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
                self.scale,
                self.scale,
                self.scale,
                1.0,
                self.friction,
                self.dt,
                self.max_iterations,
                wp.float64(self.tolerance),
                self.impulses,
                self.wrench,
                0,
                self.stats,
                False,
                self.relaxation,
            ],
            block_dim=128,
            device=self.device,
        )
        return self.wrench.numpy()[0, :3].copy()

    def estimate_trajectory(
        self,
        positions: np.ndarray,
        body_poses: np.ndarray,
        body_velocities: np.ndarray,
    ) -> np.ndarray:
        """Reconstruct [T, 3] material forces across an entire trajectory [T+1, N, 3]."""
        steps = len(positions) - 1
        forces = np.zeros((steps, 3), dtype=np.float32)
        for t in range(steps):
            vel_next = (positions[t + 1] - positions[t]) / self.dt
            forces[t] = self.estimate_step(positions[t], vel_next, body_poses[t, 0], body_velocities[t, 0])
        return forces
