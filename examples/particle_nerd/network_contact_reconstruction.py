# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Project predicted particle motion and one dynamic body through a contact network.

All contacts are frictionless in this experiment. Neighboring material points
are treated as noncohesive spheres with an effective separation distance. This
is a velocity-projection approximation, not the MPM material law. It changes
particle velocities and must therefore be assessed for both force and motion.
Recorded particle velocities currently stand in for a NeRD prediction.
"""

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import osqp
from reconstruction import box_contacts
from scipy.sparse import block_diag, coo_matrix, csc_matrix, diags
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


@dataclass
class NetworkContactResult:
    body_wrench: np.ndarray
    body_velocity: np.ndarray
    particle_velocity: np.ndarray
    converged: bool
    status: str
    solve_seconds: float
    iterations: int
    internal_contacts: int
    body_contacts: int
    momentum_correction_residual: float


def project_contact_network(
    particle_position,
    particle_velocity,
    particle_mass,
    particle_radius,
    body_pose,
    body_velocity,
    body_mass,
    body_inertia,
    body_size,
    dynamic_body,
    *,
    dt,
    gravity=(0, 0, -9.81),
    radius_scale=1.0,
    contact_offset=0.0,
    stabilization=0.0,
    pair_stabilization=0.0,
    max_iterations=10000,
):
    """Minimize mass-weighted velocity changes subject to normal contact conditions.

    Fixed bodies may have supplied kinematic velocities; the dynamic body uses
    its physical mass and inertia. Material velocities are targets from the
    predictor, so their gravity/internal update is not integrated a second time.
    A strict convex velocity objective selects a unique dynamic-body momentum
    correction even if individual contact multipliers remain nonunique.
    """
    start = perf_counter()
    x, v, mass, radius, poses, velocities, masses, inertias, sizes = [
        np.asarray(a, dtype=float)
        for a in (
            particle_position,
            particle_velocity,
            particle_mass,
            particle_radius,
            body_pose,
            body_velocity,
            body_mass,
            body_inertia,
            body_size,
        )
    ]
    if dt <= 0 or radius_scale <= 0 or contact_offset < 0 or stabilization < 0 or pair_stabilization < 0:
        raise ValueError("Invalid timestep or contact parameters.")
    if any(not np.isfinite(a).all() for a in (x, v, mass, radius, poses, velocities, masses, inertias, sizes)):
        raise ValueError("Inputs must be finite.")
    if np.any(mass <= 0) or masses[dynamic_body] <= 0 or np.any(radius <= 0):
        raise ValueError("Dynamic masses and particle radii must be positive.")
    n = len(x)
    d = dynamic_body
    offset = 3 * n
    rotation = Rotation.from_quat(poses[d, 3:]).as_matrix()
    inertia = rotation @ inertias[d] @ rotation.T
    body_matrix = block_diag((csc_matrix(masses[d] * np.eye(3)), inertia), format="csc")
    free_body = velocities[d].copy()
    free_body[:3] += dt * np.asarray(gravity)
    free_body[3:] -= dt * np.linalg.solve(inertia, np.cross(free_body[3:], inertia @ free_body[3:]))
    target_velocity = np.r_[v.ravel(), free_body]
    metric = block_diag((diags(np.repeat(mass, 3)), body_matrix), format="csc")
    rows, cols, values, lower = [], [], [], []
    contact_rows, contact_bodies, contact_normals, contact_arms = [], [], [], []
    axes = np.arange(3)
    # Include any pair that can reach its effective separation in this step.
    search_radius = 2 * radius.max() * radius_scale + 2 * dt * np.linalg.norm(v, axis=1).max()
    pairs = cKDTree(x).query_pairs(search_radius, output_type="ndarray")
    difference = x[pairs[:, 0]] - x[pairs[:, 1]]
    distance = np.linalg.norm(difference, axis=1)
    if np.any(distance < 1e-12):
        raise ValueError("Coincident particles do not define a contact normal.")
    normals = difference / distance[:, None]
    gap = distance - radius_scale * (radius[pairs[:, 0]] + radius[pairs[:, 1]])
    speed = np.einsum("ij,ij->i", normals, v[pairs[:, 0]] - v[pairs[:, 1]])
    selected = np.flatnonzero(gap <= np.maximum(-speed, 0) * dt + 1e-10)
    for k in selected:
        row = len(lower)
        for particle, sign in ((pairs[k, 0], 1), (pairs[k, 1], -1)):
            rows.extend([row] * 3)
            cols.extend(3 * particle + axes)
            values.extend(sign * normals[k])
        lower.append(-(max(gap[k], 0) + pair_stabilization * min(gap[k], 0)) / dt)
    internal_contacts = len(lower)
    for b in range(len(poses)):
        distance, normals, points = box_contacts(x, poses[b], sizes[b])
        arm = points - poses[b, :3]
        body_free = free_body if b == d else velocities[b]
        surface_velocity = body_free[:3] + np.cross(body_free[3:], arm)
        gap = distance - contact_offset
        speed = np.einsum("ij,ij->i", normals, v - surface_velocity)
        for particle in np.flatnonzero(gap <= np.maximum(-speed, 0) * dt + 1e-10):
            row = len(lower)
            normal = normals[particle]
            rows.extend([row] * 3)
            cols.extend(3 * particle + axes)
            values.extend(normal)
            bound = -(max(gap[particle], 0) + stabilization * min(gap[particle], 0)) / dt
            if b == d:
                jacobian = np.r_[normal, np.cross(arm[particle], normal)]
                rows.extend([row] * 6)
                cols.extend(offset + np.arange(6))
                values.extend(-jacobian)
            else:
                bound += np.dot(normal, surface_velocity[particle])
            lower.append(bound)
            contact_rows.append(row)
            contact_bodies.append(b)
            contact_normals.append(normal)
            contact_arms.append(arm[particle])
    count = len(lower)
    if not count:
        return NetworkContactResult(
            np.zeros((len(poses), 6)), free_body, v.copy(), True, "no contacts", perf_counter() - start, 0, 0, 0, 0.0
        )
    matrix = coo_matrix((values, (rows, cols)), shape=(count, offset + 6)).tocsc()
    problem = osqp.OSQP()
    problem.setup(
        P=metric,
        q=-metric @ target_velocity,
        A=matrix,
        l=np.array(lower),
        u=np.full(count, np.inf),
        eps_abs=1e-7,
        eps_rel=1e-7,
        max_iter=max_iterations,
        polishing=True,
        verbose=False,
    )
    fit = problem.solve(raise_error=False)
    if fit.x is None or not np.isfinite(fit.x).all():
        raise RuntimeError(f"Contact network projection failed: {fit.info.status}")
    wrench = np.zeros((len(poses), 6))
    if contact_rows:
        # The lower-bound dual is negative; its force on a rigid body points inward.
        body_force = fit.y[contact_rows, None] * np.array(contact_normals) / dt
        moments = np.cross(np.array(contact_arms), body_force)
        np.add.at(wrench, contact_bodies, np.c_[body_force, moments])
    correction = metric @ (fit.x - target_velocity)
    residual = float(np.linalg.norm(correction + matrix.T @ fit.y))
    return NetworkContactResult(
        wrench,
        fit.x[offset:],
        fit.x[:offset].reshape(n, 3),
        fit.info.status == "solved",
        fit.info.status,
        perf_counter() - start,
        fit.info.iter,
        internal_contacts,
        len(contact_rows),
        residual,
    )
