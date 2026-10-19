# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Experimental motion-to-reaction reconstruction without MPM stresses or force labels.

Closure: neighboring material particles transmit compressive central forces.
Body contacts transmit nonnegative combinations of four friction-cone rays.
A small quadratic penalty selects one solution when force balance is underdetermined.
This is a measurable baseline, not Newton's granular constitutive law.
"""

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy.optimize import lsq_linear
from scipy.sparse import bmat, coo_matrix, diags, eye, vstack
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


@dataclass
class ReconstructionResult:
    body_wrench: np.ndarray
    particle_residual: np.ndarray
    relative_residual: float
    converged: bool
    solve_seconds: float
    internal_edges: int
    contacts: int
    objective: float
    projected_gradient_relative: float
    optimizer_status: str


def box_contacts(position, pose, size):
    """Return signed distance, outward normal, and closest point for an oriented box.

    Quaternion convention is xyzw. For an interior particle, choose its nearest
    face; the signed distance is negative. All outputs use the input coordinate frame.
    """
    rotation = Rotation.from_quat(pose[3:]).as_matrix()
    local = (position - pose[:3]) @ rotation
    half = size * 0.5
    closest = np.clip(local, -half, half)
    delta = local - closest
    distance = np.linalg.norm(delta, axis=1)
    normal = delta / np.maximum(distance[:, None], 1e-12)
    inside = distance < 1e-12
    rows = np.nonzero(inside)[0]
    face = np.argmax(np.abs(local[rows]) - half, axis=1)
    sign = np.where(local[rows, face] >= 0, 1.0, -1.0)
    closest[rows, face] = sign * half[face]
    normal[rows] = 0.0
    normal[rows, face] = sign
    distance[rows] = np.abs(local[rows, face]) - half[face]
    return distance, normal @ rotation.T, closest @ rotation.T + pose[:3]


def reconstruct_reactions(
    position,
    velocity_before,
    velocity_after,
    mass,
    body_pose,
    body_size,
    friction,
    *,
    dt,
    gravity=(0.0, 0.0, -9.81),
    neighbor_radius=0.065,
    contact_distance=0.04,
    regularization=1e-4,
    max_iterations=100,
    optimizer="trf",
    known_particle_force=None,
):
    """Estimate material-on-body forces [N] and torques about body centers [N m].

    Particle positions and body poses define the contact geometry at the end of
    the interval. Acceleration is (velocity_after - velocity_before) / dt.
    No measured body reactions or body accelerations enter this function.
    Optional known_particle_force supplies a separately estimated contact load
    on each material particle. Omit those bodies from body_pose to avoid counting
    their contacts twice; the solve then checks the remaining force balance.
    Neighbor edges are rebuilt for all particles; there is no region selection.
    """
    if optimizer not in ("trf", "osqp"):
        raise ValueError("Optimizer must be trf or osqp.")
    start = perf_counter()
    position, velocity_before, velocity_after, mass, body_pose, body_size, friction = (
        np.asarray(a, dtype=np.float64)
        for a in (position, velocity_before, velocity_after, mass, body_pose, body_size, friction)
    )
    n, b = len(position), len(body_pose)
    arrays = (position, velocity_before, velocity_after, mass, body_pose, body_size, friction)
    if not all(np.isfinite(a).all() for a in arrays):
        raise ValueError("Reconstruction inputs must be finite.")
    if position.shape != (n, 3) or velocity_before.shape != position.shape or velocity_after.shape != position.shape:
        raise ValueError("Particle positions and velocities must have shape (N, 3).")
    if mass.shape != (n,) or body_pose.shape != (b, 7) or body_size.shape != (b, 3) or friction.shape != (b,):
        raise ValueError("Invalid particle mass or body geometry dimensions.")
    if dt <= 0 or neighbor_radius <= 0 or contact_distance < 0 or regularization <= 0:
        raise ValueError("Timestep, neighborhood radius, and regularization must be positive.")
    if np.any(mass <= 0) or np.any(body_size <= 0) or np.any(friction < 0):
        raise ValueError("Mass/size must be positive and friction nonnegative.")
    rhs = mass[:, None] * ((velocity_after - velocity_before) / dt - np.asarray(gravity))
    if known_particle_force is not None:
        known_particle_force = np.asarray(known_particle_force, dtype=float)
        if known_particle_force.shape != position.shape or not np.isfinite(known_particle_force).all():
            raise ValueError("Known particle forces must be finite with shape (N, 3).")
        rhs -= known_particle_force
    edges = cKDTree(position).query_pairs(neighbor_radius, output_type="ndarray")
    directions = position[edges[:, 0]] - position[edges[:, 1]]
    lengths = np.linalg.norm(directions, axis=1)
    if np.any(lengths < 1e-12):
        raise ValueError("Coincident particles do not define a central-force direction.")
    directions /= lengths[:, None]
    rows, cols, values = [], [], []
    axes = np.arange(3)
    edge_count = len(edges)
    for side, sign in ((0, 1.0), (1, -1.0)):
        rows.extend((3 * edges[:, side, None] + axes).ravel())
        cols.extend(np.repeat(np.arange(edge_count), 3))
        values.extend((sign * directions).ravel())
    # Each contact column stores an equal-and-opposite material-on-body wrench.
    contact_columns, body_ids, contact_wrenches = [], [], []
    column = edge_count
    contacts = 0
    for body in range(b):
        distance, normals, points = box_contacts(position, body_pose[body], body_size[body])
        for particle in np.flatnonzero(distance <= contact_distance):
            normal = normals[particle]
            axis = np.eye(3)[np.argmin(np.abs(normal))]
            tangent = np.cross(normal, axis)
            tangent /= np.linalg.norm(tangent)
            second_tangent = np.cross(normal, tangent)
            # An inscribed four-sided Coulomb cone: tangential magnitude <= mu * normal force.
            for direction in (tangent, -tangent, second_tangent, -second_tangent):
                force_on_particle = normal + friction[body] * direction
                rows.extend(3 * particle + axes)
                cols.extend([column] * 3)
                values.extend(force_on_particle)
                force_on_body = -force_on_particle
                moment = np.cross(points[particle] - body_pose[body, :3], force_on_body)
                contact_columns.append(column)
                body_ids.append(body)
                contact_wrenches.append(np.r_[force_on_body, moment])
                column += 1
            contacts += 1
    matrix = coo_matrix((values, (rows, cols)), shape=(3 * n, column)).tocsr()
    status = "empty system"
    if column:
        if optimizer == "trf":
            system = vstack((matrix, np.sqrt(regularization) * eye(column, format="csr")), format="csr")
            target = np.r_[rhs.ravel(), np.zeros(column)]
            fit = lsq_linear(system, target, bounds=(0, np.inf), tol=1e-6, lsmr_tol=1e-6, max_iter=max_iterations)
            coefficients = fit.x
            converged = bool(fit.success)
            status = fit.message
        else:
            import osqp

            # Residual variables avoid forming the dense normal matrix A.T @ A.
            # min 0.5 * ||r||^2 + 0.5 * regularization * ||x||^2
            # subject to A x - r = particle load, x >= 0.
            count = matrix.shape[0]
            constraints = bmat([[matrix, -eye(count)], [eye(column), None]], format="csc")
            problem = osqp.OSQP()
            problem.setup(
                P=diags(np.r_[np.full(column, regularization), np.ones(count)], format="csc"),
                q=np.zeros(column + count),
                A=constraints,
                l=np.r_[rhs.ravel(), np.zeros(column)],
                u=np.r_[rhs.ravel(), np.full(column, np.inf)],
                eps_abs=1e-7,
                eps_rel=1e-7,
                max_iter=max_iterations,
                polishing=True,
                verbose=False,
            )
            fit = problem.solve(raise_error=False)
            if fit.x is None or not np.isfinite(fit.x).all():
                raise RuntimeError(f"Force QP failed: {fit.info.status}")
            coefficients = np.maximum(fit.x[:column], 0)
            converged = fit.info.status == "solved"
            status = fit.info.status
        gradient = matrix.T @ (matrix @ coefficients - rhs.ravel()) + regularization * coefficients
        projected = np.where(coefficients > 1e-7, gradient, np.minimum(gradient, 0))
        optimality = float(np.max(np.abs(projected)) / max(1, np.max(np.abs(matrix.T @ rhs.ravel()))))
    else:
        coefficients = np.empty(0)
        converged = True
        optimality = 0.0
    wrench = np.zeros((b, 6))
    if contact_columns:
        np.add.at(wrench, body_ids, coefficients[contact_columns, None] * np.asarray(contact_wrenches))
    residual = (matrix @ coefficients).reshape(n, 3) - rhs
    return ReconstructionResult(
        body_wrench=wrench,
        particle_residual=residual,
        relative_residual=float(np.linalg.norm(residual) / max(np.linalg.norm(rhs), 1e-12)),
        converged=converged,
        solve_seconds=perf_counter() - start,
        internal_edges=edge_count,
        contacts=contacts,
        objective=float(np.sum(residual**2) + regularization * np.sum(coefficients**2)),
        projected_gradient_relative=optimality,
        optimizer_status=str(status),
    )
