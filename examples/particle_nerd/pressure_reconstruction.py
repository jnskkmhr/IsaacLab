# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Experimental inverse meshfree pressure closure; no MPM update or force network.

One pressure per material particle controls its pair forces and boundary traction.
The Wendland C2 gradient supplies internal force weights. Boundary traction uses
its analytical planar half-space integral, approximating each box locally by its
nearest face. Four tangential rays obey a Coulomb bound tied to that pressure.
This is an isotropic-stress approximation, not a granular constitutive model.
Kernel reference: SPlisHSPlasH/SPHKernels.h, WendlandQuinticC2Kernel.
"""

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import osqp
from reconstruction import box_contacts
from scipy.sparse import bmat, coo_matrix, diags, eye
from scipy.spatial import cKDTree


@dataclass
class PressureResult:
    body_wrench: np.ndarray
    pressure: np.ndarray
    relative_residual: float
    converged: bool
    status: str
    solve_seconds: float
    objective: float
    primal_residual: float
    dual_residual: float


def planar_kernel_weight(distance, radius):
    """Twice the Wendland C2 plane integral [1/m]; its one-sided integral is one."""
    q = np.clip(np.asarray(distance) / radius, 0, 1)
    primitive = q**2 / 2 - 2.5 * q**4 + 4 * q**5 - 2.5 * q**6 + 4 * q**7 / 7
    return np.maximum(42 / radius * (1 / 14 - primitive), 0)


def reconstruct_pressure_reactions(
    position,
    velocity_before,
    velocity_after,
    mass,
    body_pose,
    body_size,
    friction,
    *,
    dt,
    density=1600.0,
    gravity=(0, 0, -9.81),
    support_radius=0.08,
    boundary_radius=0.04,
    regularization=1e-6,
    max_iterations=10000,
):
    """Infer compressive pressures and bounded boundary friction from particle motion.

    Fixed nominal particle volumes are mass/density. Unknown pressures are scaled
    by 1000 Pa for numerical conditioning; the regularizer penalizes these scaled
    pressures and tangential-force coefficients, so it is a numerical prior.
    Neither measured body forces nor future body acceleration enter the solve.
    """
    start = perf_counter()
    position, before, after, mass, pose, size, mu = [
        np.asarray(a, dtype=float)
        for a in (position, velocity_before, velocity_after, mass, body_pose, body_size, friction)
    ]
    n, bodies = len(mass), len(pose)
    if dt <= 0 or density <= 0 or min(support_radius, boundary_radius) <= 0 or regularization < 0:
        raise ValueError("Invalid timestep, density, support radius, or regularization.")
    if not all(np.isfinite(a).all() for a in (position, before, after, mass, pose, size, mu)):
        raise ValueError("Nonfinite reconstruction input.")
    volume = mass / density
    pressure_scale = 1000.0
    rhs = (mass[:, None] * ((after - before) / dt - np.asarray(gravity))).ravel()
    pairs = cKDTree(position).query_pairs(support_radius, output_type="ndarray")
    displacement = position[pairs[:, 0]] - position[pairs[:, 1]]
    length = np.linalg.norm(displacement, axis=1)
    if np.any(length < 1e-12):
        raise ValueError("Coincident particles do not define a pressure gradient.")
    q = length / support_radius
    gradient = 210 / (np.pi * support_radius**4) * q * (1 - q) ** 3
    weights = volume[pairs[:, 0]] * volume[pairs[:, 1]] * gradient * pressure_scale
    forces = weights[:, None] * displacement / length[:, None]
    rows, cols, data = [], [], []
    axes = np.arange(3)
    for destination, sign in ((0, 1), (1, -1)):
        for source in (0, 1):
            rows.extend((3 * pairs[:, destination, None] + axes).ravel())
            cols.extend(np.repeat(pairs[:, source], 3))
            data.extend((sign * forces).ravel())
    wrows, wcols, wdata = [], [], []
    crows, ccols, cdata = [], [], []
    column, contact = n, 0
    for body in range(bodies):
        distances, normals, points = box_contacts(position, pose[body], size[body])
        for particle in np.flatnonzero(distances < boundary_radius):
            weight = float(
                volume[particle] * planar_kernel_weight(distances[particle], boundary_radius) * pressure_scale
            )
            normal = normals[particle]
            force = weight * normal
            rows.extend(3 * particle + axes)
            cols.extend([particle] * 3)
            data.extend(force)
            moment = np.cross(points[particle] - pose[body, :3], -force)
            wrows.extend(6 * body + np.arange(6))
            wcols.extend([particle] * 6)
            wdata.extend(np.r_[-force, moment])
            tangent = np.cross(normal, np.eye(3)[np.argmin(np.abs(normal))])
            tangent /= np.linalg.norm(tangent)
            other = np.cross(normal, tangent)
            crows.append(contact)
            ccols.append(particle)
            cdata.append(-mu[body] * weight)
            for direction in (tangent, -tangent, other, -other):
                rows.extend(3 * particle + axes)
                cols.extend([column] * 3)
                data.extend(direction)
                moment = np.cross(points[particle] - pose[body, :3], -direction)
                wrows.extend(6 * body + np.arange(6))
                wcols.extend([column] * 6)
                wdata.extend(np.r_[-direction, moment])
                crows.append(contact)
                ccols.append(column)
                cdata.append(1.0)
                column += 1
            contact += 1
    matrix = coo_matrix((data, (rows, cols)), shape=(3 * n, column)).tocsc()
    wrench_map = coo_matrix((wdata, (wrows, wcols)), shape=(6 * bodies, column)).tocsr()
    friction_bound = coo_matrix((cdata, (crows, ccols)), shape=(contact, column)).tocsc()
    constraints = bmat([[matrix, -eye(3 * n)], [eye(column), None], [friction_bound, None]], format="csc")
    problem = osqp.OSQP()
    problem.setup(
        P=diags(np.r_[np.full(column, regularization), np.ones(3 * n)], format="csc"),
        q=np.zeros(column + 3 * n),
        A=constraints,
        l=np.r_[rhs, np.zeros(column), np.full(contact, -np.inf)],
        u=np.r_[rhs, np.full(column, np.inf), np.zeros(contact)],
        eps_abs=1e-7,
        eps_rel=1e-7,
        max_iter=max_iterations,
        polishing=True,
        verbose=False,
    )
    fit = problem.solve(raise_error=False)
    if fit.x is None or not np.isfinite(fit.x).all():
        raise RuntimeError(f"Pressure QP failed: {fit.info.status}")
    coefficients = fit.x[:column]
    residual = matrix @ coefficients - rhs
    return PressureResult(
        body_wrench=(wrench_map @ coefficients).reshape(bodies, 6),
        pressure=pressure_scale * coefficients[:n],
        relative_residual=float(np.linalg.norm(residual) / max(np.linalg.norm(rhs), 1e-12)),
        converged=fit.info.status == "solved",
        status=fit.info.status,
        solve_seconds=perf_counter() - start,
        objective=float(residual @ residual + regularization * (coefficients @ coefficients)),
        primal_residual=float(fit.info.prim_res),
        dual_residual=float(fit.info.dual_res),
    )
