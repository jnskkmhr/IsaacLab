# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Nonlearned DEM-inspired force reconstruction from prescribed material-point motion.

The normal spring and approach-only damping follow the structure of Warp's DEM
example (https://github.com/NVIDIA/warp/blob/main/warp/examples/core/example_dem.py).
This implementation is noncohesive and optionally adds tangential spring history.
It evaluates contact laws; it never integrates particle or rigid-body motion.
MPM radii set an explicit effective pair rest distance, not a physical grain claim.
"""

from dataclasses import dataclass, field
from time import perf_counter

import numpy as np
from reconstruction import box_contacts
from scipy.optimize import minimize
from scipy.sparse import coo_matrix
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class DemParameters:
    normal_stiffness: float = 10000.0  # N/m, effective stiffness, NOT MPM Young's modulus.
    damping_ratio: float = 0.2
    tangential_stiffness_ratio: float = 0.5
    internal_friction: float = 0.48
    radius_scale: float = 1.0
    body_offset: float = 0.0  # MPM material-point centers contact the surface in this dataset.
    neighbor_radius: float = 0.065
    contact_distance: float = 0.04
    friction_model: str = "history"

    def __post_init__(self):
        values = np.array(
            [
                self.normal_stiffness,
                self.damping_ratio,
                self.tangential_stiffness_ratio,
                self.internal_friction,
                self.radius_scale,
                self.body_offset,
                self.neighbor_radius,
                self.contact_distance,
            ]
        )
        if not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError("DEM parameters must be finite and nonnegative.")
        if min(self.normal_stiffness, self.tangential_stiffness_ratio, self.radius_scale, self.neighbor_radius) <= 0:
            raise ValueError("Stiffness, radius scale, and neighbor radius must be positive.")
        if self.friction_model not in ("velocity", "history"):
            raise ValueError("friction_model must be velocity or history.")
        if self.body_offset > self.contact_distance:
            raise ValueError("Candidate contact distance must contain the effective body surface.")


@dataclass
class ContactHistory:
    """Tangential displacement [m], keyed by persistent particle/body indices.

    Create a fresh instance at episode reset. Call assemble_dem_frame at every
    recorded step, including steps at which no inverse solve is requested.
    Particle rotations are unavailable; histories are projected into each new
    contact tangent plane without inventing grain angular velocities.
    """

    displacement: dict = field(default_factory=dict)


@dataclass
class DemFrame:
    balance_matrix: object
    contact_matrix: object
    contact_target: np.ndarray
    rhs: np.ndarray
    direct_body_wrench: np.ndarray
    body_ids: np.ndarray
    body_contact_indices: np.ndarray
    body_moment_arms: np.ndarray
    body_count: int
    assembly_seconds: float
    internal_edges: int
    contacts: int

    def body_wrench(self, contact_forces):
        wrench = np.zeros((self.body_count, 6))
        force = -contact_forces[self.body_contact_indices]
        np.add.at(wrench[:, :3], self.body_ids, force)
        np.add.at(wrench[:, 3:], self.body_ids, np.cross(self.body_moment_arms, force))
        return wrench


def assemble_dem_frame(
    position,
    velocity,
    mass,
    radii,
    body_pose,
    body_velocity,
    body_size,
    friction,
    acceleration,
    *,
    dt,
    parameters,
    history,
    gravity=(0.0, 0.0, -9.81),
):
    """Assemble a frame using end-of-step geometry/velocity and interval acceleration.

    Body velocities are linear then angular in world axes, about body centers.
    Four nonnegative rays approximate each pair/body Coulomb cone. Pair forces
    obey action-reaction. Tangential pair forces do not include grain spin: this
    is a translational material-point approximation, not full angular DEM.
    """
    start = perf_counter()
    position, velocity, mass, radii, body_pose, body_velocity, body_size, friction, acceleration = (
        np.asarray(a, dtype=np.float64)
        for a in (position, velocity, mass, radii, body_pose, body_velocity, body_size, friction, acceleration)
    )
    n, nb = len(position), len(body_pose)
    expected = (
        (position, (n, 3)),
        (velocity, (n, 3)),
        (mass, (n,)),
        (radii, (n,)),
        (body_pose, (nb, 7)),
        (body_velocity, (nb, 6)),
        (body_size, (nb, 3)),
        (friction, (nb,)),
        (acceleration, (n, 3)),
    )
    if any(a.shape != shape or not np.isfinite(a).all() for a, shape in expected):
        raise ValueError("Invalid or nonfinite DEM state arrays.")
    if not np.isfinite(dt) or dt <= 0 or np.any(mass <= 0) or np.any(radii <= 0) or np.any(friction < 0):
        raise ValueError("dt, masses, radii must be positive and friction nonnegative.")
    if np.any(body_size <= 0) or np.asarray(gravity).shape != (3,) or not np.isfinite(gravity).all():
        raise ValueError("Invalid body sizes or gravity.")
    if 2 * radii.max(initial=0) * parameters.radius_scale > parameters.neighbor_radius:
        raise ValueError("Neighbor radius must cover all effective particle contacts.")
    pairs = cKDTree(position).query_pairs(parameters.neighbor_radius, output_type="ndarray")
    normals, overlaps, relative_velocities, effective_masses, mus, keys, particle_i, particle_j = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    body_ids, body_indices, arms = [], [], []
    for i, j in pairs:
        delta = position[i] - position[j]
        distance = np.linalg.norm(delta)
        if distance < 1e-12:
            raise ValueError("Coincident particles do not define a contact normal.")
        normals.append(delta / distance)
        overlaps.append((radii[i] + radii[j]) * parameters.radius_scale - distance)
        relative_velocities.append(velocity[i] - velocity[j])
        effective_masses.append(mass[i] * mass[j] / (mass[i] + mass[j]))
        mus.append(parameters.internal_friction)
        keys.append((0, int(i), int(j)))
        particle_i.append(i)
        particle_j.append(j)
    for body in range(nb):
        distance, normal, point = box_contacts(position, body_pose[body], body_size[body])
        for i in np.flatnonzero(distance <= parameters.contact_distance):
            arm = point[i] - body_pose[body, :3]
            surface_velocity = body_velocity[body, :3] + np.cross(body_velocity[body, 3:], arm)
            body_ids.append(body)
            body_indices.append(len(normals))
            arms.append(arm)
            normals.append(normal[i])
            overlaps.append(parameters.body_offset - distance[i])
            relative_velocities.append(velocity[i] - surface_velocity)
            effective_masses.append(mass[i])
            mus.append(friction[body])
            keys.append((1, int(i), body))
            particle_i.append(i)
            particle_j.append(-1)
    count = len(normals)
    normals = np.asarray(normals).reshape(-1, 3)
    overlaps, mus = np.asarray(overlaps), np.asarray(mus)
    rel = np.asarray(relative_velocities).reshape(-1, 3)
    vn = np.einsum("ij,ij->i", rel, normals)
    vt = rel - vn[:, None] * normals
    damping = 2 * parameters.damping_ratio * np.sqrt(parameters.normal_stiffness * np.asarray(effective_masses))
    active = overlaps > 0
    fn = np.where(active, parameters.normal_stiffness * overlaps + damping * np.maximum(-vn, 0), 0)
    # Unlike the cohesive Warp example, separated candidates have a zero force prior.
    tangential = -damping[:, None] * vt
    next_history = {}
    if parameters.friction_model == "history":
        stiffness_t = parameters.normal_stiffness * parameters.tangential_stiffness_ratio
        for c in np.flatnonzero(active):
            xi = history.displacement.get(keys[c], np.zeros(3))
            xi = xi - normals[c] * np.dot(xi, normals[c]) + dt * vt[c]
            tangential[c] -= stiffness_t * xi
            next_history[keys[c]] = xi
    tangent1 = np.cross(normals, np.eye(3)[np.argmin(np.abs(normals), axis=1)]) if count else np.empty((0, 3))
    tangent1 /= np.maximum(np.linalg.norm(tangent1, axis=1)[:, None], 1e-12)
    tangent2 = np.cross(normals, tangent1)
    # Clip to the same inscribed four-ray cone that the inverse solver uses.
    l1_tangent = np.abs(np.einsum("ij,ij->i", tangential, tangent1)) + np.abs(
        np.einsum("ij,ij->i", tangential, tangent2)
    )
    tangential *= np.minimum(1, mus * fn / np.maximum(l1_tangent, 1e-30))[:, None]
    tangential[~active] = 0
    if parameters.friction_model == "history":
        for c in np.flatnonzero(active):
            next_history[keys[c]] = -(tangential[c] + damping[c] * vt[c]) / stiffness_t
    history.displacement = next_history
    target = normals * fn[:, None] + tangential
    rays = normals[:, None, :] + mus[:, None, None] * np.stack((tangent1, -tangent1, tangent2, -tangent2), axis=1)
    # B maps four ray coefficients per contact to one physical 3D contact force.
    columns = np.broadcast_to(np.arange(4 * count).reshape(count, 4, 1), (count, 4, 3)).ravel()
    rows = np.broadcast_to(3 * np.arange(count)[:, None, None] + np.arange(3), (count, 4, 3)).ravel()
    bmat = coo_matrix((rays.ravel(), (rows, columns)), shape=(3 * count, 4 * count)).tocsr()
    rows_i = np.broadcast_to(3 * np.asarray(particle_i, dtype=int)[:, None, None] + np.arange(3), (count, 4, 3)).ravel()
    pair_count = len(pairs)
    rows_j = np.broadcast_to(
        3 * np.asarray(particle_j[:pair_count], dtype=int)[:, None, None] + np.arange(3), (pair_count, 4, 3)
    ).ravel()
    amat = coo_matrix(
        (
            np.r_[rays.ravel(), -rays[:pair_count].ravel()],
            (np.r_[rows_i, rows_j], np.r_[columns, columns[: 12 * pair_count]]),
        ),
        shape=(3 * n, 4 * count),
    ).tocsr()
    frame = DemFrame(
        amat,
        bmat,
        target,
        mass[:, None] * (acceleration - np.asarray(gravity)),
        np.zeros((nb, 6)),
        np.asarray(body_ids, dtype=int),
        np.asarray(body_indices, dtype=int),
        np.asarray(arms).reshape(-1, 3),
        nb,
        0,
        pair_count,
        len(body_indices),
    )
    frame.direct_body_wrench = frame.body_wrench(target)
    frame.assembly_seconds = perf_counter() - start
    return frame


def fit_dem_frame(frame, *, law_weight=0.1, zero_prior=False, max_iterations=1000, tolerance=1e-5):
    """Fit particle balance and physical contact-force priors inside Coulomb cones.

    Objective = 0.5 ||A q - b||² + 0.5 lambda ||B q - f_DEM||².
    q >= 0. No force labels enter this solver. The optional zero prior is an
    ablation with identical geometry, basis, and regularization weight.
    """
    if law_weight <= 0 or not np.isfinite(law_weight) or max_iterations < 1 or tolerance <= 0:
        raise ValueError("Positive finite law weight, iteration budget and tolerance required.")
    start = perf_counter()
    a, b = frame.balance_matrix, frame.contact_matrix
    target = np.zeros(frame.contact_target.size) if zero_prior else frame.contact_target.ravel()
    rhs = frame.rhs.ravel()
    at, bt = a.T.tocsr(), b.T.tocsr()
    gradient_scale = max(1.0, np.max(np.abs(at @ rhs + law_weight * (bt @ target)), initial=0))

    def objective(q):
        residual = a @ q - rhs
        prior_error = b @ q - target
        value = 0.5 * (residual @ residual + law_weight * (prior_error @ prior_error))
        gradient = at @ residual + law_weight * (bt @ prior_error)
        return value, gradient

    if a.shape[1]:
        fit = minimize(
            objective,
            np.zeros(a.shape[1]),
            jac=True,
            method="L-BFGS-B",
            bounds=[(0, None)] * a.shape[1],
            options={"maxiter": max_iterations, "ftol": 1e-13, "gtol": tolerance * gradient_scale, "maxcor": 10},
        )
        q = fit.x
        _, grad = objective(q)
        projected = np.where(q > 1e-10, grad, np.minimum(grad, 0))
        kkt = float(np.max(np.abs(projected), initial=0) / gradient_scale)
        iterations, message = int(fit.nit), str(fit.message)
        converged = kkt <= 10 * tolerance
    else:
        q = np.empty(0)
        kkt, iterations, message, converged = 0.0, 0, "No candidate contacts", True
    forces = (b @ q).reshape(-1, 3)
    residual = (a @ q - rhs).reshape(-1, 3)
    return {
        "body_wrench": frame.body_wrench(forces),
        "relative_residual": float(np.linalg.norm(residual) / max(np.linalg.norm(rhs), 1e-12)),
        "law_rmse_N": float(np.sqrt(np.mean((forces.ravel() - target) ** 2))) if target.size else 0.0,
        "converged": bool(converged),
        "kkt_relative": kkt,
        "iterations": iterations,
        "message": message,
        "solve_seconds": perf_counter() - start,
        "assembly_seconds": frame.assembly_seconds,
        "internal_edges": frame.internal_edges,
        "contacts": frame.contacts,
    }
