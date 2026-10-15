# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Infer a dynamic body's reaction from predicted material velocities and contact.

This normal-contact approximation advances a free rigid body, then minimizes
its kinetic velocity correction subject to unilateral particle/body constraints.
Soft contact slack represents an effective material contact mass. No material
stress solve, force labels, or future body acceleration enters inference.
The material velocity is prescribed by the motion predictor during this solve;
particle momentum consistency therefore requires a separate evaluation.
"""

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import osqp
from numba import njit
from reconstruction import box_contacts
from scipy.sparse import block_diag, bmat, csc_matrix, diags, eye, vstack
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


@dataclass
class BodyContactResult:
    body_wrench: np.ndarray
    body_velocity: np.ndarray
    particle_impulse: np.ndarray
    converged: bool
    status: str
    solve_seconds: float
    contacts: int
    iterations: int = 0
    impulse_residual: float = 0.0


@njit(cache=True)
def _fit_velocity_neighborhoods(position, velocity, offsets, neighbors, start, radius, regularization):
    """Accumulate one neighborhood at a time to bound temporary storage."""
    fitted = np.empty((len(offsets) - 1, 3, 3))
    for local in range(len(fitted)):
        i = start + local
        lhs = np.zeros((3, 3))
        rhs = np.zeros((3, 3))
        for index in range(offsets[local], offsets[local + 1]):
            j = neighbors[index]
            dx = position[j] - position[i]
            dv = velocity[j] - velocity[i]
            q = np.sqrt(np.dot(dx, dx)) / radius
            weight = (1 - q) ** 4 * (1 + 4 * q)
            for row in range(3):
                for col in range(3):
                    lhs[row, col] += weight * dx[row] * dx[col]
                    rhs[row, col] += weight * dx[row] * dv[col]
        ridge = regularization * np.trace(lhs) / 3 + 1e-12
        for axis in range(3):
            lhs[axis, axis] += ridge
        fitted[local] = np.linalg.solve(lhs, rhs)
    return fitted


def estimate_velocity_gradient(position, velocity, radius=0.065, regularization=0.01, device=None):
    """Weighted affine velocity fit at every particle, with bounded temporary memory.

    All neighbors within the radius enter the fit. Chunking changes storage only;
    it does not select an active material region or remove distant particles.
    The returned matrix maps row-vector displacements to velocity differences.
    """
    if device is not None:
        from velocity_gradient_warp import estimate_velocity_gradient_warp

        return estimate_velocity_gradient_warp(position, velocity, radius, regularization, device)
    tree = cKDTree(position)
    gradient = np.empty((len(position), 3, 3))
    for start in range(0, len(position), 2048):
        stop = min(start + 2048, len(position))
        groups = tree.query_ball_point(position[start:stop], radius, return_sorted=True)
        offsets = np.r_[0, np.cumsum([len(group) for group in groups])]
        neighbors = np.concatenate(groups).astype(np.int64)
        gradient[start:stop] = _fit_velocity_neighborhoods(
            position, velocity, offsets, neighbors, start, radius, regularization
        )
    return gradient


def reconstruct_body_contact(
    particle_position,
    particle_velocity,
    particle_mass,
    body_pose,
    body_velocity,
    body_mass,
    body_inertia,
    body_size,
    *,
    dt,
    gravity=(0, 0, -9.81),
    contact_offset=0.02,
    effective_mass_scale=10.0,
    max_iterations=10000,
    friction=None,
    extrapolate_velocity=False,
    velocity_gradient_device=None,
    stabilization=0.0,
    previous_particle_impulse=None,
    history_weight=0.0,
):
    """Return material-on-body wrench and a predicted body velocity (linear, angular).

    body_pose/body_velocity are the input rigid state, before the unknown contact
    impulse. Material velocities are the predicted next particle velocities.
    Contact offset is a geometric approximation for the sampled material surface.
    Positive gaps allow approach at gap/dt. Optional Baumgarte stabilization
    adds a separating-velocity target proportional to penetration/dt.
    With friction=None use the frictionless QP; otherwise use projected contact
    iterations with the supplied Coulomb coefficient. Body inertia is about COM in local axes;
    the box origin must coincide with COM, as in the collection scene.
    With history, compliance acts on the change from a transported previous
    contact impulse. Callers must advance that history at every timestep and
    reset it at episode boundaries; separated contacts immediately lose history.
    """
    if not 0 <= history_weight <= 1:
        raise ValueError("History weight must lie between zero and one.")
    start = perf_counter()
    x, v, pm, pose, velocity, inertia, size = [
        np.asarray(a, dtype=float)
        for a in (
            particle_position,
            particle_velocity,
            particle_mass,
            body_pose,
            body_velocity,
            body_inertia,
            body_size,
        )
    ]
    if dt <= 0 or body_mass <= 0 or effective_mass_scale <= 0 or contact_offset < 0 or stabilization < 0:
        raise ValueError("Invalid timestep, body mass, effective contact mass, or offset.")
    if not all(np.isfinite(a).all() for a in (x, v, pm, pose, velocity, inertia, size)) or np.any(pm <= 0):
        raise ValueError("Inputs must be finite and particle masses positive.")
    rotation = Rotation.from_quat(pose[3:]).as_matrix()
    world_inertia = rotation @ inertia @ rotation.T
    mass_matrix = block_diag((body_mass * eye(3), csc_matrix(world_inertia)), format="csc")
    free_velocity = velocity.copy()
    free_velocity[:3] += dt * np.asarray(gravity)
    free_velocity[3:] -= dt * np.linalg.solve(world_inertia, np.cross(velocity[3:], world_inertia @ velocity[3:]))
    distance, normals, points = box_contacts(x, pose, size)
    if extrapolate_velocity:
        gradient = estimate_velocity_gradient(x, v, device=velocity_gradient_device)
        v = v + np.einsum("ci,cij->cj", points - x, gradient)
    gap = distance - contact_offset
    gap_speed = (np.maximum(gap, 0) + stabilization * np.minimum(gap, 0)) / dt
    # Include any sample that could reach the surface during this interval.
    surface_velocity = free_velocity[:3] + np.cross(free_velocity[3:], points - pose[:3])
    approach = np.einsum("ij,ij->i", normals, surface_velocity - v)
    ids = np.flatnonzero(gap <= np.maximum(approach, 0) * dt)
    particle_impulse = np.zeros_like(x)
    if not len(ids):
        return BodyContactResult(
            np.zeros(6), free_velocity, particle_impulse, True, "no contacts", perf_counter() - start, 0
        )
    normal = normals[ids]
    arm = points[ids] - pose[:3]
    if history_weight and previous_particle_impulse is not None:
        previous = np.asarray(previous_particle_impulse, dtype=float)
        if previous.shape != x.shape or not np.isfinite(previous).all():
            raise ValueError("Previous particle impulses must be finite with shape (N, 3).")
        # Linearize the effective material response about its previous contact
        # impulse. This is an explicit history approximation, not an MPM stress law.
        # Transport the prior into the current unilateral friction cone.
        body_normal = -normal
        prior = -previous[ids]
        normal_magnitude = np.maximum(np.einsum("ij,ij->i", prior, body_normal), 0)
        tangent_prior = prior - np.einsum("ij,ij->i", prior, body_normal)[:, None] * body_normal
        if friction is None:
            tangent_prior[:] = 0
        else:
            tangent_length = np.linalg.norm(tangent_prior, axis=1)
            tangent_prior *= np.minimum(1, friction * normal_magnitude / np.maximum(tangent_length, 1e-30))[:, None]
        prior = normal_magnitude[:, None] * body_normal + tangent_prior
        v = v.copy()
        v[ids] += history_weight * prior / (pm[ids, None] * effective_mass_scale)
    if friction is not None:
        from contact_iterations import _project_contacts

        if not np.isfinite(friction) or friction < 0:
            raise ValueError("Friction must be finite and nonnegative.")
        # Coordinates of impulse ON THE BODY: normal points from material toward body.
        body_normal = -normal
        tangent = np.cross(body_normal, np.eye(3)[np.argmin(np.abs(body_normal), axis=1)])
        tangent /= np.linalg.norm(tangent, axis=1)[:, None]
        frames = np.stack((body_normal, tangent, np.cross(body_normal, tangent)), axis=1)
        jacobians = np.concatenate((frames, np.cross(arm[:, None, :], frames)), axis=2)
        material_velocity = np.einsum("cij,cj->ci", frames, v[ids])
        predicted, impulses, iterations, residual = _project_contacts(
            free_velocity,
            jacobians,
            material_velocity,
            np.linalg.inv(mass_matrix.toarray()),
            pm[ids] * effective_mass_scale,
            gap_speed[ids],
            float(friction),
            max_iterations,
            1e-9,
        )
        particle_impulse[ids] = -np.einsum("cji,cj->ci", frames, impulses)
        return BodyContactResult(
            (mass_matrix @ (predicted - free_velocity)) / dt,
            predicted,
            particle_impulse,
            residual < 1e-9,
            "converged" if residual < 1e-9 else "iteration limit",
            perf_counter() - start,
            len(ids),
            iterations,
            float(residual),
        )
    jacobian = np.c_[normal, np.cross(arm, normal)]
    target = np.einsum("ij,ij->i", normal, v[ids]) + gap_speed[ids]
    count = len(ids)
    # Objective has physical kinetic-energy units: body velocity correction plus
    # squared normal slip weighted by an effective material mass per contact.
    constraints = vstack(
        (
            bmat([[csc_matrix(jacobian), -eye(count)]], format="csc"),
            bmat([[csc_matrix((count, 6)), eye(count)]], format="csc"),
        ),
        format="csc",
    )
    problem = osqp.OSQP()
    problem.setup(
        P=block_diag((mass_matrix, diags(pm[ids] * effective_mass_scale)), format="csc"),
        q=np.r_[-mass_matrix @ free_velocity, np.zeros(count)],
        A=constraints,
        l=np.r_[np.full(count, -np.inf), np.zeros(count)],
        u=np.r_[target, np.full(count, np.inf)],
        eps_abs=1e-8,
        eps_rel=1e-8,
        max_iter=max_iterations,
        polishing=True,
        verbose=False,
    )
    fit = problem.solve(raise_error=False)
    if fit.x is None or not np.isfinite(fit.x).all():
        raise RuntimeError(f"Body-contact QP failed: {fit.info.status}")
    impulse = mass_matrix @ (fit.x[:6] - free_velocity)
    particle_impulse[ids] = fit.y[:count, None] * normal
    return BodyContactResult(
        impulse / dt,
        fit.x[:6],
        particle_impulse,
        fit.info.status == "solved",
        fit.info.status,
        perf_counter() - start,
        count,
    )
