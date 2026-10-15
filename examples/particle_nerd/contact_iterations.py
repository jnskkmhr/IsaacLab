# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compiled projected contact iterations; requires a Numba-compatible NumPy version."""

import numpy as np
from numba import njit


@njit(cache=True)
def _friction_disk_minimizer(matrix, rhs, radius):
    """Minimize 0.5*x.T*K*x - rhs.T*x inside a circular Coulomb disk.

    On sliding, (K + multiplier*I)*x = rhs. The nonnegative multiplier
    makes the tangential impulse oppose the final relative sliding velocity.
    Euclidean clipping of K^-1*rhs does not enforce this for anisotropic K.
    """
    if radius == 0.0:
        return np.zeros(2)
    unconstrained = np.linalg.solve(matrix, rhs)
    if np.sqrt(np.sum(unconstrained**2)) <= radius:
        return unconstrained
    lower = 0.0
    upper = np.sqrt(np.sum(rhs**2)) / radius
    x = np.zeros(2)
    for _ in range(45):
        multiplier = 0.5 * (lower + upper)
        a = matrix[0, 0] + multiplier
        b = matrix[0, 1]
        d = matrix[1, 1] + multiplier
        determinant = a * d - b * b
        x[0] = (d * rhs[0] - b * rhs[1]) / determinant
        x[1] = (a * rhs[1] - b * rhs[0]) / determinant
        if np.sqrt(np.sum(x**2)) > radius:
            lower = multiplier
        else:
            upper = multiplier
    return x


@njit(cache=True)
def _project_contacts(
    free_velocity,
    jacobians,
    material_velocity,
    inverse_mass,
    effective_mass,
    gap_speed,
    friction,
    max_iterations,
    tolerance,
):
    """Projected Gauss–Seidel with a nonassociated Coulomb sliding law."""
    count = len(effective_mass)
    impulses = np.zeros((count, 3))
    velocity = free_velocity.copy()
    compliance = np.empty((count, 3, 3))
    response = np.empty((count, 6, 3))
    for i in range(count):
        response[i] = inverse_mass @ jacobians[i].T
        compliance[i] = jacobians[i] @ response[i] + np.eye(3) / effective_mass[i]
    residual = np.inf
    iteration = 0
    for iteration in range(max_iterations):
        residual = 0.0
        for i in range(count):
            relative = jacobians[i] @ velocity - material_velocity[i] + impulses[i] / effective_mass[i]
            normal = max(0.0, impulses[i, 0] - (relative[0] + gap_speed[i]) / compliance[i, 0, 0])
            delta_normal = normal - impulses[i, 0]
            impulses[i, 0] = normal
            velocity += response[i, :, 0] * delta_normal
            relative = jacobians[i] @ velocity - material_velocity[i] + impulses[i] / effective_mass[i]
            tangent_matrix = compliance[i, 1:, 1:]
            rhs = tangent_matrix @ impulses[i, 1:] - relative[1:]
            tangent = _friction_disk_minimizer(tangent_matrix, rhs, friction * normal)
            delta_tangent = tangent - impulses[i, 1:]
            impulses[i, 1:] = tangent
            velocity += response[i, :, 1:] @ delta_tangent
            residual = max(residual, abs(delta_normal), np.sqrt(np.sum(delta_tangent**2)))
        if residual < tolerance:
            break
    return velocity, impulses, iteration + 1, residual
