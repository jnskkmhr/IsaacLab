# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Matched-frame particle and material-field metrics from the benchmark specification."""

import numpy as np
from scipy.ndimage import gaussian_filter


def particle_rmse(reference, predicted, mask=None):
    """RMS Euclidean error per corresponding particle, in millimetres."""
    if mask is not None:
        reference, predicted = reference[mask], predicted[mask]
    return float(np.sqrt(np.mean(np.sum((predicted - reference) ** 2, axis=-1))) * 1000) if len(reference) else None


def body_distance(points, state, shape, size):
    """Signed distance from particle centres to the interacting body's surface."""
    relative = points - state[:3]
    q, w = -state[3:6], state[6]
    local = relative + 2 * np.cross(q, np.cross(q, relative) + w * relative)
    half = np.asarray(size) / 2
    if shape == "sphere":
        return np.linalg.norm(local, axis=1) - half[0]
    if shape == "cylinder":
        distance = np.stack((np.linalg.norm(local[:, :2], axis=1) - half[0], np.abs(local[:, 2]) - half[2]), axis=1)
    elif shape == "box":
        distance = np.abs(local) - half
    else:
        raise ValueError(shape)
    return np.linalg.norm(np.maximum(distance, 0), axis=1) + np.minimum(distance.max(axis=1), 0)


class CommonGrid:
    """CIC mass and momentum deposition on a shared, fixed world-frame grid."""

    def __init__(self, lower, upper, spacing):
        self.spacing = float(spacing)
        self.lower = np.floor(np.asarray(lower) / spacing) * spacing
        self.shape = tuple((np.ceil((np.asarray(upper) - self.lower) / spacing).astype(int) + 1).tolist())
        if np.prod(self.shape) > 20_000_000:
            raise ValueError("Evaluation grid exceeds 20 million nodes; report rollout failure, not clipped mass")

    def deposit(self, position, mass, velocity=None):
        coords = (position - self.lower) / self.spacing
        base = np.floor(coords).astype(int)
        fraction = coords - base
        if np.any(base < 0) or np.any(base + 1 >= self.shape):
            raise ValueError("Particles outside the common evaluation grid; expand shared bounds")
        shape = self.shape
        result = np.zeros(shape, dtype=np.float64)
        momentum = np.zeros((*shape, 3), dtype=np.float64) if velocity is not None else None
        for a in (0, 1):
            for b in (0, 1):
                for c in (0, 1):
                    corner = np.array([a, b, c])
                    weights = np.prod(np.where(corner, fraction, 1 - fraction), axis=1) * mass
                    index = tuple((base + corner).T)
                    np.add.at(result, index, weights)
                    if momentum is not None:
                        np.add.at(momentum, index, weights[:, None] * velocity)
        if not np.isclose(result.sum(), np.sum(mass), rtol=1e-10, atol=1e-9):
            raise RuntimeError("Mass deposition did not conserve physical particle mass")
        return result, momentum

    def fields(self, reference, predicted, mass, reference_velocity, predicted_velocity):
        """Mass discrepancy and reference-mass-weighted velocity error with coverage."""
        a, pa = self.deposit(reference, mass, reference_velocity)
        b, pb = self.deposit(predicted, mass, predicted_velocity)
        threshold = 0.01 * float(np.min(mass))
        common = (a > threshold) & (b > threshold)
        covered = float(a[common].sum())
        velocity_rmse = None
        if covered > 0:
            difference = pb[common] / b[common, None] - pa[common] / a[common, None]
            velocity_rmse = float(np.sqrt(np.sum(a[common] * np.sum(difference**2, axis=1)) / covered))
        return (
            {
                "mass_discrepancy": float(np.abs(b - a).sum() / mass.sum()),
                "velocity_rmse_m_s": velocity_rmse,
                "velocity_reference_mass_coverage": covered / float(mass.sum()),
                "reference_mass_kg": float(a.sum()),
                "predicted_mass_kg": float(b.sum()),
                "reference_ke_j": float(np.sum(mass * np.sum(reference_velocity**2, axis=1)) / 2),
                "predicted_ke_j": float(np.sum(mass * np.sum(predicted_velocity**2, axis=1)) / 2),
            },
            a,
            b,
        )

    def surface(self, mass_grid, *, kernel_sigma=0.032, density_threshold=800.0):
        """Reconstruct the half-density isosurface with normalized Gaussian smoothing."""
        import trimesh
        from skimage.measure import marching_cubes

        density = gaussian_filter(
            mass_grid / self.spacing**3, sigma=kernel_sigma / self.spacing, mode="constant", truncate=4.0
        )
        if not density.min() < density_threshold < density.max():
            raise ValueError("Requested density isosurface does not exist")
        vertices, faces, _, _ = marching_cubes(density, level=density_threshold, spacing=(self.spacing,) * 3)
        return trimesh.Trimesh(vertices=vertices + self.lower, faces=faces, process=False)


def sample_surface(mesh, count, rng):
    """Sample mesh triangles by physical area, then uniform barycentric positions."""
    areas = np.asarray(mesh.area_faces)
    if not np.isfinite(areas).all() or areas.sum() <= 0:
        raise ValueError("Invalid surface mesh area")
    triangles = mesh.triangles[rng.choice(len(areas), size=count, p=areas / areas.sum())]
    a = np.sqrt(rng.random(count))
    b = rng.random(count)
    return (
        (1 - a)[:, None] * triangles[:, 0]
        + (a * (1 - b))[:, None] * triangles[:, 1]
        + (a * b)[:, None] * triangles[:, 2]
    )


def surface_distances(reference_mesh, predicted_mesh, *, count=4096, seed=42):
    """Equal-sized samples in each direction; distances are to the opposite mesh."""
    import trimesh

    rng = np.random.default_rng(seed)
    a = sample_surface(reference_mesh, count, rng)
    b = sample_surface(predicted_mesh, count, rng)
    da = trimesh.proximity.closest_point(predicted_mesh, a)[1]
    db = trimesh.proximity.closest_point(reference_mesh, b)[1]
    values = np.concatenate((da, db)) * 1000
    return {"surface_mean_mm": float(np.mean(values)), "surface_p95_mm": float(np.percentile(values, 95))}


def force_errors(reference, predicted):
    """World-frame component RMSE, signed bias, and vector RMSE in newtons."""
    error = predicted - reference
    return {
        "component_rmse_n": np.sqrt(np.mean(error**2, axis=0)).tolist(),
        "component_bias_n": np.mean(error, axis=0).tolist(),
        "vector_rmse_n": float(np.sqrt(np.mean(np.sum(error**2, axis=-1)))),
    }
