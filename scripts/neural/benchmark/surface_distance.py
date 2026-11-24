# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Measure how far the sand surface of the Particle NeRD run is from that of the MPM run of a benchmark case.

Reads ``nerd.npz`` and ``mpm.npz`` that ``run_case.py`` wrote into the case folder. For every second saved frame
the particle masses of each run are deposited on a 0.02 m grid and smoothed with a 0.032 m Gaussian kernel; the
surface is the 800 kg/m^3 iso-surface of that density. For every upward-facing point of the Particle NeRD surface
the distance to the MPM surface is measured, with a sign: positive where the Particle NeRD surface lies outside
(above) the MPM material, negative where it lies inside (below). The script writes

- ``surface_distance.mp4``: a top view of the signed distance over time, with the object position marked;
- ``surface_distance.json``: the mean absolute distance at the end and averaged over time, and its 95th percentile
  at the end [mm].

Run from the repository root (scikit-image is not part of the environment, so it is added for this run):

    uv run --with scikit-image python scripts/neural/benchmark/surface_distance.py logs/neural/benchmark/box-push
"""

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import imageio.v2 as iio
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from scipy.ndimage import gaussian_filter, map_coordinates
from skimage.measure import marching_cubes

SPACING = 0.02
"""Grid spacing of the density field [m]."""
SIGMA = 0.032
"""Width of the Gaussian smoothing kernel [m]."""
THRESHOLD = 800.0
"""Density of the iso-surface [kg/m^3]: half of the sand density."""
PARTICLE_MASS = 1600.0 * 0.032**3
PHYSICS_DT = 0.005
BOUNDS = ([-1.1, -0.7, -0.35], [1.1, 0.7, 0.3])
"""Region of the density field [m]: the example tray with a margin."""
FLOOR_Z = -0.30


class DensityField:
    """Smoothed mass density of a particle set on a regular grid, and its iso-surface."""

    def __init__(self, lower, upper):
        self.lower = np.floor(np.asarray(lower) / SPACING) * SPACING
        self.shape = tuple((np.ceil((np.asarray(upper) - self.lower) / SPACING).astype(int) + 2).tolist())

    def density(self, q: np.ndarray) -> np.ndarray:
        """Density [kg/m^3] from particle positions [N, 3], deposited with trilinear weights and smoothed."""
        q = np.clip(q, self.lower + SPACING, self.lower + (np.array(self.shape) - 2) * SPACING)
        cell = (q - self.lower) / SPACING
        base = np.floor(cell).astype(int)
        frac = cell - base
        grid = np.zeros(self.shape)
        for a in (0, 1):
            for b in (0, 1):
                for c in (0, 1):
                    weight = np.prod(np.where(np.array([a, b, c]), frac, 1 - frac), axis=1) * PARTICLE_MASS
                    np.add.at(grid, tuple((base + [a, b, c]).T), weight)
        return gaussian_filter(grid / SPACING**3, sigma=SIGMA / SPACING, mode="constant", truncate=4.0)

    def mesh(self, density: np.ndarray) -> trimesh.Trimesh:
        """Iso-surface of the density at ``THRESHOLD``."""
        vertices, faces, _, _ = marching_cubes(density, level=THRESHOLD, spacing=(SPACING,) * 3)
        return trimesh.Trimesh(vertices=vertices + self.lower, faces=faces, process=False)


def signed_distance(reference_mesh, reference_density, field, points):
    """Distance of ``points`` to the reference surface: positive outside the reference material, negative inside."""
    distance = trimesh.proximity.closest_point(reference_mesh, points)[1]
    index = ((points - field.lower) / SPACING).T
    inside = map_coordinates(reference_density, index, order=1, mode="nearest") > THRESHOLD
    return np.where(inside, -distance, distance)


def compare(case_dir: Path, frame_step: int = 2, colour_limit: float = 0.04) -> dict:
    """Write the video and the statistics of one case; return the statistics."""
    runs = {name: np.load(case_dir / f"{name}.npz") for name in ("nerd", "mpm")}
    q_nerd, q_mpm = runs["nerd"]["q"], runs["mpm"]["q"]
    object_xyz = runs["nerd"]["pose"][:, :3]
    dt = float(runs["nerd"]["step"][1] - runs["nerd"]["step"][0]) * PHYSICS_DT
    field = DensityField(*BOUNDS)
    count = min(len(q_nerd), len(q_mpm))
    frames, series, indices = [], [], []
    started = time.time()
    for index in range(0, count, frame_step):
        density_nerd, density_mpm = field.density(q_nerd[index]), field.density(q_mpm[index])
        try:
            mesh_nerd, mesh_mpm = field.mesh(density_nerd), field.mesh(density_mpm)
        except ValueError:  # no iso-surface in this frame
            continue
        probe = ((mesh_nerd.vertices + [0, 0, 0.03] - field.lower) / SPACING).T
        open_above = map_coordinates(density_nerd, probe, order=1, mode="nearest") < THRESHOLD
        upward = (
            (np.abs(mesh_nerd.vertex_normals[:, 2]) > 0.4) & open_above & (mesh_nerd.vertices[:, 2] > FLOOR_Z + 0.04)
        )
        if upward.sum() < 10:
            continue
        points = mesh_nerd.vertices[upward]
        distance = signed_distance(mesh_mpm, density_mpm, field, points)
        series.append((np.abs(distance).mean() * 1e3, np.percentile(np.abs(distance), 95) * 1e3))
        indices.append(index)

        fig = plt.figure(figsize=(12, 7.2), dpi=80)
        top = fig.add_axes([0.06, 0.34, 0.80, 0.60])
        curve = fig.add_axes([0.06, 0.07, 0.80, 0.18])
        scatter = top.scatter(
            points[:, 0],
            points[:, 1],
            c=distance * 1e3,
            s=22,
            cmap="RdBu_r",
            vmin=-colour_limit * 1e3,
            vmax=colour_limit * 1e3,
            marker="s",
        )
        marker = object_xyz[min(index, len(object_xyz) - 1)]
        top.plot(marker[0], marker[1], "k+", ms=16, mew=3)
        top.set_xlim(-1, 1)
        top.set_ylim(-0.6, 0.6)
        top.set_aspect("equal")
        top.set_facecolor("#eeeeee")
        top.set_title(
            f"Surface distance, Particle NeRD to MPM   t = {index * dt:.2f} s", fontsize=14, fontweight="bold"
        )
        bar = fig.colorbar(scatter, ax=[top], fraction=0.035, pad=0.02)
        bar.set_label("Signed distance [mm]   (+ NeRD surface outside MPM, − inside)")
        values = np.array(series)
        times = np.array(indices) * dt
        curve.plot(times, values[:, 0], label="mean |distance|", lw=2)
        curve.plot(times, values[:, 1], label="95th percentile", lw=2, ls="--")
        curve.set_xlabel("time [s]")
        curve.set_ylabel("mm")
        curve.legend(loc="upper left", fontsize=9)
        curve.grid(alpha=0.3)
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        plt.close(fig)
    iio.mimsave(case_dir / "surface_distance.mp4", frames, fps=20, macro_block_size=1)
    values = np.array(series)
    stats = {
        "frames": len(frames),
        "mean_mm_time_mean": float(values[:, 0].mean()),
        "mean_mm_end": float(values[-1, 0]),
        "p95_mm_end": float(values[-1, 1]),
        "seconds": round(time.time() - started, 1),
    }
    (case_dir / "surface_distance.json").write_text(json.dumps(stats, indent=1))
    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("case_dir", type=Path, help="Case folder with nerd.npz and mpm.npz.")
    args = parser.parse_args(argv)
    for name in ("nerd", "mpm"):
        if not (args.case_dir / f"{name}.npz").is_file():
            parser.error(f"{args.case_dir / (name + '.npz')} is missing; run run_case.py --solver {name} first")
    stats = compare(args.case_dir)
    print(args.case_dir.name, json.dumps(stats), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
