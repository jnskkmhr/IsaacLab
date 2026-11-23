# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Measure how fast implicit MPM produces reference frames for the dataset's sand bed.

Several independent beds (each in its own tray) are simulated by one solver, so the result shows how the frame rate
scales with the number of parallel beds on one GPU. A frame is one 200 Hz dataset step.
"""

import argparse
import json
import time
from pathlib import Path

import newton
import numpy as np
import warp as wp
from newton.solvers import SolverImplicitMPM

from ..core.terrain import DENSITY, SPACING, VOXEL_SIZE, sand_cfg
from .episodes import DT, TERRAINS, tray_boundary_boxes
from .presettle import MATERIAL

PITCH = (3.0, 2.0)
"""Centre-to-centre distance of neighbouring beds along x and y [m]."""


def build(beds: int, terrain: str, device: str, collider_basis: str | None = None):
    """One model holding ``beds`` copies of the bed and its tray on a grid.

    ``collider_basis`` overrides the solver default (the sand-only collector uses the default, the coupled Isaac Lab
    scene uses ``pic27``).
    """
    size = TERRAINS["flat"]["bed_size_m"]
    builder = newton.ModelBuilder(gravity=wp.vec3(0.0, 0.0, -9.81))
    SolverImplicitMPM.register_custom_attributes(builder)
    shape_cfg = newton.ModelBuilder.ShapeConfig(density=0.0, mu=1.0, margin=0.0)
    columns = int(np.ceil(np.sqrt(beds)))
    points = []
    for bed in range(beds):
        offset = np.array([(bed % columns) * PITCH[0], (bed // columns) * PITCH[1], 0.0], dtype=np.float32)
        for box in tray_boundary_boxes(size):
            half = np.asarray(box["size_m"], dtype=np.float32) / 2.0
            position = np.asarray(box["pose"][:3], dtype=np.float32) + offset
            builder.add_shape_box(
                body=-1,
                xform=wp.transform(wp.vec3(*position.tolist()), wp.quat(*box["pose"][3:])),
                hx=float(half[0]),
                hy=float(half[1]),
                hz=float(half[2]),
                cfg=shape_cfg,
            )
        points.append(np.asarray(sand_cfg(terrain, size, seed=42 + bed).spawn.positions, dtype=np.float32) + offset)
    points = np.concatenate(points)
    n = len(points)
    builder.add_particles(
        pos=points.tolist(),
        vel=np.zeros_like(points).tolist(),
        mass=[float(DENSITY * SPACING**3)] * n,
        radius=[SPACING / 2.0] * n,
    )
    model = builder.finalize(device=device)
    model.particle_mu = 0.48
    for name in ("young_modulus", "poisson_ratio", "yield_pressure", "yield_stress", "tensile_yield_ratio"):
        getattr(model.mpm, name).fill_(MATERIAL[name])
    model.mpm.hardening.fill_(0.0)
    model.mpm.dilatancy.fill_(0.0)
    model.mpm.friction.fill_(0.48)
    config = SolverImplicitMPM.Config()
    config.voxel_size = VOXEL_SIZE
    config.grid_type = "sparse"
    config.max_active_cell_count = max(1 << 18, beds << 16)
    config.strain_basis = "P0"
    config.velocity_basis = "Q1"
    config.transfer_scheme = "apic"
    config.max_iterations = 25
    config.tolerance = 1.0e-4
    config.critical_fraction = 0.0
    config.collider_velocity_mode = "forward"
    if collider_basis is not None:
        config.collider_basis = collider_basis
    return model, SolverImplicitMPM(model, config), n


def measure(beds: int, substeps: int, terrain: str, frames: int, warmup: int, device: str) -> dict:
    model, solver, n = build(beds, terrain, device)
    s0, s1 = model.state(), model.state()

    def advance(count):
        nonlocal s0, s1
        for _ in range(count):
            for _ in range(substeps):
                solver.step(s0, s1, None, None, DT / substeps)
                s0, s1 = s1, s0

    advance(warmup)
    wp.synchronize_device(device)
    start = time.perf_counter()
    advance(frames)
    wp.synchronize_device(device)
    seconds = time.perf_counter() - start
    start = time.perf_counter()
    for _ in range(20):
        positions = s0.particle_q.numpy()
    readback = (time.perf_counter() - start) / 20
    gpu = wp.get_device(device)
    return {
        "beds": beds,
        "substeps": substeps,
        "terrain": terrain,
        "particles": n,
        "frames_per_s": round(frames / seconds, 2),
        "bed_frames_per_s": round(beds * frames / seconds, 2),
        "ms_per_frame": round(1e3 * seconds / frames, 2),
        "readback_ms": round(1e3 * readback, 3),
        "gpu_used_gb": round((gpu.total_memory - gpu.free_memory) / 1e9, 2),
        "finite": bool(np.isfinite(positions).all()),
        "top_z_m": round(float(np.percentile(positions[:, 2], 99.5)), 4),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--beds", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--substeps", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--terrain", default="flat")
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=40)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    device = "cuda:0"
    wp.init()
    wp.set_device(device)
    results = []
    for substeps in args.substeps:
        for beds in args.beds:
            try:
                row = measure(beds, substeps, args.terrain, args.frames, args.warmup, device)
            except Exception as error:  # report a failed size (for example out of memory) and continue
                row = {"beds": beds, "substeps": substeps, "error": repr(error)[:300]}
            results.append(row)
            print(json.dumps(row), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"device": str(wp.get_device(device).name), "results": results}, indent=1))


if __name__ == "__main__":
    main()
