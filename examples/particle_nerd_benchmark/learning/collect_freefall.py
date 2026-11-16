# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Collect 400 native 200 Hz 6-second free-fall sand block/column MPM trajectories across GPUs."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import time
from pathlib import Path

import h5py
import newton
import numpy as np
import warp as wp
from newton.solvers import SolverImplicitMPM

from ..core.terrain import DENSITY, PARTICLE_JITTER, SPACING, VOXEL_SIZE
from .protocol import DT, SIZES, freefall_episodes, freefall_toss_episodes, release_velocity, rest_episodes

TRAY_BOUNDARY_BOXES = [
    {"name": "floor", "size_m": [2.2, 1.4, 0.2], "pose": [0.0, 0.0, -0.4, 0, 0, 0, 1]},
    {"name": "wall_x0", "size_m": [0.1, 1.4, 0.5], "pose": [-1.05, 0.0, -0.05, 0, 0, 0, 1]},
    {"name": "wall_x1", "size_m": [0.1, 1.4, 0.5], "pose": [1.05, 0.0, -0.05, 0, 0, 0, 1]},
    {"name": "wall_y0", "size_m": [2.0, 0.1, 0.5], "pose": [0.0, -0.65, -0.05, 0, 0, 0, 1]},
    {"name": "wall_y1", "size_m": [2.0, 0.1, 0.5], "pose": [0.0, 0.65, -0.05, 0, 0, 0, 1]},
]
"""The finite tray used by the object benchmark."""


def bed_positions(seed: int) -> np.ndarray:
    """Jittered lattice of the 2.0 x 1.2 x 0.30 m sand bed, identical to the benchmark scene (20,646 particles)."""
    length, width, depth = 2.0, 1.2, 0.30
    axes = [
        np.arange(-length / 2 + SPACING / 2, length / 2, SPACING),
        np.arange(-width / 2 + SPACING / 2, width / 2, SPACING),
        np.arange(-depth + SPACING / 2, 0.2, SPACING),
    ]
    points = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    points = points[points[:, 2] < 0.0]
    rng = np.random.default_rng(seed)
    lower = np.maximum(points - PARTICLE_JITTER, (-length / 2, -width / 2, -depth))
    upper = np.minimum(points + PARTICLE_JITTER, (length / 2, width / 2, 0.0))
    points[:, :2] = rng.uniform(lower[:, :2], upper[:, :2])
    points[:, 2] = rng.uniform(lower[:, 2], upper[:, 2])
    return points.astype(np.float32)


def build_manifest(records: list[dict], source_root: Path, tray: bool = False) -> dict:
    """Create a standard particle-nerd-object-trajectories-v1 manifest (open flat ground, or the finite tray)."""
    depth = 0.30
    boundary_boxes = [
        {"name": "floor", "size_m": [4.0, 4.0, 0.20], "pose": [0.0, 0.0, -depth - 0.10, 0, 0, 0, 1]},
        {
            "name": "wall_x0",
            "size_m": [0.10, 4.0, 0.50],
            "pose": [-10.0, 0.0, 0.0, 0, 0, 0, 1],
        },
        {
            "name": "wall_x1",
            "size_m": [0.10, 4.0, 0.50],
            "pose": [10.0, 0.0, 0.0, 0, 0, 0, 1],
        },
        {
            "name": "wall_y0",
            "size_m": [4.0, 0.10, 0.50],
            "pose": [0.0, -10.0, 0.0, 0, 0, 0, 1],
        },
        {
            "name": "wall_y1",
            "size_m": [4.0, 0.10, 0.50],
            "pose": [0.0, 10.0, 0.0, 0, 0, 0, 1],
        },
    ]
    return {
        "schema": "particle-nerd-object-trajectories-v1",
        "episodes": records,
        "particle_fields": ["particle_position"],
        "rigid_fields": ["body_state", "collider_state"],
        "frame": "World axes, origin at initial tray centre; quaternions xyzw",
        "body_state_timing": "Far-field inactive placeholder at z = +10 m",
        "collider_state_timing": "At entry to the MPM step",
        "state_columns": "xyz, quaternion xyzw, linear xyz, angular xyz",
        "dt_s": DT,
        "evaluation_reference": None,
        "force_definition": None,
        "voxel_size_m": VOXEL_SIZE,
        "particle_spacing_m": SPACING,
        "jitter_max_m": PARTICLE_JITTER,
        "body_size_m": SIZES["box"],
        "configured_body_mass_kg": 2.0,
        "material": {
            "density": DENSITY,
            "young_modulus": 1.0e15,
            "poisson_ratio": 0.3,
            "friction": 0.48,
            "yield_pressure": 1.0e15,
            "tensile_yield_ratio": 0.0,
            "yield_stress": 0.0,
            "hardening": 0.0,
            "dilatancy": 0.0,
            "viscosity": 0.0,
        },
        "source_hashes": {
            str(p.relative_to(source_root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(source_root.rglob("*.py"))
        },
        "versions": {n: importlib.metadata.version(n) for n in ("newton", "warp-lang", "torch", "numpy")},
        "boundary_boxes": TRAY_BOUNDARY_BOXES if tray else boundary_boxes,
    }


def simulate_one(record: dict, manifest: dict, manifest_sha: str, out_path: Path, device: str):
    """Simulate one 6-second free-fall sand block/column episode with SolverImplicitMPM and write HDF5."""
    if out_path.is_file():
        try:
            with h5py.File(out_path, "r") as f:
                if f.attrs.get("complete") and f.attrs.get("manifest_sha256") == manifest_sha:
                    return
        except Exception:
            pass

    dt = float(record["dt_s"])
    steps = round(float(record["duration_s"]) / dt)
    substeps = 2
    sub_dt = dt / substeps
    radius = SPACING / 2.0
    p_mass = float(DENSITY * (SPACING**3))
    z_floor = -0.30

    wx, wy, hz = record["block_size_m"]
    nx = max(6, round(wx / SPACING))
    ny = max(6, round(wy / SPACING))
    nz = max(8, round(hz / SPACING))
    cx = -0.15 * record["direction"] if record["motion"] in ("slide", "throw", "tumble") else 0.0
    cy = float(record["y"])
    z_bot = z_floor + radius + float(record["release_height"])

    xs = (np.arange(nx, dtype=np.float32) - (nx - 1) * 0.5) * SPACING + cx
    ys = (np.arange(ny, dtype=np.float32) - (ny - 1) * 0.5) * SPACING + cy
    zs = np.arange(nz, dtype=np.float32) * SPACING + z_bot
    pts = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1).reshape(-1, 3)

    if record.get("column_shape") == "cylinder":
        rad = 0.5 * min(wx, wy)
        keep = (pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2 <= rad**2
        pts = pts[keep]
    elif record.get("column_shape") == "sphere":
        rad = 0.5 * min(wx, wy)
        zc = z_bot + 0.5 * (nz - 1) * SPACING
        keep = (pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2 + (pts[:, 2] - zc) ** 2 <= rad**2
        pts = pts[keep]

    rng = np.random.default_rng(int(record["particle_seed"]))
    pts = pts + rng.uniform(-PARTICLE_JITTER, PARTICLE_JITTER, size=pts.shape).astype(np.float32)
    pts[:, 2] = np.maximum(pts[:, 2], z_bot)
    if record.get("column_shape") == "bed":
        pts = bed_positions(int(record["particle_seed"]))  # resting tray bed: replaces the released block

    n_pts = len(pts)
    rel_v = release_velocity(record)
    lin_v = rel_v[:3]
    ang_v = rel_v[3:6]
    com_0 = pts.mean(axis=0, keepdims=True)
    vel_0 = lin_v[None, :] + np.cross(ang_v[None, :], pts - com_0)
    vel_0 = vel_0.astype(np.float32)

    builder = newton.ModelBuilder(gravity=wp.vec3(0.0, 0.0, -9.81))
    SolverImplicitMPM.register_custom_attributes(builder)
    shape_cfg = newton.ModelBuilder.ShapeConfig(density=0.0, mu=1.0, margin=0.0)
    for wall in manifest["boundary_boxes"]:
        half = np.asarray(wall["size_m"], dtype=np.float32) / 2.0
        pose = wall["pose"]
        builder.add_shape_box(
            body=-1,
            xform=wp.transform(wp.vec3(*pose[:3]), wp.quat(*pose[3:])),
            hx=float(half[0]),
            hy=float(half[1]),
            hz=float(half[2]),
            cfg=shape_cfg,
            label=f"/World/envs/env_0/{wall['name'].title().replace('_', '')}/Shape",
        )
    builder.add_particles(
        pos=pts.tolist(),
        vel=vel_0.tolist(),
        mass=[p_mass] * n_pts,
        radius=[radius] * n_pts,
    )
    model = builder.finalize(device=device)
    model.particle_mu = 0.48
    model.mpm.young_modulus.fill_(1.0e15)
    model.mpm.poisson_ratio.fill_(0.3)
    model.mpm.yield_pressure.fill_(1.0e15)
    model.mpm.yield_stress.fill_(0.0)
    model.mpm.tensile_yield_ratio.fill_(0.0)
    model.mpm.hardening.fill_(0.0)
    model.mpm.dilatancy.fill_(0.0)
    model.mpm.friction.fill_(0.48)

    mpm_cfg = SolverImplicitMPM.Config()
    mpm_cfg.voxel_size = VOXEL_SIZE
    mpm_cfg.grid_type = "sparse"
    mpm_cfg.max_active_cell_count = 1 << 18
    mpm_cfg.strain_basis = "P0"
    mpm_cfg.velocity_basis = "Q1"
    mpm_cfg.transfer_scheme = "apic"
    mpm_cfg.max_iterations = 25
    mpm_cfg.tolerance = 1.0e-4
    mpm_cfg.critical_fraction = 0.0
    mpm_cfg.collider_velocity_mode = "forward"

    solver = SolverImplicitMPM(model, mpm_cfg)
    s0, s1 = model.state(), model.state()

    positions = np.empty((steps + 1, n_pts, 3), dtype=np.float32)
    positions[0] = s0.particle_q.numpy().copy()
    for step in range(steps):
        for _ in range(substeps):
            solver.step(s0, s1, None, None, sub_dt)
            s0, s1 = s1, s0
        positions[step + 1] = s0.particle_q.numpy().copy()

    far_body = np.zeros((steps + 1, 13), dtype=np.float32)
    far_body[:, 2] = 10.0
    far_body[:, 6] = 1.0
    far_collider = far_body[:-1].copy()

    labels = [
        "/World/envs/env_0/Object/Shape",
        "/World/envs/env_0/Floor/Shape",
        "/World/envs/env_0/WallX0/Shape",
        "/World/envs/env_0/WallX1/Shape",
        "/World/envs/env_0/WallY0/Shape",
        "/World/envs/env_0/WallY1/Shape",
    ]
    tmp_path = out_path.with_suffix(".tmp.h5")
    with h5py.File(tmp_path, "w") as f:
        f.create_dataset(
            "particle_position",
            data=positions,
            chunks=(min(16, steps + 1), n_pts, 3),
            compression="gzip",
            compression_opts=1,
            shuffle=True,
        )
        f.create_dataset("particle_mass", data=np.full(n_pts, p_mass, dtype=np.float32))
        f.create_dataset("particle_radius", data=np.full(n_pts, radius, dtype=np.float32))
        f.create_dataset("body_mass", data=np.array([2.0], dtype=np.float32))
        f.create_dataset("body_inertia", data=(np.eye(3, dtype=np.float32) * 0.01)[None])
        f.create_dataset("body_com", data=np.zeros((1, 3), dtype=np.float32))
        f.create_dataset("body_state", data=far_body)
        f.create_dataset("collider_state", data=far_collider)
        f.create_dataset("shape_labels", data=np.asarray(labels, dtype=h5py.string_dtype("utf-8")))
        f.create_dataset("shape_material_mu", data=np.ones(6, dtype=np.float32))
        f.create_dataset("shape_body", data=np.array([0, -1, -1, -1, -1, -1], dtype=np.int32))
        f.attrs["complete"] = True
        f.attrs["manifest_sha256"] = manifest_sha
    tmp_path.replace(out_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=int(os.environ.get("RANK", 0)))
    parser.add_argument("--world-size", type=int, default=int(os.environ.get("WORLD_SIZE", 1)))
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument(
        "--set",
        choices=("freefall", "toss", "rest"),
        default="freefall",
        help="Episode set: 400 vertical drops; drops and horizontal tosses of blocks; or 20 s resting tray beds",
    )
    parser.add_argument("--count", type=int, default=240, help="Number of episodes for --set toss")
    args = parser.parse_args()

    device = f"cuda:{int(os.environ.get('LOCAL_RANK', args.rank))}"
    wp.init()
    wp.set_device(device)

    records = (
        freefall_episodes()
        if args.set == "freefall"
        else freefall_toss_episodes(args.count)
        if args.set == "toss"
        else rest_episodes(args.count)
    )
    source_root = Path(__file__).parents[1]
    manifest = build_manifest(records, source_root, tray=args.set == "rest")
    serialized = json.dumps(manifest, indent=2, default=str)
    manifest_sha = hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "manifest.json"
    if args.rank == 0:
        manifest_path.write_text(serialized)

    global_rank = args.shard_id * args.world_size + args.rank
    global_world = args.num_shards * args.world_size
    my_records = records[global_rank::global_world]
    t0 = time.perf_counter()
    for i, rec in enumerate(my_records, 1):
        simulate_one(rec, manifest, manifest_sha, args.output / f"{rec['id']}.h5", device)
        print(
            json.dumps(
                {
                    "global_rank": global_rank,
                    "completed": f"{i}/{len(my_records)}",
                    "episode": rec["id"],
                    "motion": rec["motion"],
                    "split": rec["split"],
                    "elapsed_s": round(time.perf_counter() - t0, 1),
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
