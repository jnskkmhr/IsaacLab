# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Record the MPM reference episodes that the Particle NeRD network is trained and evaluated on.

Every episode runs in the scene of ``scene.py``: the same sand, tray and solver settings. Only the surface of the
bed (``flat``, ``slope``, ``wave``) and what happens on it change. Four categories move a rigid object
(``lateral_push``, ``vertical_push``, ``impact``, ``rest``) and two have sand only (``freefall``: a block of sand
released above the bed; ``rest_bed``: the untouched bed). The bed is simulated for 2 s before recording starts, so
the first recorded frame is a settled bed. One file is written per episode:
``<dataset>/<split>/<terrain>/<category>/NNNN.hdf5``. Finished files are skipped, so a run can be repeated.

The script works in one of two ways. Run it from the repository root.

Episodes that share one scene, in this process. Object categories need ``--shape`` and ``--size-level``, because
the scene is built for one object; a free-fall process holds one block, so pass one index:

    uv run python scripts/neural/record.py --dataset /data/particle_nerd --terrain flat \\
        --category lateral_push --shape cube --size-level 1

All episodes of a terrain: ``--select`` names groups of categories, and one process per scene runs on each GPU.
A summary of the files follows, with the check that every bed had settled:

    uv run python scripts/neural/record.py --dataset /data/particle_nerd --terrain flat --select all --gpus 2
"""

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Queue

import h5py
import numpy as np
import torch
from dataset import (
    DT,
    HOVER_S,
    OBJECT_CATEGORIES,
    PER_MOTION,
    PRODUCTION_TERRAINS,
    SAND_CATEGORIES,
    SHAPES,
    SIZE_SCALES,
    TERRAINS,
    episodes,
    lateral_pose,
    object_geometry,
    tray_boundary_boxes,
    vertical_pose,
    write_episode,
)
from metrics import summarize_files

from isaaclab.app import add_launcher_args, launch_simulation

GRAVITY = 9.81
PARKED_POSE = (0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 1.0)
"""Pose of the rigid object while it must not touch the sand (pre-settle, and all of a sand-only episode)."""
SELECTIONS = {
    "lateral": ("lateral_push",),
    "vertical_impact": ("vertical_push", "impact"),
    "rest_sand": ("rest", "freefall", "rest_bed"),
    "all": (*OBJECT_CATEGORIES, *SAND_CATEGORIES),
}
"""Groups of categories for ``--select``. Each of the first three writes roughly 30-50 GB for a full terrain."""
SAND_CHUNK = 15
"""Bare-bed episodes per process."""
COST = {"lateral_push": 45, "vertical_push": 45, "impact": 45, "rest": 70, "freefall": 45, "rest_bed": 70}
"""Rough seconds per episode, used only to start the long units first."""


def _rotate(quat_xyzw: np.ndarray, vec: np.ndarray) -> np.ndarray:
    u, w = quat_xyzw[:3], quat_xyzw[3]
    return vec + 2.0 * w * np.cross(u, vec) + 2.0 * np.cross(u, np.cross(u, vec))


def _yaw_quat(yaw: float) -> np.ndarray:
    return np.array([0.0, 0.0, np.sin(yaw / 2), np.cos(yaw / 2)], dtype=np.float32)


def block_particles(record: dict, surface_z: float, spacing: float, jitter: float) -> np.ndarray:
    """Jittered lattice of a cube, sphere or vertical cylinder of sand above the bed surface at its start position."""
    radius = spacing / 2.0
    wx, wy, hz = record["block_size_m"]
    nx, ny, nz = max(6, round(wx / spacing)), max(6, round(wy / spacing)), max(8, round(hz / spacing))
    cx, cy = record["start_xy"]
    z_bottom = surface_z + radius + float(record["release_height"])
    xs = (np.arange(nx, dtype=np.float32) - (nx - 1) * 0.5) * spacing + cx
    ys = (np.arange(ny, dtype=np.float32) - (ny - 1) * 0.5) * spacing + cy
    zs = np.arange(nz, dtype=np.float32) * spacing + z_bottom
    points = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1).reshape(-1, 3)
    half = 0.5 * min(wx, wy)
    if record["shape"] == "cylinder":
        points = points[(points[:, 0] - cx) ** 2 + (points[:, 1] - cy) ** 2 <= half**2]
    elif record["shape"] == "sphere":
        zc = z_bottom + 0.5 * (nz - 1) * spacing
        points = points[(points[:, 0] - cx) ** 2 + (points[:, 1] - cy) ** 2 + (points[:, 2] - zc) ** 2 <= half**2]
    rng = np.random.default_rng(int(record["particle_seed"]))
    points = points + rng.uniform(-jitter, jitter, size=points.shape).astype(np.float32)
    points[:, 2] = np.maximum(points[:, 2], z_bottom)
    return points.astype(np.float32)


def select_records(
    terrain: str,
    category: str,
    per_motion: int = PER_MOTION,
    shape: str | None = None,
    size_level: int | None = None,
    variant: str | None = None,
    indices: list[int] | None = None,
) -> list[dict]:
    """Episodes of one terrain and category, narrowed to one object, bed variant or list of indices."""
    records = episodes(terrain, category, per_motion)
    if category in OBJECT_CATEGORIES:
        records = [r for r in records if r["shape"] == shape and r["size_level"] == size_level]
    if indices:
        records = [r for r in records if r["index"] in indices]
    if variant:
        records = [r for r in records if r["terrain_variant"] == variant]
    return records


def build_units(terrain: str, select: list[str], per_motion: int) -> list[dict]:
    """Split the episodes of one terrain into units; one unit is one process and one scene.

    An object category gives one unit per (shape, size level, bed variant). A free-fall episode is a unit of its
    own, because the particle count of the block is fixed when the scene is built. Bare beds are chunked per bed
    variant. The units are ordered longest first.
    """
    units = []
    for category in dict.fromkeys(c for name in select for c in SELECTIONS[name]):
        records = episodes(terrain, category, per_motion)
        if category in OBJECT_CATEGORIES:
            groups = sorted({(r["shape"], r["size_level"], r["terrain_variant"]) for r in records})
            for shape, level, variant in groups:
                count = sum(
                    1 for r in records if (r["shape"], r["size_level"], r["terrain_variant"]) == (shape, level, variant)
                )
                units.append(
                    {
                        "label": f"{terrain}-{category}-{shape}-s{level}-{variant}",
                        "args": [
                            *("--terrain", terrain, "--category", category, "--shape", shape),
                            *("--size-level", str(level), "--variant", variant, "--per-motion", str(per_motion)),
                        ],
                        "cost": 60 + count * COST[category],
                    }
                )
            continue
        if category == "freefall":
            chunks = [(r["terrain_variant"], [r["index"]]) for r in records]
        else:
            chunks = []
            for variant in sorted({r["terrain_variant"] for r in records}):
                indices = [r["index"] for r in records if r["terrain_variant"] == variant]
                chunks += [
                    (variant, indices[start : start + SAND_CHUNK]) for start in range(0, len(indices), SAND_CHUNK)
                ]
        for variant, chunk in chunks:
            units.append(
                {
                    "label": f"{terrain}-{category}-{variant}-{chunk[0]:04d}",
                    "args": [
                        *("--terrain", terrain, "--category", category, "--variant", variant),
                        *("--per-motion", str(per_motion), "--indices", *map(str, chunk)),
                    ],
                    "cost": 60 + len(chunk) * COST[category],
                }
            )
    return sorted(units, key=lambda u: -u["cost"])


def run_unit(unit: dict, gpu: str, dataset: Path, log_dir: Path, attempts: int) -> dict:
    """Run one unit as a process of this script on one GPU; its output goes to a log file."""
    log_path = log_dir / f"{unit['label']}.log"
    command = [sys.executable, str(Path(__file__).resolve()), *unit["args"], "--dataset", str(dataset)]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu, "PYTHONUNBUFFERED": "1"}
    start = time.perf_counter()
    status = "failed"
    for attempt in range(1, attempts + 1):
        with open(log_path, "a") as log:
            log.write(f"=== attempt {attempt}: {' '.join(command)}\n")
            log.flush()
            code = subprocess.run(
                command, env=env, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL
            ).returncode
        if code == 0:
            status = "ok"
            break
        print(f"[{unit['label']}] attempt {attempt} exited with {code}", flush=True)
        tail = log_path.read_text().splitlines()[-25:]
        print("\n".join(f"[{unit['label']}]   | {line[:200]}" for line in tail), flush=True)
    print(f"[{unit['label']}] {status} in {time.perf_counter() - start:.0f} s on GPU {gpu}", flush=True)
    return {"label": unit["label"], "status": status, "seconds": round(time.perf_counter() - start, 1)}


def run_units(args: argparse.Namespace, units: list[dict]) -> None:
    """Run ``units`` on ``args.gpus`` GPUs, one at a time per GPU, then summarize the files of the terrain."""
    log_dir = args.dataset / "logs" / args.terrain
    log_dir.mkdir(parents=True, exist_ok=True)
    gpus: Queue = Queue()
    for index in range(args.gpus):
        gpus.put(str(index))

    def work(unit):
        gpu = gpus.get()
        try:
            return run_unit(unit, gpu, args.dataset, log_dir, args.attempts)
        finally:
            gpus.put(gpu)

    print(f"{len(units)} units on {args.gpus} GPU(s), terrain {args.terrain}, selection {args.select}", flush=True)
    with ThreadPoolExecutor(max_workers=args.gpus) as pool:
        results = list(pool.map(work, units))
    failed = [r for r in results if r["status"] != "ok"]
    name = f"{args.terrain}-{'+'.join(args.select)}"
    (args.dataset / "logs" / f"{name}.units.json").write_text(json.dumps(results, indent=2))
    print(f"{len(results) - len(failed)} units ok, {len(failed)} failed: {[r['label'] for r in failed]}", flush=True)
    # The exit code stays 0 so that a job still uploads its files; failures are listed above and in the units file.
    summarize_files(args.dataset, args.dataset / "logs" / f"{name}.files.jsonl", args.terrain)


def record(args: argparse.Namespace, records: list[dict], variant: str) -> None:  # noqa: C901
    """Simulate ``records`` with MPM in one scene and write one file per episode.

    Args:
        args: Command-line arguments (terrain, category, shape, size level, dataset folder, device).
        records: Episodes to record. They share the object, the bed variant and, for free fall, the particle
            count of the block.
        variant: Surface of the bed.
    """
    import newton
    import warp as wp
    from isaaclab_newton.physics import NewtonManager, NewtonMPMManager
    from scene import (
        DENSITY,
        PARTICLE_JITTER,
        SPACING,
        VOXEL_SIZE,
        BodyWrenchAction,
        QuietObjectStateAction,
        make_env_cfg,
        sand_cfg,
        surface_height,
    )

    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import ActionTermCfg

    sand_only = args.category in SAND_CATEGORIES
    size = TERRAINS[args.terrain]["bed_size_m"]
    box_scene = sand_only or args.shape in ("cube", "box")
    cfg = make_env_cfg("box" if box_scene else args.shape, variant, size)
    cfg.sim.device = args.device
    cfg.sim.visualizer_cfgs = []
    geometry = None
    if not sand_only:
        geometry = object_geometry(args.shape, args.size_level)
        spawn = cfg.scene.object.spawn
        if args.shape in ("cube", "box"):
            spawn.size = tuple(geometry["object_size_m"])
        elif args.shape == "sphere":
            spawn.radius = geometry["object_size_m"][0] / 2.0
        else:
            spawn.radius = geometry["object_size_m"][0] / 2.0
            spawn.height = geometry["object_size_m"][2]
        spawn.mass_props.mass = geometry["mass_kg"]
    # Sand-only episodes keep the object parked kinematically; the sand never touches it.
    kinematic = sand_only or args.category in ("lateral_push", "vertical_push")
    if kinematic:
        cfg.actions.motion = ActionTermCfg(class_type=QuietObjectStateAction, asset_name="object")
    else:
        cfg.scene.object.spawn.rigid_props.kinematic_enabled = False
        cfg.actions.motion = ActionTermCfg(class_type=BodyWrenchAction, asset_name="object")
    mass = float(cfg.scene.object.spawn.mass_props.mass)
    material = cfg.scene.sand.spawn.material.to_dict()

    # Free fall: the released sand block is part of the same particle set, after the bed's particles.
    blocks = {}
    if args.category == "freefall":
        for r in records:
            surface = float(surface_height(np.float64(r["start_xy"][0]), variant))
            blocks[r["id"]] = block_particles(r, surface, SPACING, PARTICLE_JITTER)
        if len({len(b) for b in blocks.values()}) != 1:
            raise ValueError("Free-fall blocks with different particle counts need separate processes; pass --indices")
        cfg.scene.sand.spawn.positions = list(cfg.scene.sand.spawn.positions) + blocks[records[0]["id"]].tolist()
    num_block = len(blocks[records[0]["id"]]) if blocks else 0

    terrain_json = args.dataset / "terrain" / f"{args.terrain}.json"
    terrain_json.parent.mkdir(parents=True, exist_ok=True)
    if not terrain_json.exists():
        terrain_json.write_text(
            json.dumps(
                {
                    "terrain": args.terrain,
                    "bed_size_m": list(size),
                    "surface": TERRAINS[args.terrain]["surface"],
                    "slope_deg": 10.0 if args.terrain == "slope" else None,
                    "boundary_boxes": tray_boundary_boxes(size),
                    "particle_spacing_m": SPACING,
                    "voxel_size_m": VOXEL_SIZE,
                    "jitter_max_m": PARTICLE_JITTER,
                    "density": DENSITY,
                    "material": material,
                    "gravity": [0.0, 0.0, -GRAVITY],
                    "frame": "World axes, origin at the tray centre on the nominal surface; quaternions xyzw",
                },
                indent=2,
            )
        )

    with launch_simulation(cfg, args):
        env = ManagerBasedRLEnv(cfg)
        try:
            with torch.inference_mode():
                model = NewtonManager.get_model()
                if model.body_count != 1 or env.num_envs != 1:
                    raise RuntimeError("Collector expects one interacting rigid body in one world")
                entry = NewtonManager._solver._entries["sand"]
                body_com = model.body_com.numpy()[0]
                num_bed = model.particle_count - num_block
                origin = env.scene.env_origins[0]

                def set_block(positions, velocity):
                    """Overwrite the state of the sand block (the last ``num_block`` particles) in the live state."""
                    seen = set()
                    for state in (NewtonManager.get_state_0(), entry.state_0):
                        if id(state) in seen:
                            continue
                        seen.add(id(state))
                        wp.to_torch(state.particle_q)[num_bed:] = positions
                        wp.to_torch(state.particle_qd)[num_bed:] = velocity

                for record in records:
                    target = (
                        args.dataset / record["split"] / args.terrain / args.category / f"{record['index']:04d}.hdf5"
                    )
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if target.exists():
                        with h5py.File(target, "r") as existing:
                            if existing.attrs.get("complete", False):
                                continue
                    env.reset(seed=record["seed"])
                    initial = np.asarray(
                        sand_cfg(variant, size, seed=record["particle_seed"]).spawn.positions, dtype=np.float32
                    )
                    if num_block:
                        initial = np.concatenate((initial, blocks[record["id"]]))
                    if len(initial) != model.particle_count:
                        raise RuntimeError(
                            f"Particle count {len(initial)} differs from the model {model.particle_count}"
                        )
                    sand = env.scene["sand"]
                    nodal = sand.data.default_nodal_state_w.torch.clone()
                    nodal[0, :, :3] = torch.as_tensor(initial, device=env.device) + origin
                    nodal[0, :, 3:] = 0.0
                    sand.write_nodal_state_to_sim_index(nodal)

                    def lateral_command(t):
                        pose = (lateral_pose if args.category == "lateral_push" else vertical_pose)(t, record)
                        pose[2] += float(surface_height(pose[0], variant))
                        return pose

                    parked = np.array(PARKED_POSE, dtype=np.float32)
                    if sand_only:
                        initial_pose = parked
                    elif kinematic:
                        initial_pose = lateral_command(0.0)
                    else:
                        x0, y0 = record["start_xy"]
                        height = (
                            record["release_height"] if args.category == "impact" else record["half_height"] - 0.005
                        )
                        initial_pose = np.array(
                            [x0, y0, float(surface_height(x0, variant)) + height, *_yaw_quat(record["yaw"])],
                            dtype=np.float32,
                        )
                    # Pre-settle: park the object above the bed and let the sand settle before recording.
                    env.scene["object"].write_root_pose_to_sim_index(
                        root_pose=torch.as_tensor(parked[None], device=env.device)
                    )
                    env.scene["object"].write_root_velocity_to_sim_index(
                        root_velocity=torch.zeros((1, 6), device=env.device)
                    )
                    env.sim.forward()
                    NewtonMPMManager.reset_solver_state(flags=newton.StateFlags.BODY | newton.StateFlags.PARTICLE)
                    env.scene.update(DT)
                    if kinematic:
                        hold = np.concatenate((parked, np.zeros(6, dtype=np.float32)))[None]
                    else:
                        hold = np.array([[0.0, 0.0, mass * GRAVITY, 0.0, 0.0, 0.0]], dtype=np.float32)
                    hold = torch.as_tensor(hold, device=env.device)
                    if num_block:
                        # The block waits at its release position while the bed settles: its particles are put
                        # back after every step, so they never fall or interact with the bed.
                        block_start = torch.as_tensor(blocks[record["id"]], device=env.device) + origin
                        block_rest = torch.zeros_like(block_start)
                    hold_drift = 0.0
                    for _ in range(round(record["settle_s"] / DT)):
                        env.step(hold)
                        if num_block:
                            now = wp.to_torch(NewtonManager.get_state_0().particle_q)[num_bed:]
                            hold_drift = max(hold_drift, float((now - block_start).norm(dim=-1).max()))
                            set_block(block_start, block_rest)
                    if hold_drift > 2.0e-3:
                        # One step of free fall from rest moves a particle 0.25 mm; more means the hold did not work.
                        raise RuntimeError(f"The sand block moved {hold_drift * 1e3:.1f} mm while the bed settled")
                    if num_block:
                        # Release: linear velocity plus rigid rotation about the block centre.
                        lin = torch.as_tensor(record["release_velocity"][:3], device=env.device, dtype=torch.float32)
                        ang = torch.as_tensor(record["release_velocity"][3:], device=env.device, dtype=torch.float32)
                        offset = block_start - block_start.mean(0, keepdim=True)
                        set_block(block_start, lin[None] + torch.linalg.cross(ang[None].expand_as(offset), offset))
                    if not sand_only:
                        # Place the object at its start pose with zero velocity; keep the settled sand state.
                        env.scene["object"].write_root_pose_to_sim_index(
                            root_pose=torch.as_tensor(initial_pose[None], device=env.device)
                        )
                        env.scene["object"].write_root_velocity_to_sim_index(
                            root_velocity=torch.zeros((1, 6), device=env.device)
                        )
                        env.sim.forward()
                        NewtonMPMManager.reset_solver_state(flags=newton.StateFlags.BODY)
                        env.scene.update(DT)

                    steps = round(record["duration_s"] / DT)
                    positions = np.empty((steps + 1, model.particle_count, 3), dtype=np.float32)
                    body_state = np.empty((steps + 1, 13), dtype=np.float32)
                    ext = np.zeros((steps, 6), dtype=np.float32)
                    contact = np.zeros((steps, 6), dtype=np.float32)

                    def capture(index):
                        state = NewtonManager.get_state_0()
                        positions[index] = state.particle_q.numpy()
                        body_state[index] = np.concatenate((state.body_q.numpy(), state.body_qd.numpy()), axis=-1)[0]

                    capture(0)
                    if not np.allclose(body_state[0, :7], initial_pose, atol=1e-5, rtol=0):
                        raise RuntimeError(
                            f"Initial pose not synchronized: state {body_state[0, :7]} vs start {initial_pose}"
                        )
                    hover_steps = round(HOVER_S / DT)
                    hover = args.category == "impact"
                    start = time.perf_counter()

                    for step in range(steps):
                        t = step * DT
                        if sand_only:
                            env.step(hold)
                            capture(step + 1)
                            continue
                        if kinematic:
                            now, ahead = lateral_command(t), lateral_command(t + DT)
                            command = np.concatenate((now, (ahead[:3] - now[:3]) / DT, np.zeros(3, dtype=np.float32)))
                            env.step(torch.as_tensor(command[None].astype(np.float32), device=env.device))
                        else:
                            wrench = np.zeros(6, dtype=np.float32)
                            if hover and step < hover_steps:
                                wrench[2] = mass * GRAVITY
                            elif hover and step == hover_steps:
                                velocity = np.asarray(record.get("release_velocity", [0.0] * 6), dtype=np.float32)
                                env.scene["object"].write_root_velocity_to_sim_index(
                                    root_velocity=torch.as_tensor(velocity[None], device=env.device)
                                )
                            ext[step] = wrench
                            env.step(torch.as_tensor(wrench[None], device=env.device))
                        capture(step + 1)
                        if args.category == "rest":
                            # Sink only: hold x, y and orientation and remove lateral and angular velocity every step.
                            vz = float(body_state[step + 1, 9])
                            keep = torch.tensor([[0.0, 0.0, vz, 0.0, 0.0, 0.0]], device=env.device)
                            env.scene["object"].write_root_velocity_to_sim_index(root_velocity=keep, skip_forward=True)
                            held = initial_pose.copy()
                            held[2] = body_state[step + 1, 2]
                            env.scene["object"].write_root_pose_to_sim_index(
                                root_pose=torch.as_tensor(held[None], device=env.device), skip_forward=True
                            )
                        impulses, points, ids = entry.solver.collect_collider_impulses(entry.state_0)
                        ids = wp.to_torch(ids).long()
                        keep = ids >= 0
                        owner = wp.to_torch(entry.solver.collider_body_index)[ids[keep]]
                        use = (owner >= 0).cpu().numpy()
                        force = wp.to_torch(impulses)[keep].cpu().numpy()[use] / DT
                        where = wp.to_torch(points)[keep].cpu().numpy()[use]
                        pose_now = body_state[step + 1, :7]
                        com = pose_now[:3] + _rotate(pose_now[3:7], body_com)
                        contact[step, :3] = force.sum(0)
                        contact[step, 3:] = np.cross(where - com, force).sum(0)

                    common = {
                        "particle_spacing_m": SPACING,
                        "voxel_size_m": VOXEL_SIZE,
                        "particle_mass_kg": DENSITY * SPACING**3,
                        "material": material,
                        "bed_size_m": list(size),
                        "scene": "coupled MPM scene of scripts/neural/scene.py, the same for every category",
                    }
                    if sand_only:
                        if np.abs(body_state[:, :3] - parked[:3]).max() > 1e-3:
                            raise RuntimeError(f"The parked object moved in {record['id']}")
                        body = None
                        attrs = {
                            **common,
                            "num_block_particles": num_block,
                            "block_hold_drift_m": hold_drift,
                            "particle_order": (
                                "bed particles first, then the released sand block" if num_block else "bed only"
                            ),
                            "collection_seconds": time.perf_counter() - start,
                        }
                    else:
                        body = {
                            "pose": body_state[:, :7],
                            "vel": body_state[:, 7:13],
                            "ext_wrench": ext,
                            "contact_wrench": contact,
                        }
                        if not all(np.isfinite(a).all() for a in body.values()):
                            raise RuntimeError(f"Nonfinite body state in {record['id']}")
                        attrs = {
                            **common,
                            "object_mass_kg": mass,
                            "object_size_m": geometry["object_size_m"],
                            "size_level": args.size_level,
                            "hover_until_s": HOVER_S if hover else 0.0,
                            "constraint": (
                                "vertical translation only (x, y, orientation, lateral and angular velocity "
                                "reset every step)"
                            )
                            if args.category == "rest"
                            else None,
                            "wrench_convention": (
                                "world frame, force [N] then torque [N m] about the body centre of mass"
                            ),
                            "ext_wrench_note": (
                                "applied wrench per step, including the anti-gravity hover wrench before release"
                            ),
                            "contact_wrench_note": (
                                "MPM sand-on-body wrench per step (impulse/dt), tray forces excluded"
                            ),
                            "pose_columns": "xyz, quaternion xyzw",
                            "vel_columns": "linear xyz, angular xyz (world frame)",
                            "wrench_alignment": "row k is applied between state k and state k+1",
                            "collection_seconds": time.perf_counter() - start,
                        }
                    write_episode(target, record, positions, body, attrs)
                    print(
                        json.dumps(
                            {
                                "episode": record["id"],
                                "shape": record["shape"],
                                "motion": record["motion"],
                                "steps": steps,
                                "seconds": round(time.perf_counter() - start, 1),
                                "bytes": target.stat().st_size,
                            }
                        ),
                        flush=True,
                    )
        finally:
            env.close()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, required=True, help="Folder of the dataset; files are added to it.")
    parser.add_argument("--terrain", choices=tuple(TERRAINS), required=True, help="Sand bed.")
    parser.add_argument(
        "--category",
        choices=(*OBJECT_CATEGORIES, *SAND_CATEGORIES),
        help="Record episodes of this category that share one scene, in this process.",
    )
    parser.add_argument(
        "--select",
        nargs="+",
        choices=tuple(SELECTIONS),
        help="Record these groups of categories, one process per scene on each GPU.",
    )
    parser.add_argument("--shape", choices=SHAPES, help="Object shape (with an object --category).")
    parser.add_argument(
        "--size-level", type=int, choices=range(len(SIZE_SCALES)), help="Object size level (with an object --category)."
    )
    parser.add_argument("--variant", help="Bed surface, for example slope_p10; needed when the episodes mix two.")
    parser.add_argument("--indices", type=int, nargs="+", help="Record only these episode indices (with --category).")
    parser.add_argument("--duration", type=float, help="Shorten every episode to this many seconds (checks only).")
    parser.add_argument("--per-motion", type=int, default=PER_MOTION, help="Episodes per motion; 1 for a test run.")
    parser.add_argument("--gpus", type=int, default=1, help="GPUs to spread the processes over (with --select).")
    parser.add_argument("--attempts", type=int, default=2, help="Tries per process (with --select).")
    parser.add_argument("--list", action="store_true", help="Print the processes of --select and exit.")
    parser.add_argument("--only", nargs="+", help="Run only the processes with these labels (with --select).")
    add_launcher_args(parser)
    parser.set_defaults(visualizer=[])
    args = parser.parse_args(argv)

    if (args.category is None) == (args.select is None):
        parser.error("name either one --category, to record in this process, or --select, to record groups of them")
    if args.per_motion < 1 or args.gpus < 1 or args.attempts < 1:
        parser.error("--per-motion, --gpus and --attempts must be 1 or more")
    if args.select is not None:
        single = {"--shape": args.shape, "--size-level": args.size_level, "--variant": args.variant}
        single.update({"--indices": args.indices, "--duration": args.duration})
        given = [name for name, value in single.items() if value is not None]
        if given:
            parser.error(f"{', '.join(given)} belong to --category; --select chooses them for every process")
        if args.terrain not in PRODUCTION_TERRAINS:
            parser.error(f"--select records the terrains {', '.join(PRODUCTION_TERRAINS)}; got {args.terrain}")
        units = build_units(args.terrain, args.select, args.per_motion)
        if args.only:
            unknown = sorted(set(args.only) - {u["label"] for u in units})
            if unknown:
                parser.error(f"no process is labelled {', '.join(unknown)}; --list prints the labels")
            units = [u for u in units if u["label"] in set(args.only)]
        if args.list:
            for unit in units:
                print(f"{unit['cost']:6d}  {unit['label']}")
            print(f"{len(units)} units")
            return
        run_units(args, units)
        return

    if args.list or args.only:
        parser.error("--list and --only belong to --select")
    if args.duration is not None and args.duration <= 0.0:
        parser.error("--duration must be positive")
    if args.category in OBJECT_CATEGORIES and (args.shape is None or args.size_level is None):
        parser.error(f"--shape and --size-level are required for the object category {args.category}")
    if args.category in SAND_CATEGORIES and (args.shape is not None or args.size_level is not None):
        parser.error(f"the sand-only category {args.category} has no object; leave out --shape and --size-level")
    records = select_records(
        args.terrain, args.category, args.per_motion, args.shape, args.size_level, args.variant, args.indices
    )
    if not records:
        print("No episode matches this shape, size level, variant and indices.", flush=True)
        return
    variants = sorted({r["terrain_variant"] for r in records})
    if len(variants) != 1:
        parser.error(f"the episodes mix the bed surfaces {', '.join(variants)}; choose one with --variant")
    if args.duration is not None:
        for r in records:
            r["duration_s"] = args.duration
            r["smoke_check"] = True
    record(args, records, variants[0])


if __name__ == "__main__":
    main()
