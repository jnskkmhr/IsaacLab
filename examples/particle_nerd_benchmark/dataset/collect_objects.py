# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Collect object and sand-only episodes on one sand terrain with MPM, all in the same coupled scene.

Every category (lateral push, vertical push, impact, rest, rest_bed, freefall) runs in the scene of `core.physics`
and `core.terrain`, so the sand material, the tray colliders and the solver settings are identical. Only the bed
surface and the object (shape, size, motion) change. Sand-only episodes keep the rigid object parked above the bed.

One process handles one object shape and one bed variant, because the environment is built for that geometry. A
free-fall process handles blocks of one particle count (pass ``--indices``).
"""

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch

from isaaclab.app import add_launcher_args, launch_simulation

from .episodes import (
    DT,
    HOVER_S,
    OBJECT_CATEGORIES,
    PER_MOTION,
    SAND_CATEGORIES,
    SHAPES,
    SIZE_SCALES,
    TERRAINS,
    episodes,
    lateral_pose,
    object_geometry,
    tray_boundary_boxes,
    vertical_pose,
)
from .hdf5_io import write_episode

GRAVITY = 9.81
PARKED_POSE = (0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 1.0)
"""Pose of the rigid object while it must not touch the sand (pre-settle, and all of a sand-only episode)."""


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


def main():  # noqa: C901
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--terrain", choices=tuple(TERRAINS), required=True)
    parser.add_argument("--category", choices=(*OBJECT_CATEGORIES, *SAND_CATEGORIES), required=True)
    parser.add_argument("--shape", choices=SHAPES, help="Object shape (object categories only)")
    parser.add_argument("--variant", help="Bed surface variant (for example slope_p10); default: the only one")
    parser.add_argument(
        "--size-level", type=int, choices=range(len(SIZE_SCALES)), help="Object size level (object categories only)"
    )
    parser.add_argument("--per-motion", type=int, default=PER_MOTION, help="Episodes per motion (test run: 1)")
    parser.add_argument("--indices", type=int, nargs="+", help="Collect only these episode indices")
    parser.add_argument("--duration", type=float, help="Override the duration (smoke check only)")
    parser.add_argument("--output", type=Path, required=True)
    add_launcher_args(parser)
    parser.set_defaults(visualizer=[])
    args = parser.parse_args()

    sand_only = args.category in SAND_CATEGORIES
    records = episodes(args.terrain, args.category, args.per_motion)
    if not sand_only:
        if args.shape is None or args.size_level is None:
            parser.error("--shape and --size-level are required for object categories")
        records = [r for r in records if r["shape"] == args.shape and r["size_level"] == args.size_level]
    if args.indices:
        records = [r for r in records if r["index"] in args.indices]
    if args.variant:
        records = [r for r in records if r["terrain_variant"] == args.variant]
    if not records:
        print("No matching episodes for this shape/variant.", flush=True)
        return
    variants = {r["terrain_variant"] for r in records}
    if len(variants) != 1:
        raise ValueError(f"Mixed bed variants {sorted(variants)}; pass --variant")
    variant = variants.pop()
    if args.duration is not None:
        for r in records:
            r["duration_s"] = args.duration
            r["smoke_check"] = True

    import newton
    import warp as wp
    from isaaclab_newton.physics import NewtonManager, NewtonMPMManager

    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import ActionTermCfg

    from ..core.case_config import make_config
    from ..core.mdp.actions import BodyWrenchAction
    from ..core.terrain import (
        DENSITY,
        PARTICLE_JITTER,
        SPACING,
        VOXEL_SIZE,
        configure_terrain,
        sand_cfg,
        surface_height,
    )
    from .actions import QuietObjectStateAction

    size = TERRAINS[args.terrain]["bed_size_m"]
    box_scene = sand_only or args.shape in ("cube", "box")
    cfg = make_config("horizontal", "flat", geometry="box" if box_scene else args.shape)
    configure_terrain(cfg.scene, variant, size)
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

    terrain_json = args.output / "terrain" / f"{args.terrain}.json"
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
                        args.output / record["split"] / args.terrain / args.category / f"{record['index']:04d}.hdf5"
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
                        "scene": "coupled MPM scene of core.physics (same for every object and sand-only category)",
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


if __name__ == "__main__":
    main()
