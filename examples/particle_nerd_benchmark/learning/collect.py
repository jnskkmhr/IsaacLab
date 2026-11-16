# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Collect only particle positions and rigid-body states at native 200 Hz."""

import argparse
import hashlib
import importlib.metadata
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch

from isaaclab.app import add_launcher_args, launch_simulation

from .protocol import DT, SHAPES, SIZES, command, episodes, pose, release_velocity


def select_records(args):
    """Resolve full campaign episodes or separately labeled evaluation references."""
    records = [r for r in episodes() if r["shape"] == args.shape and r["mode"] == args.mode]
    if args.episode_ids:
        records = [r for r in records if r["id"] in args.episode_ids]
        if len(records) != len(args.episode_ids):
            raise ValueError("Episode IDs must match the selected shape and mode")
    if args.limit is not None:
        records = records[: args.limit]
    if args.duration is not None:
        for record in records:
            record["duration_s"] = args.duration
            record["smoke_check"] = True
    if args.reference:
        expected_mode = "prescribed" if args.reference in ("horizontal-flat", "fixed-hold") else "dynamic"
        if args.mode != expected_mode:
            raise ValueError(f"{args.reference} requires --mode {expected_mode}")
        record = records[0].copy()
        record.update(
            id=f"{args.shape}-{args.reference}",
            split="test",
            seed=42,
            particle_seed=42,
            yaw=0.0,
            y=0.0,
            direction=1,
            release_height=0.25,
            motion="horizontal" if expected_mode == "prescribed" else "drop",
            duration_s=6.0 if expected_mode == "prescribed" else 20.0,
        )
        if args.reference == "fixed-hold":
            record.update(
                motion="hold",
                entry_s=1.0,
                depth=-0.025,
                duration_s=23.0,
                hold_start_s=2.0,
                consistency_start_s=3.0,
            )
        records = [record]
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", choices=SHAPES, required=True)
    parser.add_argument("--mode", choices=("prescribed", "dynamic"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, help="Limit episodes for a local smoke check")
    parser.add_argument("--episode-ids", nargs="+", help="Collect specified complete episodes for validation")
    parser.add_argument("--duration", type=float, help="Override duration for a smoke check only")
    parser.add_argument(
        "--reference",
        choices=("horizontal-flat", "static-support", "fixed-hold"),
        help="Record a separate evaluation reference, including MPM material forces",
    )
    add_launcher_args(parser)
    parser.set_defaults(visualizer=[])
    args = parser.parse_args()
    import newton
    from isaaclab_newton.physics import NewtonManager, NewtonMPMManager

    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import ActionTermCfg

    from ..core.case_config import make_config
    from ..core.mdp.actions import BodyWrenchAction
    from ..core.terrain import PARTICLE_JITTER, SPACING, VOXEL_SIZE, sand_cfg

    records = select_records(args)
    cfg = make_config("horizontal", "flat", geometry=args.shape)
    cfg.sim.device = args.device
    cfg.sim.visualizer_cfgs = []
    dynamic = args.mode == "dynamic"
    if dynamic:
        cfg.scene.object.spawn.rigid_props.kinematic_enabled = False
        cfg.actions.motion = ActionTermCfg(class_type=BodyWrenchAction, asset_name="object")
    args.output.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).parents[1]
    manifest = {
        "schema": "particle-nerd-object-trajectories-v1",
        "episodes": records,
        "particle_fields": ["particle_position"],
        "rigid_fields": ["body_state", "collider_state"],
        "frame": "World axes, origin at initial tray centre; quaternions xyzw",
        "body_state_timing": "After each completed physics step, including the initial state",
        "collider_state_timing": "At entry to the MPM step, after proxy rewind, before material solve",
        "collider_state_velocity": "MPM input velocity at body COM after removal of lagged material feedback",
        "state_columns": "xyz, quaternion xyzw, linear xyz, angular xyz",
        "dt_s": DT,
        "evaluation_reference": args.reference,
        "force_definition": (
            "MPM material-on-object impulse/dt in world axes, direct rigid tray forces excluded"
            if args.reference
            else None
        ),
        "voxel_size_m": VOXEL_SIZE,
        "particle_spacing_m": SPACING,
        "jitter_max_m": PARTICLE_JITTER,
        "body_size_m": SIZES[args.shape],
        "configured_body_mass_kg": 2.0,
        "body_mass_note": (
            "Native kinematic bodies have zero solver mass/inertia; dynamic bodies have the configured mass"
        ),
        "material": cfg.scene.sand.spawn.material.to_dict(),
        "physics": cfg.sim.physics.to_dict(),
        "source_hashes": {
            str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source.rglob("*.py")
        },
        "versions": {n: importlib.metadata.version(n) for n in ("newton", "warp-lang", "torch", "numpy")},
        "boundary_boxes": [
            {
                "name": name,
                "size_m": getattr(cfg.scene, name).spawn.size,
                "pose": [*getattr(cfg.scene, name).init_state.pos, 0, 0, 0, 1],
            }
            for name in ("floor", "wall_x0", "wall_x1", "wall_y0", "wall_y1")
        ],
    }
    manifest_path = args.output / "manifest.json"
    serialized = json.dumps(manifest, indent=2, default=str)
    if manifest_path.exists() and manifest_path.read_text() != serialized:
        raise RuntimeError("Output manifest differs; use a new directory")
    manifest_path.write_text(serialized)
    with launch_simulation(cfg, args):
        env = ManagerBasedRLEnv(cfg)
        try:
            with torch.inference_mode():
                model = NewtonManager.get_model()
                if model.body_count != 1 or env.num_envs != 1:
                    raise RuntimeError("Compact collector expects one interacting rigid body in one world")
                entry = NewtonManager._solver._entries["sand"]
                original_step = entry.solver.step
                input_states = []

                def record_input(state_in, *step_args, **step_kwargs):
                    input_states.append(np.concatenate((state_in.body_q.numpy(), state_in.body_qd.numpy()), axis=-1)[0])
                    return original_step(state_in, *step_args, **step_kwargs)

                entry.solver.step = record_input
                for record in records:
                    target = args.output / (record["id"] + ".h5")
                    if target.exists():
                        with h5py.File(target, "r") as existing:
                            if not existing.attrs.get("complete", False):
                                raise RuntimeError(f"Incomplete existing episode: {target}")
                        continue
                    env.reset(seed=record["seed"])
                    initial = np.asarray(sand_cfg(seed=record["particle_seed"]).spawn.positions, dtype=np.float32)
                    sand = env.scene["sand"]
                    nodal = sand.data.default_nodal_state_w.torch.clone()
                    nodal[0, :, :3] = torch.as_tensor(initial, device=env.device) + env.scene.env_origins[0]
                    sand.write_nodal_state_to_sim_index(nodal)
                    if dynamic:
                        initial_pose = pose(0.0, record)
                        initial_pose[:3] = (-record["direction"] * 0.30, record["y"], record["release_height"])
                    else:
                        initial_pose = pose(0.0, record)
                    if args.reference == "horizontal-flat":
                        from ..core.motions import object_command

                        initial_pose = object_command(0.0, "horizontal", "flat", DT)[:7]
                    env.scene["object"].write_root_pose_to_sim_index(
                        root_pose=torch.as_tensor(initial_pose[None], device=env.device)
                    )
                    env.scene["object"].write_root_velocity_to_sim_index(
                        root_velocity=torch.zeros((1, 6), device=env.device)
                    )
                    env.sim.forward()
                    NewtonMPMManager.reset_solver_state(flags=newton.StateFlags.BODY | newton.StateFlags.PARTICLE)
                    env.scene.update(DT)
                    steps = round(record["duration_s"] / DT)
                    positions = np.empty((steps + 1, model.particle_count, 3), dtype=np.float32)
                    body_states = np.empty((steps + 1, 13), dtype=np.float32)
                    forces = np.empty((steps, 3), dtype=np.float32) if args.reference else None
                    input_states.clear()
                    start = time.perf_counter()

                    def capture(index):
                        state = NewtonManager.get_state_0()
                        positions[index] = state.particle_q.numpy()
                        body_states[index] = np.concatenate((state.body_q.numpy(), state.body_qd.numpy()), axis=-1)[0]

                    capture(0)
                    if not np.allclose(positions[0], initial, atol=1e-7, rtol=0):
                        raise RuntimeError("Initial particle positions do not match the episode seed")
                    if not np.allclose(body_states[0, :7], initial_pose, atol=1e-6, rtol=0):
                        raise RuntimeError("Initial rigid-body pose was not synchronized before recording")
                    for step in range(steps):
                        if dynamic:
                            action = torch.zeros((1, 6), device=env.device)
                            if step < round(1.0 / DT):
                                action[0, 2] = 2.0 * 9.81
                            elif step == round(1.0 / DT):
                                env.scene["object"].write_root_velocity_to_sim_index(
                                    root_velocity=torch.as_tensor(release_velocity(record)[None], device=env.device)
                                )
                            env.step(action)
                        else:
                            value = (
                                object_command(step * DT, "horizontal", "flat", DT)
                                if args.reference == "horizontal-flat"
                                else command(step * DT, record)
                            )
                            env.step(torch.as_tensor(value[None], device=env.device))
                        capture(step + 1)
                        if forces is not None:
                            import warp as wp

                            impulses, _, collider_ids = entry.solver.collect_collider_impulses(entry.state_0)
                            ids = wp.to_torch(collider_ids).long()
                            keep = ids >= 0
                            bodies = wp.to_torch(entry.solver.collider_body_index)[ids[keep]]
                            forces[step] = wp.to_torch(impulses)[keep][bodies >= 0].sum(0).cpu().numpy() / DT
                    colliders = np.asarray(input_states, dtype=np.float32)
                    if len(colliders) != steps:
                        raise RuntimeError(
                            f"Expected one MPM collider input per step; got {len(colliders)} for {steps}"
                        )
                    if not all(np.isfinite(a).all() for a in (positions, body_states, colliders)):
                        raise RuntimeError(f"Nonfinite state in {record['id']}")
                    partial = target.with_suffix(".partial.h5")
                    with h5py.File(partial, "w") as handle:
                        handle.attrs["episode"] = json.dumps(record)
                        handle.attrs["manifest_sha256"] = hashlib.sha256(serialized.encode()).hexdigest()
                        handle.create_dataset(
                            "particle_position",
                            data=positions,
                            chunks=(8, model.particle_count, 3),
                            compression="gzip",
                            compression_opts=1,
                            shuffle=True,
                        )
                        handle.create_dataset("body_state", data=body_states)
                        handle.create_dataset("collider_state", data=colliders)
                        if forces is not None:
                            if not np.isfinite(forces).all():
                                raise RuntimeError("Nonfinite material force in reference")
                            handle.create_dataset("material_force", data=forces)
                        for name in ("particle_mass", "particle_radius", "body_mass", "body_inertia", "body_com"):
                            handle.create_dataset(name, data=getattr(model, name).numpy())
                        for name in (
                            "shape_body",
                            "shape_type",
                            "shape_scale",
                            "shape_transform",
                            "shape_margin",
                            "shape_material_mu",
                        ):
                            handle.create_dataset(name, data=getattr(model, name).numpy())
                        handle.create_dataset(
                            "shape_labels", data=np.asarray(model.shape_label, dtype=h5py.string_dtype())
                        )
                        handle.attrs["complete"] = True
                        handle.attrs["collection_seconds"] = time.perf_counter() - start
                    partial.rename(target)
                    print(
                        json.dumps(
                            {
                                "episode": record["id"],
                                "split": record["split"],
                                "shape": args.shape,
                                "mode": args.mode,
                                "steps": steps,
                                "seconds": round(time.perf_counter() - start, 2),
                                "bytes": target.stat().st_size,
                            }
                        ),
                        flush=True,
                    )
                entry.solver.step = original_step
        finally:
            env.close()


if __name__ == "__main__":
    main()
