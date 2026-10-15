# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Collect complete MPM episodes using Isaac Lab's recorder manager."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import subprocess
import time
from pathlib import Path

import h5py
import torch
from env_cfg import ParticleEnvCfg
from scene_cfg import MOVING_BODY_ID, VOXEL_SIZE

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.envs import ManagerBasedEnv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New .hdf5 file; existing files are rejected.")
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--episodes_per_env", type=int, default=2)
    parser.add_argument(
        "--steps", type=int, default=400, help="400 steps records two seconds of free motion and contact."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--body-mass", type=float, default=2.0, help="Dynamic box mass [kg].")
    parser.add_argument("--mpm-iterations", type=int, choices=(25, 50), default=25)
    parser.add_argument("--material-friction", type=float, default=None)
    parser.add_argument("--material-density", type=float, default=None, help="Material density [kg/m^3].")
    parser.add_argument("--split", choices=("train", "validation", "test"), default="train")
    add_launcher_args(parser)
    args = parser.parse_args()
    if min(args.num_envs, args.episodes_per_env, args.steps) < 1:
        parser.error("Environment, episode, and step counts must be positive.")
    output = args.output.expanduser().resolve()
    if output.suffix != ".hdf5":
        parser.error("--output must have the .hdf5 extension.")
    if output.exists():
        parser.error(f"Refusing to overwrite {output}")
    cfg = ParticleEnvCfg()
    cfg.scene.num_envs = args.num_envs
    if not 0 < args.body_mass < float("inf"):
        parser.error("Body mass must be finite and positive.")
    cfg.seed = args.seed
    cfg.scene.body5.spawn.mass_props.mass = args.body_mass
    if args.material_friction is not None:
        if not 0 <= args.material_friction < float("inf"):
            parser.error("Material friction must be finite and nonnegative.")
        cfg.scene.sand.spawn.material.friction = args.material_friction
    if args.material_density is not None:
        if not 0 < args.material_density < float("inf"):
            parser.error("Material density must be finite and positive.")
        cfg.scene.sand.spawn.material.density = args.material_density
    cfg.sim.physics.solver_cfg.entries[1].solver_cfg.max_iterations = args.mpm_iterations
    cfg.recorders.dataset_export_dir_path = str(output.parent)
    cfg.recorders.dataset_filename = output.stem
    # Native sparse allocations remain dynamic for this correctness experiment.
    # CUDA graph capture and collection throughput are separate optimization work.
    repo = Path(__file__).resolve().parents[2]
    manifest = {
        "schema_version": 2,
        "split": args.split,
        "seed": args.seed,
        "isaaclab_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        "isaaclab_dirty_files": subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=repo, text=True
        ).splitlines(),
        "packages": {
            name: {
                "version": importlib.metadata.version(name),
                "direct_url": json.loads(importlib.metadata.distribution(name).read_text("direct_url.json") or "null"),
            }
            for name in ("newton", "warp-lang", "torch")
        },
        "dt": cfg.sim.dt,
        "gravity": list(cfg.sim.gravity),
        "voxel_size": VOXEL_SIZE,
        "max_iterations": cfg.sim.physics.solver_cfg.entries[1].solver_cfg.max_iterations,
        "tolerance": cfg.sim.physics.solver_cfg.entries[1].solver_cfg.tolerance,
        "collider_velocity_mode": "forward",
        "material": {
            name: getattr(cfg.scene.sand.spawn.material, name)
            for name in (
                "density",
                "young_modulus",
                "poisson_ratio",
                "friction",
                "yield_pressure",
                "tensile_yield_ratio",
                "yield_stress",
                "hardening",
                "dilatancy",
                "viscosity",
            )
        },
        "units": "m, s, kg, N s, N m s",
        "quaternion_order": "xyzw",
        "positions": "environment-relative; axes aligned with simulation world",
        "body_impulse": "material on body, torque impulse about body center; transition t to t+1",
        "motion": "dynamic randomized box; MJWarp-MPM lagged two-way coupling",
        "coupling_iterations": 1,
        "rigid_substeps": 1,
        "applied_body_impulse": "previous MPM feedback applied during current rigid transition",
        "initial_pose_offset_ranges": cfg.events.randomize_box.params["pose_range"],
        "initial_velocity_ranges": cfg.events.randomize_box.params["velocity_range"],
        "body_mass_kg": cfg.scene.body5.spawn.mass_props.mass,
        "moving_object_id": MOVING_BODY_ID,
        "complete": False,
    }
    start = time.perf_counter()
    with launch_simulation(cfg, args):
        env = ManagerBasedEnv(cfg)
        try:
            for episode in range(args.episodes_per_env):
                env.reset()
                for step in range(args.steps):
                    env.step(torch.empty((args.num_envs, 0), device=env.device))
                env.recorder_manager.export_episodes()
                print(f"Recorded batch {episode + 1}/{args.episodes_per_env}: {args.num_envs} episodes", flush=True)
        finally:
            env.close()
    manifest["elapsed_seconds_including_initialization"] = time.perf_counter() - start
    manifest["complete"] = True
    with h5py.File(output, "a") as dataset:
        dataset.attrs["particle_nerd_manifest"] = json.dumps(manifest)
    from dataset import validate_dataset

    print(json.dumps(validate_dataset(output), indent=2))


if __name__ == "__main__":
    main()
