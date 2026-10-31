# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run or benchmark the complete batched Transolver simulation."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .simulation import Simulation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument(
        "--effective-mass-scale",
        type=float,
        help="Auxiliary material mass multiplier (default: 180 for shovel, 10 otherwise)",
    )
    parser.add_argument(
        "--active-particles",
        type=lambda v: "auto" if v.lower() in ("auto", "heuristic") else int(v),
        default=None,
        help="Active particles per environment ('auto' for proximity+material heuristic, or integer cap)",
    )
    parser.add_argument(
        "--force-model",
        choices=("auto", "analytical", "rft", "neural_contact"),
        default="auto",
        help="Select neural contact head, RFT directional contact solver, or analytical contact estimator",
    )
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--no-capture", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.effective_mass_scale is not None and not 0 < args.effective_mass_scale < float("inf"):
        parser.error("--effective-mass-scale must be finite and positive")
    if args.steps < 1 or args.warmup < 0:
        parser.error("Require positive steps and nonnegative warmup")
    torch.set_num_threads(1)
    sim = Simulation.from_files(
        args.checkpoint,
        args.episode,
        args.num_envs,
        compile_model=args.compile,
        capture=not args.no_capture,
        effective_mass_scale=args.effective_mass_scale,
        active_particles=args.active_particles,
        force_model=args.force_model,
    )
    print(
        f"Transolver batch: {sim.environments} worlds × {sim.particles} particles "
        f"(active={sim.features.active_k}, force={'neural' if sim.use_neural_contact else 'analytical'}); "
        f"complete CUDA graph={sim.capture_enabled}",
        flush=True,
    )
    for _ in range(args.warmup):
        sim.step()
    torch.cuda.synchronize()
    durations = []
    for _ in range(args.steps):
        start = time.perf_counter()
        sim.step()
        torch.cuda.synchronize()
        durations.append(1000 * (time.perf_counter() - start))
    health = sim.diagnostics()
    result = dict(
        environments=sim.environments,
        particles_per_environment=sim.particles,
        active_particles_per_environment=sim.features.active_k,
        force_model="neural_contact" if sim.use_neural_contact else "analytical",
        particle_dt_s=sim.dt,
        rigid_dt_s=sim.rigid_dt,
        compiled=sim.compiled,
        effective_mass_scale=None if sim.estimator is None else sim.estimator.scale,
        full_step_cuda_graph=sim.capture_enabled,
        gpu=torch.cuda.get_device_name(),
        wall_median_ms=float(np.median(durations)),
        wall_p95_ms=float(np.percentile(durations, 95)),
        raw_wall_ms=durations,
        diagnostics=health,
        scope="Complete batched simulation; excludes initialization, capture, diagnostics and rendering",
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "raw_wall_ms"}), flush=True)
    if not health["finite"] or not health["converged"] or health["rigid_overflow"]:
        raise RuntimeError("Simulation failed finite/convergence/capacity checks")


if __name__ == "__main__":
    main()
