# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Comprehensive single-GPU batch scaling benchmark for Particle NeRD."""

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch

from .simulation import Simulation


def measure_config(
    checkpoint: Path,
    episode: Path,
    envs: int,
    active_particles: int | None,
    force_model: str,
    compile_model: bool,
    warmup: int = 20,
    steps: int = 60,
) -> dict:
    if compile_model:
        torch._dynamo.reset()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    sim = Simulation.from_files(
        checkpoint,
        episode,
        environments=envs,
        active_particles=active_particles,
        force_model=force_model,
        compile_model=compile_model,
        capture=True,
    )
    for _ in range(warmup):
        sim.step()
    torch.cuda.synchronize()
    durations = []
    for _ in range(steps):
        t0 = time.perf_counter()
        sim.step()
        torch.cuda.synchronize()
        durations.append(1000.0 * (time.perf_counter() - t0))
    health = sim.diagnostics()
    peak_mib = float(torch.cuda.max_memory_allocated() / (1024.0 * 1024.0))
    median_ms = float(np.median(durations))
    p95_ms = float(np.percentile(durations, 95))
    row = dict(
        environments=envs,
        particles_per_environment=sim.particles,
        active_particles=sim.features.active_k,
        force_model=health["force_model"],
        compiled=compile_model,
        wall_median_ms=median_ms,
        wall_p95_ms=p95_ms,
        ms_per_env_step=median_ms / envs,
        env_steps_per_second=1000.0 * envs / median_ms,
        realtime_factor_total=(20.0 * envs) / median_ms,
        peak_allocated_mib=peak_mib,
        finite=health["finite"],
        converged=health["converged"],
        rigid_overflow=health["rigid_overflow"],
    )
    del sim
    gc.collect()
    torch.cuda.empty_cache()
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--episode", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--steps", type=int, default=60)
    args = p.parse_args()

    torch.set_num_threads(1)
    configs = [
        # Baseline full 16384 + analytical contact solver
        ("full_16384_analytical", None, "analytical", False, [1, 8, 32]),
        # Full 16384 + neural contact head (no analytical solver)
        ("full_16384_neural_contact", None, "neural_contact", False, [1, 8, 32]),
        # Active-1024 + neural contact head (uncompiled)
        ("active_1024_neural_contact", 1024, "neural_contact", False, [1, 8, 32, 64, 128, 256]),
        # Active-1024 + neural contact head (compiled)
        ("active_1024_neural_contact_compiled", 1024, "neural_contact", True, [1, 8, 32, 64, 128, 256]),
        # Active-512 + neural contact head (compiled)
        ("active_512_neural_contact_compiled", 512, "neural_contact", True, [1, 8, 32, 64, 128, 256]),
    ]
    results = {}
    for label, active_k, force_mode, compiled, env_list in configs:
        rows = []
        for envs in env_list:
            try:
                row = measure_config(
                    args.checkpoint,
                    args.episode,
                    envs=envs,
                    active_particles=active_k,
                    force_model=force_mode,
                    compile_model=compiled,
                    warmup=args.warmup,
                    steps=args.steps,
                )
                rows.append(row)
                print(label, json.dumps(row), flush=True)
            except Exception as exc:
                print(label, f"envs={envs} failed: {exc}", flush=True)
                rows.append(dict(environments=envs, error=str(exc)))
                break
        results[label] = rows

    payload = dict(
        gpu=torch.cuda.get_device_name(),
        torch_version=torch.__version__,
        checkpoint=str(args.checkpoint),
        episode=str(args.episode),
        warmup=args.warmup,
        steps=args.steps,
        results=results,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print("BENCHMARK_COMPLETE", args.output, flush=True)


if __name__ == "__main__":
    main()
