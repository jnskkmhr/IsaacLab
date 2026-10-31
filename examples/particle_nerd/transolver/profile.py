# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Warm GPU and synchronized wall timings; no rendering or MPM solve."""

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch

from particle_nerd.transolver.features import material_features
from particle_nerd.transolver.features import features
from particle_nerd.transolver.inference import Predictor, box_ground_state, manifest_colliders


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--episode", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, default=50)
    args = p.parse_args()
    if args.steps < 1:
        p.error("--steps must be positive")
    saved = torch.load(args.checkpoint, map_location="cuda:0", weights_only=False)
    with h5py.File(args.episode) as f:
        m = json.loads(f.attrs["manifest"])
        frame = int(np.searchsorted(f["frame_steps"][:], m["settled_step"]))
        h = saved["config"]["model"]["history"]
        history = torch.as_tensor(
            np.stack([f["particle_position"][max(0, frame - k)] for k in reversed(range(h))], axis=1), device="cuda:0"
        )
        step = int(f["frame_steps"][frame])
        pose, velocity = [torch.as_tensor(f[k][step], device="cuda:0") for k in ("body_pose", "body_velocity")]
        material = torch.as_tensor(
            material_features(m, float(f["particle_mass"][0]), float(f["particle_radius"][0])), device="cuda:0"
        )
    colliders = manifest_colliders(m, "cuda:0")
    predictor = Predictor(saved, history, material, *colliders)
    poses, velocities = box_ground_state(pose, velocity)

    def geometry():
        return features(
            history,
            poses,
            velocities,
            *colliders,
            material,
            predictor.dt,
            **predictor.config["features"],
            sdf=predictor.sdf,
        )

    inputs = geometry()

    def network():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return predictor.model(*inputs)

    rows = {}
    with torch.no_grad():
        for name, operation in [
            ("features", geometry),
            ("neural_network", network),
            ("whole_particle_step", lambda: predictor.step(pose, velocity)),
        ]:
            for _ in range(5):
                operation()
            gpu, wall = [], []
            for _ in range(args.steps):
                begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()
                start = time.perf_counter()
                begin.record()
                operation()
                end.record()
                end.synchronize()
                gpu.append(begin.elapsed_time(end))
                wall.append((time.perf_counter() - start) * 1000)
            rows[name] = dict(median_gpu_ms=float(np.median(gpu)), median_wall_ms=float(np.median(wall)))
    result = dict(
        gpu=torch.cuda.get_device_name(),
        particles=len(history),
        colliders=len(colliders[0]),
        configuration=predictor.config,
        captured=False,
        compiled=False,
        precision="Torch BF16 autocast; Warp FP32",
        measured_steps=args.steps,
        timings=rows,
        caveat=(
            "Compute timing only; checkpoint quality is not established. "
            "No force estimator, rigid solver, or rendering included."
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
