# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Separate environment batching from total particle workload in Transolver."""

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch
import warp as wp

from . import io, sdf
from .features import FeatureBuffers
from .model import ParticleTransolver


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--envs", type=int, nargs="+", default=[1, 8, 32])
    parser.add_argument("--particles", type=int, nargs="+", default=[512, 2048, 16384])
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    saved, manifest, initial = io.load(args.checkpoint, args.episode)
    device = "cuda:0"
    wp.init()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    warp_stream = wp.stream_from_torch(stream)
    rows = []
    with (
        torch.cuda.stream(stream),
        wp.ScopedStream(warp_stream),
        torch.inference_mode(),
        torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False),
    ):
        history = torch.as_tensor(initial["history"], device=device)
        kinds = torch.tensor(
            [{"box": 1, "sphere": 2, "shovel": 3}[manifest.get("geometry", "box")], 0], dtype=torch.int32, device=device
        )
        sizes = torch.as_tensor(
            np.array([np.array(manifest["body_size_m"]) / 2, [0, 0, 0]], dtype=np.float32), device=device
        )
        friction = torch.tensor([manifest.get("body_contact_friction", 0.6), 0.6], device=device)
        features = FeatureBuffers(
            history,
            torch.as_tensor(initial["material"], device=device),
            kinds,
            sizes,
            friction,
            sdf.tensors(device, manifest.get("geometry", "box")),
            saved["config"]["dt"],
            **saved["config"]["features"],
        )
        features.build(
            wp.array(initial["body_pose"], dtype=wp.transform, device=device),
            wp.array(initial["body_velocity"], dtype=wp.spatial_vector, device=device),
        )
        model = ParticleTransolver(**saved["config"]["model"]).to(device).eval()
        model.load_state_dict(saved["model"])
        if args.compile:
            model = torch.compile(model, fullgraph=True, dynamic=False, options={"triton.cudagraphs": False})
        for particles in args.particles:
            if not 0 < particles <= history.shape[1]:
                raise ValueError("Particle count must be within the loaded episode")
            for environments in args.envs:
                inputs = tuple(
                    value[:, :particles].repeat(environments, *([1] * (value.ndim - 1))).contiguous()
                    for value in features.inputs
                )
                for _ in range(3):
                    model(*inputs)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    output = model(*inputs)
                for _ in range(20):
                    graph.replay()
                torch.cuda.synchronize()
                durations = []
                for _ in range(60):
                    start = time.perf_counter()
                    graph.replay()
                    torch.cuda.synchronize()
                    durations.append(1000 * (time.perf_counter() - start))
                config = saved["config"]["model"]
                row = dict(
                    environments=environments,
                    particles_per_environment=particles,
                    total_particles=environments * particles,
                    wall_median_ms=float(np.median(durations)),
                    wall_p95_ms=float(np.percentile(durations, 95)),
                    fp32_slice_weight_MiB=environments * config["heads"] * particles * config["slices"] * 4 / 2**20,
                    raw_wall_ms=durations,
                    finite=bool(torch.isfinite(output).all()),
                )
                rows.append(row)
                print({k: v for k, v in row.items() if k != "raw_wall_ms"}, flush=True)
                del graph, output, inputs
                gc.collect()
        result = dict(
            gpu=torch.cuda.get_device_name(),
            torch_version=torch.__version__,
            compiled=args.compile,
            cuda_graph=True,
            scope="NN only, fixed recorded input subset; not a physics accuracy evaluation",
            measurements=rows,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
