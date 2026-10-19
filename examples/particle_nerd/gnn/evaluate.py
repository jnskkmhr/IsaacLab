# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Held-out one-step and recurrent particle rollout evaluation without MPM.

Box motion is replayed from reference data to isolate particle-model error.
This is explicitly not a closed-loop body-force or box-motion evaluation.
"""

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch

from particle_nerd.gnn.data import Episodes
from particle_nerd.gnn.features import Features
from particle_nerd.gnn.inference import ParticlePredictor
from particle_nerd.gnn.model import ParticleGNN
from particle_nerd.gnn.train import evaluate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--episodes", type=int, default=12)
    p.add_argument("--steps", type=int, default=250)
    p.add_argument("--compile", action="store_true")
    p.add_argument("--split", choices=("validation", "test"), default="test")
    p.add_argument(
        "--exclude-episode",
        type=int,
        nargs="*",
        default=[2],
        help="Episode 2 was inspected during development; exclude it from final test results",
    )
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    saved = torch.load(args.checkpoint, map_location="cuda", weights_only=False)
    model = ParticleGNN(**saved["config"]).cuda().eval()
    model.load_state_dict(saved["model"])
    episodes = Episodes(args.data, args.split, model.config["history"])
    if args.split == "test":
        episodes.records = [r for r in episodes.records if r["index"] not in args.exclude_episode]
    if not episodes.records:
        raise ValueError("No evaluation episodes remain after development exclusions")
    features = Features(16384, history=model.config["history"])
    one_step = evaluate(model, episodes, features, samples=48)
    del features, model
    rows = []
    for record in episodes.records[: args.episodes]:
        with h5py.File(record["path"], "r") as f:
            manifest = json.loads(f.attrs["manifest"])
            steps = f["frame_steps"][:]
            start = int(np.searchsorted(steps, manifest["settled_step"]))
            stop = min(start + args.steps, len(steps) - 1)
            predictor = ParticlePredictor(
                args.checkpoint,
                f["particle_position"][start],
                f["particle_velocity"][start],
                manifest,
                float(f["particle_mass"][0]),
                float(f["particle_radius"][0]),
                compile_model=args.compile,
            )
            predictor.history.copy_(
                torch.as_tensor(
                    np.concatenate(
                        [
                            f["particle_velocity"][max(0, start - k)]
                            for k in range(predictor.model.config["history"] - 1, -1, -1)
                        ],
                        axis=-1,
                    ),
                    device="cuda",
                )
            )
            poses = torch.as_tensor(f["body_pose"][:], device="cuda")
            velocities = torch.as_tensor(f["body_velocity"][:], device="cuda")
            # Recording/evaluation synchronizes intentionally; predictor.step does not.
            positions = [predictor.x.cpu().numpy().copy()]
            errors = []
            frozen_errors = []
            ground_violations = []
            edge_counts = []
            timings = []
            wall_timings = []
            failure = None
            displaced_count = 0
            displaced_error = None
            frozen_displaced_error = None
            with h5py.File(args.output / f"rollout-{record['index']:05d}.hdf5", "w") as out:
                ds = out.create_dataset(
                    "particle_position",
                    (stop - start + 1, 16384, 3),
                    dtype="f4",
                    compression="lzf",
                )
                ds[0] = positions[0]
                for i in range(start, stop):
                    begin, end = (
                        torch.cuda.Event(enable_timing=True),
                        torch.cuda.Event(enable_timing=True),
                    )
                    wall_start = time.perf_counter()
                    begin.record()
                    predictor.step(poses[steps[i]], velocities[steps[i]])
                    end.record()
                    end.synchronize()
                    wall_timings.append(1000 * (time.perf_counter() - wall_start))
                    timings.append(begin.elapsed_time(end))
                    diagnostic = predictor.diagnostics()
                    if diagnostic["graph_overflow"] or not diagnostic["finite"]:
                        failure = dict(frame=i - start + 1, **diagnostic)
                        break
                    x = predictor.x.cpu().numpy()
                    edge_counts.append(diagnostic["edge_count"])
                    ground_violations.append(float(np.mean(x[:, 2] < -0.04)))
                    frozen_errors.append(
                        float(
                            np.sqrt(
                                np.mean(
                                    np.sum(
                                        (positions[0] - f["particle_position"][i + 1]) ** 2,
                                        axis=-1,
                                    )
                                )
                            )
                        )
                    )
                    ds[i - start + 1] = x
                    errors.append(
                        float(
                            np.sqrt(
                                np.mean(
                                    np.sum(
                                        (x - f["particle_position"][i + 1]) ** 2,
                                        axis=-1,
                                    )
                                )
                            )
                        )
                    )
                out.attrs["valid_frames"] = len(errors) + 1
                out.attrs["reference_episode"] = record["index"]
                out.attrs["start_frame"] = start
                out.attrs["evaluation"] = "MPM-free particles; reference rigid motion replay"
                if errors:
                    reference_final = f["particle_position"][start + len(errors)]
                    displaced = np.linalg.norm(reference_final - positions[0], axis=1) > 0.01
                    displaced_count = int(displaced.sum())
                    if displaced_count:
                        displaced_error = float(
                            np.sqrt(
                                np.mean(
                                    np.sum(
                                        (ds[len(errors)][displaced] - reference_final[displaced]) ** 2,
                                        axis=1,
                                    )
                                )
                            )
                        )
                        frozen_displaced_error = float(
                            np.sqrt(
                                np.mean(
                                    np.sum(
                                        (positions[0][displaced] - reference_final[displaced]) ** 2,
                                        axis=1,
                                    )
                                )
                            )
                        )
            row = dict(
                index=record["index"],
                case=record["case"],
                frames=len(errors),
                failure=failure,
                reference_displaced_particle_count_final=displaced_count,
                displaced_particle_final_position_rmse_m=displaced_error,
                frozen_displaced_particle_final_position_rmse_m=frozen_displaced_error,
                position_rmse_m=errors,
                final_position_rmse_m=errors[-1] if errors else None,
                frozen_particle_final_position_rmse_m=frozen_errors[-1] if frozen_errors else None,
                max_fraction_below_ground_by_more_than_grid=max(ground_violations, default=0.0),
                min_directed_edges=min(edge_counts, default=0),
                max_directed_edges=max(edge_counts, default=0),
                median_particle_step_gpu_ms=float(np.median(timings[3:])) if len(timings) > 3 else None,
                median_particle_step_wall_ms=float(np.median(wall_timings[3:])) if len(wall_timings) > 3 else None,
                max_particle_step_gpu_ms=float(max(timings)) if timings else None,
            )
            rows.append(row)
            print(
                json.dumps({k: v for k, v in row.items() if k != "position_rmse_m"}),
                flush=True,
            )
    result = dict(
        device=torch.cuda.get_device_name(),
        torch_version=torch.__version__,
        timing_warmup_steps=3,
        compiled_network=args.compile,
        split=args.split,
        eligible_episodes=len(episodes.records),
        development_exclusions=args.exclude_episode if args.split == "test" else [],
        one_step=one_step,
        rollouts=rows,
        checkpoint_step=saved["step"],
        caveat=(
            "Box follows held-out MPM recording; these results do not establish coupled force or rigid-motion accuracy"
        ),
    )
    (args.output / "evaluation.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
