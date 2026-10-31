# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Open-loop ParticleTransolver rollout evaluation on recorded rigid-body motion."""

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch

from particle_nerd.transolver.inference import Predictor, manifest_colliders
from particle_nerd.transolver.train import PositionEpisodes

TEST_IDS = [117, 453, 456, 460, 519, 527, 575, 578, 689, 850, 883, 885]
VALIDATION_IDS = [10, 423, 560, 654, 717, 768]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--split", choices=("validation", "test"), default="test")
    p.add_argument("--ids", nargs="+", type=int)
    p.add_argument("--all-episodes", action="store_true", help="Evaluate every episode in the selected split")
    p.add_argument("--steps", type=int, default=250)
    p.add_argument("--mode", choices=("all",), default="all")
    p.add_argument("--active-k", type=int, default=None, help="Optional active particle count")
    args = p.parse_args()
    if args.ids and args.all_episodes:
        p.error("Choose --ids or --all-episodes")
    data = PositionEpisodes(args.data, args.split)
    ids = (
        [r["index"] for r in data.records]
        if args.all_episodes
        else args.ids or (TEST_IDS if args.split == "test" else VALIDATION_IDS)
    )
    chosen = [(i, r) for i, r in enumerate(data.records) if r["index"] in ids]
    if set(ids) != {r["index"] for _, r in chosen}:
        raise ValueError("Requested evaluation IDs missing from selected split")
    saved = torch.load(args.checkpoint, map_location="cuda:0", weights_only=False)
    args.output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(90518)
    rows = []
    for index, record in chosen:
        with h5py.File(record["path"], "r") as f:
            m = json.loads(f.attrs["manifest"])
            frames = f["frame_steps"][:]
            start = int(np.searchsorted(frames, m["settled_step"]))
            end = min(start + args.steps, len(frames) - 1)
            sample = data.sample(rng, episode=index, frame=start)
            if not np.allclose(np.diff(frames) * m["dt"], saved["config"]["dt"]):
                raise ValueError("Recorded particle interval differs from checkpoint prediction timestep")
            colliders = manifest_colliders(m, "cuda:0")
            predictor = Predictor(
                saved,
                sample["position_history"],
                sample["material"],
                *colliders,
                geometry=m.get("geometry", "box"),
                active_k=args.active_k,
            )
            pose = torch.as_tensor(f["body_pose"][:], device="cuda:0")
            velocity = torch.as_tensor(f["body_velocity"][:], device="cuda:0")
            initial = f["particle_position"][start]
            reference = f["particle_position"][start : end + 1]
            stationary = np.linalg.norm(reference - initial[None], axis=-1).max(0) < 0.001
            errors, active, below, gpu_times, wall_times = [], [], [], [], []
            max_drift = np.zeros(len(initial), dtype=np.float32)
            failure = None
            with h5py.File(args.output / f"rollout-{record['index']:05d}.hdf5", "w") as out:
                ds = out.create_dataset(
                    "particle_position", shape=(end - start + 1, len(initial), 3), dtype="f4", compression="lzf"
                )
                ds[0] = initial
                out.create_dataset("body_pose", data=f["body_pose"][:])
                out.create_dataset("frame_steps", data=frames[start : end + 1])
                for t in range(start, end):
                    torch.cuda.synchronize()
                    begin, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    wall = time.perf_counter()
                    begin.record()
                    predictor.step(pose[frames[t]], velocity[frames[t]])
                    end_event.record()
                    end_event.synchronize()
                    wall_times.append((time.perf_counter() - wall) * 1000)
                    gpu_times.append(begin.elapsed_time(end_event))
                    x = predictor.x.cpu().numpy()
                    if not np.isfinite(x).all():
                        failure = dict(frame=t - start + 1, reason="nonfinite particles")
                        break
                    ds[t - start + 1] = x
                    max_drift = np.maximum(max_drift, np.linalg.norm(x - initial, axis=-1))
                    errors.append(float(np.sqrt(np.mean(np.sum((x - reference[t - start + 1]) ** 2, -1)))))
                    active.append(predictor.diagnostics()["active_fraction"])
                    below.append(float(np.mean(x[:, 2] < -0.04)))
                out.attrs["start_frame"] = start
                out.attrs["valid_frames"] = len(errors) + 1
                out.attrs["reference_episode"] = record["index"]
                out.attrs["evaluation"] = saved["architecture"] + "; predicted particles; recorded rigid-body motion"
                final = ds[len(errors)]
                truth = reference[len(errors)]
                displaced = np.linalg.norm(truth - initial, axis=-1) > 0.01
                displaced_rmse = (
                    float(np.sqrt(np.mean(np.sum((final[displaced] - truth[displaced]) ** 2, -1))))
                    if displaced.any()
                    else None
                )
                row = dict(
                    index=record["index"],
                    case=record["case"],
                    frames=len(errors),
                    failure=failure,
                    prediction_dt_s=float(saved["config"]["dt"]),
                    elapsed_prediction_s=len(errors) * float(saved["config"]["dt"]),
                    half_second_position_rmse_m=(
                        errors[round(0.5 / saved["config"]["dt"]) - 1]
                        if len(errors) >= round(0.5 / saved["config"]["dt"])
                        else None
                    ),
                    final_position_rmse_m=errors[-1] if errors else None,
                    moving_final_rmse_m=displaced_rmse,
                    moving_final_particle_count=int(displaced.sum()),
                    frozen_final_position_rmse_m=float(np.sqrt(np.mean(np.sum((initial - truth) ** 2, -1)))),
                    frozen_moving_final_rmse_m=(
                        float(np.sqrt(np.mean(np.sum((initial[displaced] - truth[displaced]) ** 2, -1))))
                        if displaced.any()
                        else None
                    ),
                    stationary_max_drift_over_1cm=int(np.sum(stationary & (max_drift > 0.01))),
                    stationary_count=int(stationary.sum()),
                    stationary_moved_over_1cm=int(
                        np.sum(stationary & (np.linalg.norm(final - initial, axis=-1) > 0.01))
                    ),
                    mean_active_fraction=float(np.mean(active)) if active else None,
                    position_rmse_m=errors,
                    max_fraction_below_ground_by_more_than_grid=max(below, default=0),
                    median_particle_step_gpu_ms=float(np.median(gpu_times[3:])) if len(gpu_times) > 3 else None,
                    median_particle_step_wall_ms=float(np.median(wall_times[3:])) if len(wall_times) > 3 else None,
                )
            rows.append(row)
            print(json.dumps({k: v for k, v in row.items() if k != "position_rmse_m"}), flush=True)
            result = dict(
                architecture=saved["architecture"],
                config=saved["config"],
                checkpoint=str(args.checkpoint),
                checkpoint_step=saved["step"],
                split=args.split,
                mode=args.mode,
                ids=ids,
                rollouts=rows,
            )
            (args.output / "evaluation.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
