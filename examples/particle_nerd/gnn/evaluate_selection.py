# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compare unrestricted, reference-label and learned selection with fixed motion weights.

Reference activity is explicitly an oracle diagnostic, never a deployable input.
The full graph remains present: this experiment measures accuracy, not sparsity speedup.
"""

import argparse
import gc
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch

from particle_nerd.gnn.data import Episodes
from particle_nerd.gnn.region_selection import SelectedPredictor, reference_activity

PREVIOUS_TEST_IDS = [2, 288, 352, 704, 800, 896, 960, 193, 257, 385, 577, 234, 330]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--selector", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--split", choices=["validation", "test"], default="validation")
    p.add_argument(
        "--modes",
        nargs="+",
        choices=["all", "oracle", "learned", "velocity", "frozen"],
        default=["all", "oracle"],
    )
    p.add_argument("--episodes", type=int, default=6)
    p.add_argument("--ids", nargs="*", type=int)
    p.add_argument("--label-distance", type=float, default=0.0005)
    p.add_argument("--label-horizon", type=int, default=5)
    p.add_argument("--compile", action="store_true")
    p.add_argument("--steps", type=int, default=250)
    p.add_argument("--allow-previous-test", action="store_true")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    records = Episodes(args.data, args.split).records
    if args.split == "test" and not args.allow_previous_test:
        records = [r for r in records if r["index"] not in PREVIOUS_TEST_IDS]
    if args.ids is not None:
        records = [r for r in records if r["index"] in args.ids]
    records = records[: args.episodes]
    if not records:
        raise ValueError("No eligible trajectories")
    results = []
    for record in records:
        with h5py.File(record["path"]) as ref:
            manifest = json.loads(ref.attrs["manifest"])
            steps = ref["frame_steps"][:]
            start = int(np.searchsorted(steps, manifest["settled_step"]))
            stop = min(start + args.steps, len(steps) - 1)
            reference = ref["particle_position"][start : stop + 1]
            initial = reference[0]
            quiet = np.linalg.norm(reference - initial, axis=-1).max(0) < 0.001
            moving_final = np.linalg.norm(reference[-1] - initial, axis=-1) > 0.01
            masks = np.stack(
                [reference_activity(ref, k, args.label_horizon, args.label_distance) for k in range(start, stop)]
            )
            for mode in args.modes:
                predictor = SelectedPredictor(
                    args.checkpoint,
                    initial,
                    ref["particle_velocity"][start],
                    manifest,
                    float(ref["particle_mass"][0]),
                    float(ref["particle_radius"][0]),
                    compile_model=args.compile,
                    selector_checkpoint=args.selector,
                    mode=mode,
                    speed_threshold=args.label_distance / (args.label_horizon * 0.02),
                )
                predictor.history.copy_(
                    torch.as_tensor(
                        np.concatenate(
                            [ref["particle_velocity"][max(0, start - k)] for k in (2, 1, 0)],
                            axis=-1,
                        ),
                        device="cuda",
                    )
                )
                if mode == "learned" and (
                    predictor.label_horizon != args.label_horizon or predictor.label_distance != args.label_distance
                ):
                    raise ValueError("Evaluation labels do not match selector training")
                poses = torch.as_tensor(ref["body_pose"][:], device="cuda")
                velocities = torch.as_tensor(ref["body_velocity"][:], device="cuda")
                oracle = torch.as_tensor(masks, device="cuda", dtype=torch.int32)
                errors = []
                fractions = []
                timings = []
                counts = np.zeros(4, dtype=np.int64)
                max_displacement = np.zeros(len(initial))
                failure = None
                ground = []
                target = args.output / f"{mode}-{record['index']:05d}.hdf5"
                with h5py.File(target, "w") as out:
                    ds = out.create_dataset(
                        "particle_position",
                        (stop - start + 1, len(initial), 3),
                        dtype="f4",
                        compression="lzf",
                    )
                    ds[0] = initial
                    for j, k in enumerate(range(start, stop)):
                        begin = time.perf_counter()
                        predictor.step(
                            poses[steps[k]],
                            velocities[steps[k]],
                            active_mask=oracle[j] if mode == "oracle" else None,
                        )
                        torch.cuda.synchronize()
                        timings.append(1000 * (time.perf_counter() - begin))
                        diagnostic = predictor.diagnostics()
                        if diagnostic["graph_overflow"] or not diagnostic["finite"]:
                            failure = dict(step=j + 1, **diagnostic)
                            break
                        x = predictor.x.cpu().numpy()
                        ds[j + 1] = x
                        active = predictor.active.cpu().numpy().astype(bool)
                        truth = masks[j]
                        counts += [
                            np.sum(active & truth),
                            np.sum(active & ~truth),
                            np.sum(~active & truth),
                            np.sum(~active & ~truth),
                        ]
                        errors.append(float(np.sqrt(np.mean(np.sum((x - reference[j + 1]) ** 2, axis=-1)))))
                        fractions.append(float(active.mean()))
                        ground.append(float((x[:, 2] < -0.04).mean()))
                        max_displacement = np.maximum(max_displacement, np.linalg.norm(x - initial, axis=-1))
                    out.attrs.update(
                        valid_frames=len(errors) + 1,
                        start_frame=start,
                        reference_episode=record["index"],
                        evaluation=f"{mode} selection; full graph; reference box motion",
                    )
                    final = ds[len(errors)]
                tp, fp, fn, tn = counts
                finished = len(errors) == stop - start
                rows = dict(
                    index=record["index"],
                    case=record["case"],
                    mode=mode,
                    speed_threshold=args.label_distance / (args.label_horizon * 0.02),
                    frames=len(errors),
                    failure=failure,
                    position_rmse_m=errors,
                    final_position_rmse_m=errors[-1] if errors else None,
                    frozen_final_position_rmse_m=float(
                        np.sqrt(np.mean(np.sum((reference[len(errors)] - initial) ** 2, axis=-1)))
                    ),
                    moving_final_particle_count=int(moving_final.sum()),
                    moving_final_rmse_m=float(
                        np.sqrt(
                            np.mean(
                                np.sum(
                                    (final[moving_final] - reference[-1, moving_final]) ** 2,
                                    axis=-1,
                                )
                            )
                        )
                    )
                    if moving_final.any() and finished
                    else None,
                    frozen_moving_final_rmse_m=float(
                        np.sqrt(
                            np.mean(
                                np.sum(
                                    (initial[moving_final] - reference[-1, moving_final]) ** 2,
                                    axis=-1,
                                )
                            )
                        )
                    )
                    if moving_final.any()
                    else None,
                    reference_stationary_particles=int(quiet.sum()),
                    stationary_particles_moved_over_1cm=int((quiet & (max_displacement > 0.01)).sum()),
                    mean_active_fraction=float(np.mean(fractions)),
                    activity_precision=float(tp / max(1, tp + fp)),
                    activity_recall=float(tp / max(1, tp + fn)),
                    false_positive_rate=float(fp / max(1, fp + tn)),
                    max_fraction_below_ground_by_004m=max(ground, default=0),
                    median_particle_step_wall_ms=float(np.median(timings[3:])) if len(timings) > 3 else None,
                )
                results.append(rows)
                print(
                    json.dumps({k: v for k, v in rows.items() if k != "position_rmse_m"}),
                    flush=True,
                )
                del predictor
                gc.collect()
                torch.cuda.empty_cache()
                report = dict(
                    split=args.split,
                    device=torch.cuda.get_device_name(),
                    baseline_checkpoint=str(args.checkpoint),
                    selector_checkpoint=str(args.selector),
                    label_horizon=args.label_horizon,
                    label_distance_m=args.label_distance,
                    compiled=args.compile,
                    excluded_previous_test_ids=[] if args.allow_previous_test else PREVIOUS_TEST_IDS,
                    computation="Full graph preserved; no neural compute reduction claimed",
                    box_motion="Reference recording; no force estimator",
                    rollouts=results,
                )
                (args.output / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output / "COMPLETE.json").write_text(json.dumps(dict(trajectories=len(records), modes=args.modes)) + "\n")


if __name__ == "__main__":
    main()
