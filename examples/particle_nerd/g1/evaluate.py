# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate autoregressive particle motion with recorded robot geometry.

This is a particle-prediction test, not closed-loop robot simulation. Past three
frames initialize history once; later particle predictions feed the next step.
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
import warp as wp

from particle_nerd.g1.data import collider_state, compact_step, read_geometry, resolve_active_k
from particle_nerd.transolver.features import material_features
from particle_nerd.transolver.model import ParticleTransolver


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--episode", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--duration", type=float, default=5.0)
    p.add_argument(
        "--active-k",
        type=str,
        default=None,
        help="Optional active particle count ('auto', 'none', or integer K)",
    )
    p.add_argument("--anchor-m", type=int, default=512)
    p.add_argument(
        "--settled-frame",
        type=int,
        default=None,
        help="Start frame after MPM gravity settling (defaults to checkpoint config or 150)",
    )
    args = p.parse_args()
    torch.set_num_threads(2)
    wp.init()
    device = "cuda:0"
    saved = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = ParticleTransolver(**saved["config"]["model"]).to(device).eval()
    model.load_state_dict(saved["model"])
    dt = saved["config"]["dt"]
    stride = round(dt / 0.005)
    if not np.isclose(dt, stride * 0.005):
        raise ValueError("Model timestep incompatible with records")
    args.output.mkdir(parents=True, exist_ok=True)

    def tensor(a):
        return torch.as_tensor(np.asarray(a), device=device, dtype=torch.float32)

    metrics, snapshots = [], []
    with h5py.File(args.episode, "r") as f, torch.inference_mode():
        manifest = json.loads(f.attrs["manifest"])
        if manifest["split"] == "train":
            raise ValueError("Use validation or test trajectories for evaluation")
        geometry = read_geometry(f, device)
        material = tensor(material_features(manifest, float(f["particle_mass"][0]), float(f["particle_radius"][0]), dt))
        h = saved["config"]["model"]["history"]
        default_settled = saved["config"].get("settled_frame", 0)
        settled_frame = int(default_settled if args.settled_frame is None else args.settled_frame)
        start_frame = max(settled_frame, (h - 1) * stride)
        history = tensor(np.stack([f["particle_position"][start_frame - k * stride] for k in reversed(range(h))], 1))
        initial = history[:, -1].clone()
        last = min(len(f["time_s"]) - 1, start_frame + round(args.duration / 0.005))
        raw_active_k = args.active_k if args.active_k is not None else saved["config"].get("active_k")
        resolved_k = resolve_active_k(raw_active_k, len(history), material)
        anchor_m = int(saved["config"].get("anchor_m", args.anchor_m) or args.anchor_m)
        outcome = "duration"
        for frame in range(start_frame, last - stride + 1, stride):
            pose, velocity = tensor(f["body_pose"][frame]), tensor(f["body_velocity"][frame])
            cp, cv = collider_state(pose, velocity, geometry)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                displacement = compact_step(
                    model,
                    history,
                    cp,
                    cv,
                    geometry,
                    material,
                    dt,
                    support=saved["config"]["features"]["support"],
                    active_k=resolved_k,
                    anchor_m=anchor_m,
                )
                prediction = history[:, -1] + displacement
            ref = tensor(f["particle_position"][frame + stride])
            if not torch.isfinite(prediction).all():
                outcome = "nonfinite_prediction"
                break
            squared = (prediction - ref).square().sum(-1)
            ref_displacement = (ref - initial).square().sum(-1)
            stationary = ref_displacement < 0.001**2
            stationary_error = float(squared[stationary].mean().sqrt()) if stationary.any() else None
            metrics.append(
                dict(
                    time_s=float(f["time_s"][frame + stride]),
                    elapsed_prediction_s=(frame + stride - start_frame) * 0.005,
                    position_rmse_m=float(squared.mean().sqrt()),
                    stationary_rmse_m=stationary_error,
                    frozen_rmse_m=float(ref_displacement.mean().sqrt()),
                    max_error_m=float(squared.max().sqrt()),
                )
            )
            if (frame - start_frame) % max(1, round(0.04 / 0.005)) == 0:
                snapshots.append(
                    dict(
                        time_s=metrics[-1]["time_s"],
                        particle_position=prediction.cpu().numpy(),
                        reference_position=ref.cpu().numpy(),
                        body_pose=f["body_pose"][frame + stride],
                        foot_wrench=f["foot_material_wrench"][frame + stride],
                    )
                )
            history = torch.cat((history[:, 1:], prediction[:, None]), 1)
            if len(metrics) % 100 == 0:
                print("ROLLOUT", json.dumps(metrics[-1]), flush=True)
        report = dict(
            scope="autoregressive particles; recorded reference robot geometry",
            episode=str(args.episode),
            checkpoint=str(args.checkpoint),
            trial=manifest["trial"],
            split=manifest["split"],
            initial_history_frames=h,
            dt=dt,
            outcome=outcome,
            metrics=metrics,
        )
        (args.output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
        if snapshots:
            np.savez(args.output / "snapshots.npz", **{k: np.stack([s[k] for s in snapshots]) for k in snapshots[0]})
        print(
            "PARTICLE_EVALUATION_COMPLETE",
            json.dumps(dict(outcome=outcome, steps=len(metrics), final=metrics[-1] if metrics else None)),
            flush=True,
        )


if __name__ == "__main__":
    main()
