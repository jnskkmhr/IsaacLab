# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Plot matched G1 motion and sand-on-foot forces; report comparison scope."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--prediction", type=Path, required=True)
    p.add_argument("--particle-evaluation", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    ref, pred = [dict(np.load(x / "trace.npz")) for x in [args.reference, args.prediction]]
    metadata = [json.loads((x / "metadata.json").read_text()) for x in [args.reference, args.prediction]]
    starts = [np.load(x / "snapshots.npz") for x in [args.reference, args.prediction]]
    for key in ["particle_position", "body_pose"]:
        np.testing.assert_allclose(
            starts[0][key][0],
            starts[1][key][0],
            atol=1e-6,
            rtol=0,
            err_msg=f"Initial {key} mismatch; not a matched comparison",
        )
    for s in starts:
        s.close()
    count = min(len(ref["time_s"]), len(pred["time_s"]))
    np.testing.assert_allclose(ref["time_s"][:count], pred["time_s"][:count])
    dx = pred["root_pose"][:count, :3] - ref["root_pose"][:count, :3]
    result = dict(
        scope="closed-loop trajectories diverge; force-profile differences are not force-reconstruction error",
        reference_outcome=metadata[0]["outcome"],
        prediction_outcome=metadata[1]["outcome"],
        reference_duration_s=metadata[0]["duration_s"],
        prediction_duration_s=metadata[1]["duration_s"],
        common_duration_s=float(ref["time_s"][count - 1]),
        initial_positions_match=True,
        pelvis_position_rmse_m=float(np.sqrt((dx**2).sum(1).mean())),
        final_pelvis_error_m=float(np.linalg.norm(dx[-1])),
        reference_final_height_m=float(ref["root_pose"][count - 1, 2]),
        prediction_final_height_m=float(pred["root_pose"][count - 1, 2]),
        learned_material_steps=metadata[1]["learned_steps"],
        estimator=metadata[1]["estimator"],
    )
    args.output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for trace, label, color in [
        (ref, "MPM reference", "#365e7c"),
        (pred, "Particle NeRD + analytical forces", "#bc703d"),
    ]:
        xyz, t = trace["root_pose"][:, :3], trace["time_s"]
        axes[0, 0].plot(xyz[:, 0], xyz[:, 1], label=label, color=color)
        axes[0, 1].plot(t, xyz[:, 2], color=color)
        for foot, ax in enumerate(axes[1]):
            ax.plot(t, trace["foot_wrench"][:, foot, 2], color=color, lw=1)
    axes[0, 0].set(xlabel="Pelvis x [m]", ylabel="Pelvis y [m]", title="Horizontal robot trajectory")
    axes[0, 0].axis("equal")
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].set(xlabel="Time [s]", ylabel="Pelvis height above initial sand surface [m]")
    for foot, ax in enumerate(axes[1]):
        ax.set(xlabel="Time [s]", ylabel="Sand force on foot, world z [N]", title=["Left foot", "Right foot"][foot])
    for ax in axes.flat:
        ax.grid(alpha=0.2)
    fig.savefig(args.output / "coupled-comparison.svg")
    plt.close(fig)
    if args.particle_evaluation:
        data = json.loads((args.particle_evaluation / "metrics.json").read_text())
        rows = data["metrics"]
        fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
        t = [r["elapsed_prediction_s"] for r in rows]
        for key, label in [
            ("position_rmse_m", "Particle NeRD"),
            ("frozen_rmse_m", "Frozen initial particles"),
            ("stationary_rmse_m", "Stationary reference particles: NeRD error"),
        ]:
            ax.plot(t, [r[key] for r in rows], label=label)
        ax.set(
            xlabel="Autoregressive prediction time [s]",
            ylabel="Position RMSE [m]",
            title="Recorded robot geometry; predicted particles",
        )
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
        fig.savefig(args.output / "particle-rollout.svg")
        plt.close(fig)
        result["particle_rollout"] = {k: data[k] for k in ["scope", "trial", "split", "outcome"]}
        result["particle_rollout"]["final_metrics"] = rows[-1] if rows else None
    (args.output / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
