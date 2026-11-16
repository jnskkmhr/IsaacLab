# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compare closed-loop feedback filtering for the frozen MPM-trained G1 policy."""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CASES = {
    "mpm_tau0": ("Implicit MPM", "#17365D", "-"),
    "nerd_tau0": ("NeRD, unfiltered", "#187B38", "-"),
    "nerd_tau10": ("NeRD, EMA 10 ms", "#92B957", "--"),
    "nerd_tau20": ("NeRD, EMA 20 ms", "#565656", ":"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    traces, metadata, metrics = {}, {}, {}
    for case in CASES:
        traces[case] = dict(np.load(args.data / case / "trace.npz"))
        metadata[case] = json.loads((args.data / case / "metadata.json").read_text())
        trace, meta = traces[case], metadata[case]
        assert all(np.isfinite(v).all() for v in trace.values()), f"Nonfinite trace: {case}"
        assert np.allclose(np.diff(trace["time"]), 0.005), f"Wrong sampling: {case}"
        metrics[case] = {"outcome": meta["outcome"], "duration_s": meta["recorded_seconds"], "windows": {}}
        rigid_contact = np.flatnonzero(np.linalg.norm(trace["rigid_force"], axis=-1).max(axis=1) > 5)
        metrics[case]["first_rigid_contact_above_5N_s"] = (
            float(trace["time"][rigid_contact[0]]) if len(rigid_contact) else None
        )
        alpha = meta["force_filter"]["alpha"]
        # Capture warm-up executes extra passes initially. After capture, the feedback
        # must follow the configured EMA exactly on every physics step, including torque.
        predicted = alpha * trace["raw_sand_wrench"][1:] + (1 - alpha) * trace["feedback_wrench"][:-1]
        errors = trace["feedback_wrench"][1:] - predicted
        mask = trace["time"][1:] > 1
        metrics[case]["ema_max_abs_residual_after_1s"] = float(np.abs(errors[mask]).max())
        assert np.allclose(trace["feedback_wrench"][1:][mask], predicted[mask], atol=0.001, rtol=1e-5), case
        for start, stop in ((1, 4), (4, 10), (10, 20), (20, 30)):
            mask = (trace["time"] >= start) & (trace["time"] < stop)
            if mask.sum() < 2:
                continue
            force = trace["force"][mask]
            metrics[case]["windows"][f"{start}-{stop}s"] = {
                "samples": int(mask.sum()),
                "complete_window": bool(trace["time"][-1] >= stop),
                "mean_Fxyz_N_left_right": force.mean(axis=0).tolist(),
                "std_Fxyz_N_left_right": force.std(axis=0).tolist(),
                "Fxyz_step_change_rms_N_left_right": np.sqrt(np.mean(np.diff(force, axis=0) ** 2, axis=0)).tolist(),
                "raw_Fz_std_N_left_right": trace["raw_sand_wrench"][mask, :, 2].std(axis=0).tolist(),
                "foot_z_excursion_cm_left_right": (100 * np.ptp(trace["foot_position"][mask, :, 2], axis=0)).tolist(),
                "foot_speed_rms_m_s_left_right": np.sqrt(
                    np.mean(np.sum(trace["foot_velocity"][mask] ** 2, axis=-1), axis=0)
                ).tolist(),
                "mean_sum_Fz_N": float(force[:, :, 2].sum(axis=1).mean()),
                "mean_sum_sand_Fz_N": float(trace["sand_force"][mask, :, 2].sum(axis=1).mean()),
                "mean_sum_rigid_Fz_N": float(trace["rigid_force"][mask, :, 2].sum(axis=1).mean()),
                "root_height_range_m": [
                    float(trace["root_position"][mask, 2].min()),
                    float(trace["root_position"][mask, 2].max()),
                ],
                "max_root_xy_displacement_m": float(
                    np.linalg.norm(trace["root_position"][mask, :2] - trace["root_position"][0, :2], axis=1).max()
                ),
            }
    reference = metadata["mpm_tau0"]
    for case, meta in metadata.items():
        for key in (
            "checkpoint_sha256",
            "initial_particles_sha256",
            "particle_grid",
            "material",
            "particle_count",
            "seed",
            "physics_dt_s",
            "control_dt_s",
        ):
            assert meta[key] == reference[key], (case, key)
    (args.output / "filter_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    plt.rcParams.update(
        {"font.family": "DejaVu Serif", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )
    fig, axes = plt.subplots(3, 2, figsize=(13, 9), sharex=True, sharey="row")
    for row, component in enumerate("xyz"):
        for foot, name in enumerate(("Left", "Right")):
            ax = axes[row, foot]
            for case, (label, color, style) in CASES.items():
                trace = traces[case]
                ax.plot(
                    trace["time"], trace["force"][:, foot, row], label=label, color=color, ls=style, lw=0.9, alpha=0.9
                )
            ax.set_title(f"({chr(97 + row * 2 + foot)}) {name} foot")
            ax.set_ylabel(f"$F_{component}$ (N)")
            ax.grid(alpha=0.18)
            ax.set_xlim(0, 30)
            if row == 2:
                ax.set_xlabel("Simulation time (s)")
    finish(
        fig,
        axes[0, 0],
        args.output / "policy_jqur5jhm_zero_command_filtering.png",
        "Figure 7: Sand feedback plus rigid-ground reaction on the feet. (a,b) Fx; (c,d) Fy; (e,f) Fz. "
        "Left/right columns: left/right feet.\n"
        "Independent closed-loop rollouts at 200 Hz. "
        "EMA time constants: 10/20 ms; sand force and torque filtered before feedback to the robot.",
    )
    fig, axes = plt.subplots(3, 2, figsize=(13, 9), sharex=True)
    for case, (label, color, style) in CASES.items():
        trace = traces[case]
        values = [
            trace["foot_position"][:, 0, 2],
            trace["foot_position"][:, 1, 2],
            trace["root_position"][:, 2],
            np.linalg.norm(trace["root_position"][:, :2] - trace["root_position"][0, :2], axis=1),
            trace["sand_force"][:, :, 2].sum(axis=1),
            trace["rigid_force"][:, :, 2].sum(axis=1),
        ]
        for ax, value in zip(axes.flat, values, strict=True):
            ax.plot(trace["time"], value, color=color, ls=style, label=label, lw=1)
    for ax, title, ylabel in zip(
        axes.flat,
        (
            "(a) Left ankle-roll body",
            "(b) Right ankle-roll body",
            "(c) Robot root height",
            "(d) Robot root displacement",
            "(e) Sand support",
            "(f) Rigid-ground support",
        ),
        (
            "z relative to initial\nsand surface (m)",
            "z relative to initial\nsand surface (m)",
            "World z (m)",
            "Distance from initial root xy (m)",
            "Sum of foot $F_z$ (N)",
            "Sum of foot $F_z$ (N)",
        ),
        strict=True,
    ):
        ax.set(title=title, ylabel=ylabel, xlabel="Simulation time (s)", xlim=(0, 30))
        ax.grid(alpha=0.18)
    finish(
        fig,
        axes[0, 0],
        args.output / "diagnostics/policy_jqur5jhm_zero_command_filtering_motion.png",
        "Figure 8: (a,b) Left/right ankle-roll body z relative to the initial sand surface; "
        "(c) robot root world z; (d) root xy displacement.\n"
        "(e,f) Sum of left/right foot Fz from sand/rigid-ground contact. "
        "Same policy, seed and initial particles; independent closed-loop rollouts.",
    )


def finish(fig, ax, output, caption):
    fig.suptitle("MPM-trained | jqur5jhm | checkpoint 19,999 | Zero command", y=0.99)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.96), ncol=4)
    fig.text(0.5, 0.01, caption, ha="center", va="bottom", fontsize=9)
    fig.tight_layout(rect=(0, 0.09, 1, 0.91))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
