# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Plot recorded G1 foot forces and an offline causal moving-average diagnostic."""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

POLICIES = {
    "jqur5jhm": "MPM-trained",
    "iylbaqt3": "NeRD scratch",
    "y4t5ybra": "NeRD fine-tuned",
}
COLORS = {"mpm": "#17365D", "nerd": "#187B38"}
LABELS = {"mpm": "Implicit MPM", "nerd": "Particle NeRD"}


def moving_average(values, samples):
    """Causal filter, replicating the initial value only for its startup history."""
    return np.convolve(np.pad(values, (samples - 1, 0), mode="edge"), np.ones(samples) / samples, mode="valid")


def spectrum(values, dt):
    centered = values - values.mean()
    window = np.hanning(len(values))
    power = np.abs(np.fft.rfft(centered * window)) ** 2 * dt / np.sum(window**2)
    if len(values) % 2:
        power[1:] *= 2
    else:
        power[1:-1] *= 2
    return np.fft.rfftfreq(len(values), dt), power


def load(root, run, backend, speed=0):
    folder = root / f"{run}_{backend}_{speed:g}"
    return dict(np.load(folder / "trace.npz")), json.loads((folder / "metadata.json").read_text())


def save(fig, output, name, caption):
    fig.text(0.5, 0.01, caption, ha="center", va="bottom", fontsize=9)
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    fig.savefig(output / name, dpi=180, facecolor="white")
    plt.close(fig)


def plot_policy(traces, run, speed, figure, output):
    """Compare signed world-frame foot reactions for one policy and command."""
    mpm, nerd = [traces[backend][1] for backend in COLORS]
    for key in ("initial_particles_sha256", "checkpoint_sha256", "particle_count", "particle_grid", "material"):
        assert mpm[key] == nerd[key], f"Unmatched comparison for {run}: {key}"
    command = "Zero command" if speed == 0 else f"Forward command: {speed:g} m/s"
    suffix = "zero_command" if speed == 0 else f"v={speed:g}"
    fig, axes = plt.subplots(3, 2, figsize=(12, 9), sharex=True, sharey="row")
    fig.suptitle(f"{POLICIES[run]} | {run} | checkpoint {mpm['checkpoint_iteration']:,} | {command}", y=0.99)
    for row, component in enumerate(("x", "y", "z")):
        for foot, name in enumerate(("Left", "Right")):
            ax = axes[row, foot]
            for backend, (trace, _) in traces.items():
                ax.plot(
                    trace["time"],
                    trace["force"][:, foot, row],
                    color=COLORS[backend],
                    label=LABELS[backend],
                    lw=1.05,
                    alpha=0.9,
                )
            ax.set_title(f"({chr(97 + row * 2 + foot)}) {name} foot", fontsize=11)
            ax.set_ylabel(f"$F_{component}$ (N)")
            ax.grid(alpha=0.18)
            if row == 2:
                ax.set_xlabel("Simulation time (s)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.96), ncol=2)
    save(
        fig,
        output,
        f"policy_{run}_{suffix}.png",
        f"Figure {figure}: World-frame forces on the feet. (a,b) Fx; (c,d) Fy; (e,f) Fz. "
        "Left/right columns: left/right feet.\n"
        "Same policy and fixed sand bed; independent closed-loop rollouts. "
        "Raw signed forces sampled at 200 Hz; +z points upward.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {"font.family": "DejaVu Serif", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )
    traces = {}
    metrics = {}
    titles = {}
    for run in POLICIES:
        traces[run] = {backend: load(args.data, run, backend) for backend in COLORS}
        mpm, nerd = [traces[run][backend][1] for backend in COLORS]
        titles[run] = f"{POLICIES[run]} ({mpm['checkpoint_iteration']:,})"
        for key in ("initial_particles_sha256", "checkpoint_sha256", "particle_count", "particle_grid", "material"):
            assert mpm[key] == nerd[key], f"Unmatched comparison for {run}: {key}"
        metrics[run] = {}
        for backend, (trace, metadata) in traces[run].items():
            mask = trace["time"] >= 1.0
            if mask.sum() < 10:
                raise ValueError(f"Insufficient post-transient samples: {run}/{backend}")
            force = trace["force"][mask]
            vertical = force[:, :, 2]
            net = vertical.sum(axis=1)
            hz, power = spectrum(net, metadata["physics_dt_s"])
            contact = np.linalg.norm(force, axis=-1) > 5
            result = {
                "outcome": metadata["outcome"],
                "duration_s": metadata["recorded_seconds"],
                "analysis_start_s": 1.0,
                "robot_weight_N": metadata["robot_mass_kg"] * 9.81,
                "mean_vertical_force_N": vertical.mean(axis=0).tolist(),
                "std_vertical_force_N": vertical.std(axis=0).tolist(),
                "p99_vertical_force_N": np.percentile(vertical, 99, axis=0).tolist(),
                "mean_horizontal_force_N": np.linalg.norm(force[:, :, :2], axis=-1).mean(axis=0).tolist(),
                "net_vertical_mean_N": float(net.mean()),
                "net_vertical_step_change_rms_N": float(np.sqrt(np.mean(np.diff(net) ** 2))),
                "net_vertical_power_above_20Hz_fraction": float(power[hz >= 20].sum() / max(power.sum(), 1e-30)),
                "foot_vertical_excursion_m": np.ptp(trace["foot_position"][mask, :, 2], axis=0).tolist(),
                "foot_vertical_speed_rms_m_s": np.sqrt(
                    np.mean(trace["foot_velocity"][mask, :, 2] ** 2, axis=0)
                ).tolist(),
                "foot_contact_switches_5N": np.count_nonzero(np.diff(contact, axis=0), axis=0).tolist(),
                "rigid_force_max_N": float(np.linalg.norm(trace["rigid_force"][mask], axis=-1).max()),
                "root_xy_displacement_m": float(
                    np.linalg.norm(trace["root_position"][-1, :2] - trace["root_position"][0, :2])
                ),
                "moving_average": {},
            }
            for samples in (2, 4, 8):
                filtered = moving_average(trace["force"][:, :, 2].sum(axis=1), samples)[mask]
                filtered_feet = np.stack(
                    [moving_average(trace["force"][:, foot, 2], samples)[mask] for foot in range(2)], axis=1
                )
                result["moving_average"][f"{samples * 5}ms"] = {
                    "linear_phase_delay_ms": (samples - 1) * 2.5,
                    "step_change_rms_N": float(np.sqrt(np.mean(np.diff(filtered) ** 2))),
                    "raw_difference_rms_N": float(np.sqrt(np.mean((filtered - net) ** 2))),
                    "per_foot_vertical_std_N": filtered_feet.std(axis=0).tolist(),
                }
            metrics[run][backend] = result
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    for figure, run in enumerate(POLICIES, start=1):
        plot_policy(traces[run], run, 0, figure, args.output)
    if all((args.data / f"jqur5jhm_{backend}_0.5/trace.npz").exists() for backend in COLORS):
        walking = {backend: load(args.data, "jqur5jhm", backend, 0.5) for backend in COLORS}
        plot_policy(walking, "jqur5jhm", 0.5, 4, args.output)

    diagnostics = args.output / "diagnostics"
    diagnostics.mkdir(exist_ok=True)
    fig, axes = plt.subplots(3, 2, figsize=(12, 9), sharex=True)
    for row, (run, policy) in enumerate(titles.items()):
        for col in range(2):
            ax = axes[row, col]
            for backend, (trace, metadata) in traces[run].items():
                values = trace["foot_position"][:, col, 2]
                ax.plot(trace["time"], values, color=COLORS[backend], label=LABELS[backend], lw=1.05, alpha=0.9)
            ax.set_title(f"({chr(97 + row * 2 + col)}) {policy} — {'left' if col == 0 else 'right'} foot", fontsize=11)
            ax.set_ylabel("Foot-link z relative to\ninitial sand surface (m)", fontsize=9)
            ax.grid(alpha=0.18)
            if row == 2:
                ax.set_xlabel("Simulation time (s)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)
    save(
        fig,
        diagnostics,
        "foot_motion_zero_command.png",
        "Figure 5: Zero-command fixed-sand comparison. "
        "(a,b) MPM-trained; (c,d) NeRD scratch; (e,f) NeRD fine-tuned.\n"
        "Left/right columns show left/right feet. Same policy in each pair; "
        "independent closed-loop motion. Sampling: 200 Hz.",
    )

    fig, axes = plt.subplots(3, 2, figsize=(12, 9))
    for row, (run, policy) in enumerate(titles.items()):
        for backend, (trace, metadata) in traces[run].items():
            net = trace["force"][:, :, 2].sum(axis=1)
            axes[row, 0].plot(trace["time"], net, color=COLORS[backend], label=LABELS[backend], lw=0.9, alpha=0.65)
            mask = trace["time"] >= 1
            hz, power = spectrum(net[mask], 0.005)
            axes[row, 1].semilogy(hz[1:], np.maximum(power[1:], 1e-12), color=COLORS[backend], label=LABELS[backend])
            if backend == "nerd":
                filtered = moving_average(net, 4)
                axes[row, 0].plot(trace["time"], filtered, color="#70B77E", label="NeRD, causal 20 ms average", lw=1.5)
                hz, power = spectrum(filtered[mask], 0.005)
                axes[row, 1].semilogy(hz[1:], np.maximum(power[1:], 1e-12), color="#70B77E")
        axes[row, 0].set_title(f"({chr(97 + 2 * row)}) {policy}")
        axes[row, 1].set_title(f"({chr(98 + 2 * row)}) {policy}")
        axes[row, 0].set(xlabel="Simulation time (s)", ylabel="Sum of foot $F_z$ (N)")
        axes[row, 1].set(xlabel="Frequency (Hz)", ylabel="Force PSD (N²/Hz)", xlim=(0, 100))
        axes[row, 1].axvline(20, color="gray", ls=":", lw=0.7)
        for ax in axes[row]:
            ax.grid(alpha=0.18)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3)
    save(
        fig,
        diagnostics,
        "force_filter_diagnostic.png",
        "Figure 6: (a,c,e) Sum of foot Fz; (b,d,f) spectra after the first second. "
        "Rows: MPM-trained, NeRD scratch, NeRD fine-tuned.\n"
        "Four-sample causal average adds 7.5 ms group delay. "
        "Filtering is offline only; no filtered-force simulation is shown.",
    )


if __name__ == "__main__":
    main()
