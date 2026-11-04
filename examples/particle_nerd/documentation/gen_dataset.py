# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Plot archived G1 reference-trajectory counts, including native termination outcomes."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from figure_style import COLORS, save_figure


def main() -> None:
    """Read audited counts and plot dataset composition without model-performance claims."""
    data = json.loads(Path(__file__).with_name("dataset_counts.json").read_text())
    groups = data["groups"]
    assert sum(sum(g[s] for s in data["split_counts"]) for g in groups) == data["trajectory_count"]
    assert sum(data["outcome_counts"].values()) == data["trajectory_count"]
    fig, axes = plt.subplots(1, 2, figsize=(7.3, 3.3), gridspec_kw={"width_ratios": [2.5, 1]}, layout="constrained")
    ax = axes[0]
    bottom = np.zeros(len(groups))
    for split, label, color, hatch in zip(
        ("train", "validation", "test"),
        ("Training", "Validation", "Test"),
        COLORS,
        ("", "///", "..."),
        strict=True,
    ):
        values = np.array([g[split] for g in groups])
        assert int(values.sum()) == data["split_counts"][split]
        ax.bar(np.arange(len(groups)), values, bottom=bottom, color=color, hatch=hatch, edgecolor="white", label=label)
        bottom += values
    ax.set_xticks(np.arange(len(groups)), [g["group"].capitalize() for g in groups], rotation=30, ha="right")
    ax.set_ylabel("Recorded trajectories (count)")
    ax.set_ylim(0, max(bottom) * 1.4)
    ax.legend(frameon=False, ncol=3, loc="upper center", handlelength=1.2, columnspacing=0.9)
    for i, value in enumerate(bottom):
        ax.text(i, value + 2, str(int(value)), ha="center", fontsize=9)
    ax = axes[1]
    values = [data["outcome_counts"][k] for k in ("duration", "native_termination")]
    bars = ax.bar([0, 1], values, color=["0.5", "white"], edgecolor="0.25", hatch=["", "///"], width=0.65)
    ax.set_xticks([0, 1], ["Requested\nduration", "Native\ntermination"])
    ax.set_ylabel("Recorded trajectories (count)")
    ax.set_ylim(0, data["trajectory_count"] * 1.12)
    ax.bar_label(bars, padding=4, fontsize=9)
    save_figure(fig, "dataset")


if __name__ == "__main__":
    main()
