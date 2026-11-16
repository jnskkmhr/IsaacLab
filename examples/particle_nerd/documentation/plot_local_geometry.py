# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Draw Section 2.2 geometry and gap encodings from documented code defaults.

Run: uv run --no-project --with matplotlib --with numpy python
examples/particle_nerd/documentation/plot_local_geometry.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle

FIGURES = Path(__file__).with_name("figures")
BLUE = "#224c70"
GREEN = "#348242"
GRAY = "#7b8186"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "mathtext.fontset": "stix", "svg.fonttype": "path"})


def save(fig, name):
    """Export vector artwork and a raster preview."""
    for suffix in ("svg", "pdf", "png"):
        fig.savefig(FIGURES / f"{name}.{suffix}", dpi=240, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def geometry():
    """Illustrate an exterior particle and a planar rigid-collider surface."""
    fig, ax = plt.subplots(figsize=(7.3, 3.3))
    ax.set(xlim=(0, 10.4), ylim=(0, 4.6), aspect="equal")
    ax.axis("off")
    ax.add_patch(Rectangle((0.5, 0.5), 6, 1.1, facecolor="#e0e8ee", edgecolor=BLUE, linewidth=1.5))
    ax.text(0.75, 0.72, "Rigid collider body", color=BLUE, fontsize=11)
    x, y, surface, radius = 3.2, 3.35, 1.6, 0.4
    ax.add_patch(Circle((x, y), radius, facecolor="#cce3c8", edgecolor=GREEN, linewidth=1.6))
    ax.plot(x, y, "o", color=GREEN, markersize=4)
    ax.text(x - 0.3, y + 0.65, r"Particle center $\mathbf{x}_i^t$", ha="center", fontsize=11)
    ax.plot(x, surface, "o", markerfacecolor="white", markeredgecolor=BLUE, markersize=6)
    ax.text(x + 0.38, surface - 0.28, r"$\mathbf{s}_{ib}$", ha="right", fontsize=12, color=BLUE)
    ax.plot([x, x], [surface, y], ":", color=GRAY, linewidth=1)
    ax.annotate("", (x, surface + 0.9), (x, surface), arrowprops={"arrowstyle": "->", "color": BLUE, "lw": 1.8})
    ax.text(x - 0.15, surface + 0.63, r"$\mathbf{n}_{ib}$", ha="right", fontsize=12, color=BLUE)
    ax.plot(1.25, 1.15, "o", color=BLUE, markersize=4)
    ax.text(0.85, 1.32, "Shape origin", fontsize=10)
    ax.text(1.45, 1.08, r"$\mathbf{p}_b$", fontsize=12, color=BLUE)
    ax.annotate("", (x + radius, y), (x, y), arrowprops={"arrowstyle": "<->", "color": GREEN, "lw": 1})
    ax.text(x + radius + 0.12, y, r"$r_i$", fontsize=12, va="center", color=GREEN)
    for xx, upper, label in [(4.55, y, r"$d_{ib}$"), (5.85, y - radius, r"$g_{ib}$")]:
        ax.plot([x + 0.12, xx + 0.12], [upper, upper], ":", color=GRAY, linewidth=0.8)
        ax.annotate("", (xx, upper), (xx, surface), arrowprops={"arrowstyle": "<->", "color": BLUE, "lw": 1.2})
        ax.text(xx + 0.12, (upper + surface) / 2, label, fontsize=13, color=BLUE)
    ax.text(6.85, 3.55, "Center-to-surface distance", fontsize=10)
    ax.text(6.85, 3.17, r"$d_{ib}>0$ outside the collider", fontsize=11)
    ax.text(6.85, 2.45, "Particle-surface clearance", fontsize=10)
    ax.text(6.85, 2.05, r"$g_{ib}=d_{ib}-r_i$", fontsize=13)
    ax.text(6.85, 1.35, "Closest surface point", fontsize=10)
    ax.text(6.85, 0.95, r"$\mathbf{s}_{ib}=\mathbf{x}_i^t-d_{ib}\mathbf{n}_{ib}$", fontsize=12)
    save(fig, "local_surface_geometry")


def gap_curves():
    """Compare the old cutoff and smooth one- and two-scale feature encodings."""
    params = json.loads((FIGURES / "local_gap_parameters.json").read_text())
    far, gap_mix, influence_mix = (params[k] for k in ("far_support_scale", "far_gap_weight", "far_influence_weight"))
    s = np.linspace(-2, 10, 2401)
    positive = np.maximum(s, 0)
    gap = (1 - gap_mix) * np.tanh(s) + gap_mix * np.tanh(s / far)
    influence = (1 - influence_mix) * (1 - np.tanh(positive) ** 2) + influence_mix * (1 - np.tanh(positive / far) ** 2)
    hard_gap = np.clip(s, -1, 1)
    hard_influence = np.clip(1 - positive, 0, 1)
    # Check the cutoff and contact limits independently of the visual rendering.
    assert np.all(hard_influence[s >= 1] == 0)
    assert np.all(influence[s > 1] > 0)
    assert np.isclose(gap[s == 0], 0).all()
    assert np.isclose(influence[s == 0], 1).all()
    fig, axes = plt.subplots(1, 2, figsize=(7.3, 3.5), layout="constrained")
    fig.get_layout_engine().set(rect=(0, 0.12, 1, 0.88))
    series = [
        (hard_gap, hard_influence, "Hard cutoff", BLUE, "--"),
        (np.tanh(s), 1 - np.tanh(positive) ** 2, "Single-scale tanh", GRAY, ":"),
        (gap, influence, "Two-scale tanh", GREEN, "-"),
    ]
    for a, b, label, color, style in series:
        axes[0].plot(s, a, label=label, color=color, linestyle=style, linewidth=1.8)
        axes[1].plot(s, b, label=label, color=color, linestyle=style, linewidth=1.8)
    for label, ax in zip(["(a)", "(b)"], axes):
        ax.text(0, 1.04, label, transform=ax.transAxes, fontsize=11)
        ax.axvline(1, color="#929292", linewidth=0.8)
        ax.set_xlim(-2, 10)
        ax.set_xlabel(r"Normalized surface gap $s_{ib}=g_{ib}/\ell_s$")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.14)
    axes[0].set_ylabel(r"Encoded gap $\tilde{g}_{ib}$")
    axes[1].set_ylabel(r"Collider influence $w_{ib}$")
    axes[1].set_ylim(-0.03, 1.05)
    inset = axes[1].inset_axes([0.40, 0.49, 0.55, 0.43])
    for _, b, _, color, style in series:
        inset.plot(s, b, color=color, linestyle=style, linewidth=1.3)
    inset.set(xlim=(1, 10), ylim=(-0.005, 0.18), xticks=[1, 5, 10], yticks=[0, 0.1])
    inset.tick_params(labelsize=8)
    inset.text(0.5, 1.06, r"Beyond $s_{ib}=1$", transform=inset.transAxes, ha="center", fontsize=9)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3, frameon=False, fontsize=9)
    save(fig, "local_gap_scaling")
    np.savetxt(
        FIGURES / "local_gap_scaling.csv",
        np.column_stack([s, hard_gap, gap, hard_influence, influence]),
        delimiter=",",
        header="normalized_gap,hard_gap,two_scale_gap,hard_influence,two_scale_influence",
        comments="",
    )


if __name__ == "__main__":
    geometry()
    gap_curves()
