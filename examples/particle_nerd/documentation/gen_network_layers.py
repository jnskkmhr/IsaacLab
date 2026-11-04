# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Draw the Particle NeRD predictor, its attention blocks, and input encoders hierarchically."""

import matplotlib.pyplot as plt
from figure_style import COLORS, save_figure
from matplotlib.patches import Circle, Rectangle


def box(ax, x, y, width, height, label, fontsize=9):
    """Draw a network operation with a centered label."""
    ax.add_patch(Rectangle((x, y), width, height, fill=False, edgecolor="0.25", lw=1))
    ax.text(x + width / 2, y + height / 2, label, ha="center", va="center", fontsize=fontsize, linespacing=1.5)


def arrow(ax, start, end, color="0.25"):
    """Draw a directed dependency."""
    ax.annotate("", end, start, arrowprops={"arrowstyle": "->", "lw": 1, "color": color, "shrinkA": 0, "shrinkB": 0})


def route(ax, points, color="0.25"):
    """Draw a bent dependency with an arrow on its last segment."""
    ax.plot([p[0] for p in points[:-1]], [p[1] for p in points[:-1]], color=color, lw=1)
    arrow(ax, points[-2], points[-1], color)


def add_node(ax, x, y):
    """Draw a residual or feature-summation node."""
    ax.add_patch(Circle((x, y), 0.13, facecolor="white", edgecolor="0.25", lw=1, zorder=4))
    ax.text(x, y, "+", ha="center", va="center", fontsize=10, zorder=5)


def main() -> None:
    """Show predictor inputs/outputs, a block, and the input feature-encoding branches."""
    fig, axes = plt.subplots(3, 1, figsize=(7.3, 5.9), gridspec_kw={"height_ratios": [2.8, 2.1, 4.0]})
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.98, hspace=0.12)
    for ax, height, panel in zip(axes, (2.8, 2.1, 4.0), ("(a)", "(b)", "(c)"), strict=True):
        ax.set(xlim=(-0.1, 12.1), ylim=(-0.2, height), aspect="equal")
        ax.axis("off")
        ax.text(0, height - 0.1, panel, weight="bold", va="top", fontsize=10)

    ax = axes[0]
    box(
        ax,
        0.1,
        0.3,
        2.55,
        1.8,
        "Particle position history\nMaterial, mass, radius\nCollider geometry, state\n$\\Delta t$, gravity",
    )
    ax.add_patch(Rectangle((3.05, 0.35), 7.1, 1.95, fill=False, edgecolor="0.55", lw=1, linestyle="--"))
    ax.text(6.6, 2.05, "Particle NeRD predictor", ha="center", va="center", fontsize=10, weight="bold")
    box(ax, 3.35, 0.8, 1.8, 0.8, "Input encoders\nsee (c)")
    box(ax, 5.6, 0.8, 1.85, 0.8, "4 attention blocks\nsee (b)", fontsize=8.5)
    box(ax, 7.9, 0.8, 1.95, 0.8, "Displacement\ndecoder")
    arrow(ax, (2.65, 1.2), (3.35, 1.2))
    arrow(ax, (5.15, 1.2), (5.6, 1.2))
    arrow(ax, (7.45, 1.2), (7.9, 1.2))
    arrow(ax, (9.85, 1.2), (10.45, 1.2))
    ax.text(
        11.2, 1.2, "Particle\ndisplacement\n$\\Delta\\widehat{\\mathbf{x}}_i$", ha="center", va="center", fontsize=10
    )
    ax.text(6.525, 0.02, "$w_i$", ha="center", va="center", fontsize=11)
    arrow(ax, (6.525, 0.22), (6.525, 0.8))

    ax = axes[1]
    ax.text(0.4, 0.9, "$\\mathbf{h}^{\\ell}$", ha="center", va="center", fontsize=11)
    box(ax, 1.1, 0.55, 1.05, 0.7, "Layer\nNorm")
    box(ax, 2.55, 0.55, 2.4, 0.7, "Slice attention\n4 heads, $S=64$")
    arrow(ax, (0.7, 0.9), (1.1, 0.9))
    arrow(ax, (2.15, 0.9), (2.55, 0.9))
    arrow(ax, (4.95, 0.9), (5.27, 0.9))
    route(ax, [(0.8, 0.9), (0.8, 1.65), (5.4, 1.65), (5.4, 1.03)], COLORS[0])
    add_node(ax, 5.4, 0.9)
    box(ax, 6.0, 0.55, 1.05, 0.7, "Layer\nNorm")
    box(ax, 7.45, 0.55, 2.3, 0.7, "Linear–GELU–Linear\n128 → 512 → 128")
    arrow(ax, (5.53, 0.9), (6.0, 0.9))
    arrow(ax, (7.05, 0.9), (7.45, 0.9))
    arrow(ax, (9.75, 0.9), (10.17, 0.9))
    route(ax, [(5.7, 0.9), (5.7, 0.15), (10.3, 0.15), (10.3, 0.77)], COLORS[0])
    add_node(ax, 10.3, 0.9)
    arrow(ax, (10.43, 0.9), (11.0, 0.9))
    ax.text(11.6, 0.9, "$\\mathbf{h}^{\\ell+1}$", ha="center", va="center", fontsize=11)
    ax.text(3.75, 0.03, "$w_i$", ha="center", va="center", fontsize=11)
    arrow(ax, (3.75, 0.22), (3.75, 0.55))

    ax = axes[2]
    box(ax, 0.1, 2.3, 2.55, 1.2, "Position history\nMaterial, mass, radius\n$\\Delta t$, gravity")
    box(ax, 3.1, 2.3, 2.45, 1.2, "Finite differences\n+ normalization\n25 features")
    box(ax, 6.0, 2.3, 2.6, 1.2, "Linear–GELU–Linear\n25 → 256 → 128")
    arrow(ax, (2.65, 2.9), (3.1, 2.9))
    arrow(ax, (5.55, 2.9), (6.0, 2.9))
    arrow(ax, (8.6, 2.9), (10.47, 2.9))
    add_node(ax, 10.6, 2.9)
    arrow(ax, (10.73, 2.9), (11.2, 2.9))
    ax.text(11.65, 2.9, "$\\mathbf{h}_i^0$", ha="center", va="center", fontsize=11)
    box(ax, 0.1, 0.4, 2.55, 1.3, "Particle kinematics\nCollider geometry\nPose, velocity, friction")
    box(ax, 3.1, 0.4, 2.45, 1.3, "Surface queries\nGap, normal, relative\nvelocity, friction (8)")
    box(ax, 6.0, 0.4, 2.6, 1.3, "Shared encoder\nLinear–GELU–Linear\n8 → 128 → 128")
    box(ax, 9.05, 0.4, 2.8, 1.3, "Distance-weighted sum\nover collider\nembeddings")
    arrow(ax, (2.65, 1.05), (3.1, 1.05))
    arrow(ax, (5.55, 1.05), (6.0, 1.05))
    arrow(ax, (8.6, 1.05), (9.05, 1.05))
    arrow(ax, (10.6, 1.7), (10.6, 2.77))
    route(ax, [(4.325, 0.4), (4.325, 0.0), (10.45, 0.0), (10.45, 0.4)], COLORS[0])
    ax.text(7.3, 0.0, "Distance weights $w_{ib}$", ha="center", va="center", backgroundcolor="white", fontsize=9)
    save_figure(fig, "network_layers")


if __name__ == "__main__":
    main()
