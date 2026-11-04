# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Render authored collider-query and free-body sketches; coordinates are not measured data."""

import matplotlib.pyplot as plt
from figure_style import COLORS, save_figure
from matplotlib.patches import Circle, Rectangle


def arrow(ax, start, end, color="0.2", style="->", lw=1.2):
    """Draw a schematic vector or dimension line."""
    ax.annotate("", end, start, arrowprops={"arrowstyle": style, "color": color, "lw": lw})


def main() -> None:
    """Distinguish pose origin, COM, particle centers, and surface queries."""
    fig, axes = plt.subplots(1, 2, figsize=(7.3, 3.6))
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.98, wspace=0.14)
    for ax, panel in zip(axes, ("(a)", "(b)"), strict=True):
        ax.set(xlim=(-0.1, 4.3), ylim=(-0.15, 4.3), aspect="equal")
        ax.axis("off")
        ax.text(0, 4.08, panel, weight="bold")
        ax.add_patch(Rectangle((0.6, 2.1), 2.9, 1.6, facecolor="none", edgecolor="0.2", lw=1.3))
        ax.text(2.05, 3.42, "Collider body", ha="center", fontsize=10)
        ax.plot(2.15, 2.95, "ko", ms=4)
        ax.text(2.32, 2.92, "$\\mathbf{c}_b$ (COM)", fontsize=10, va="center")
    ax = axes[0]
    ax.plot(1.08, 2.6, marker="+", color="0.2", ms=8, mew=1.3)
    ax.text(0.8, 2.32, "$\\mathbf{p}_b$ (origin)", fontsize=9)
    for x, y in ((0.8, 0.48), (1.55, 0.42), (2.4, 0.55), (3.22, 0.4), (3.7, 0.88), (0.55, 1.02)):
        ax.add_patch(Circle((x, y), 0.13, edgecolor="0.55", facecolor="none", lw=0.9))
        ax.plot(x, y, ".", color="0.55", ms=2)
    for x, y in ((1.5, 1.13), (3.12, 1.32)):
        ax.add_patch(Circle((x, y), 0.18, edgecolor=COLORS[0], facecolor="none", lw=1.4))
        ax.plot(x, y, ".", color="0.15", ms=4)
        ax.plot([x, x], [y, 2.1], "--", color="0.5", lw=1)
        ax.plot(x, 2.1, marker="s", ms=5, markerfacecolor="white", markeredgecolor="0.2")
    ax.text(1.36, 0.84, "$\\mathbf{x}_i$", ha="right", fontsize=11)
    ax.text(3.35, 1.25, "$\\mathbf{x}_j$", fontsize=11)
    ax.text(1.25, 1.91, "$\\mathbf{s}_{ib}$", ha="right", fontsize=11)
    ax.text(3.4, 1.94, "$\\mathbf{s}_{jb}$", fontsize=11)
    arrow(ax, (1.96, 1.13), (1.96, 2.1), style="<->", lw=0.9)
    ax.text(2.05, 1.6, "$d_{ib}$", fontsize=11, va="center")
    arrow(ax, (1.5, 1.13), (1.68, 1.13), style="-", lw=1)
    ax.text(1.78, 1.0, "$r_i$", fontsize=10)
    arrow(ax, (2.6, 2.1), (2.6, 1.48), color=COLORS[1])
    ax.text(2.72, 1.65, "$\\mathbf{o}_{ib}$", fontsize=11)
    ax.text(2.0, -0.02, "Material-particle centers", ha="center", fontsize=9)
    arrow(ax, (0.12, 1.55), (0.12, 2.1))
    ax.text(0.13, 2.2, "$\\mathbf{e}_z$", fontsize=11, ha="center")

    ax = axes[1]
    surface = (1.1, 2.1)
    ax.plot([2.15, surface[0]], [2.95, surface[1]], "--", color="0.55", lw=1)
    ax.text(2.28, 2.46, "$\\mathbf{s}_{ib}-\\mathbf{c}_b$", fontsize=11)
    ax.plot(*surface, marker="s", ms=5, markerfacecolor="white", markeredgecolor="0.2", zorder=5)
    ax.text(0.84, 1.95, "$\\mathbf{s}_{ib}$", ha="right", fontsize=11)
    arrow(ax, surface, (1.1, 2.9), color=COLORS[1])
    ax.text(0.98, 2.82, "$\\mathbf{n}_{ib}$", ha="right", fontsize=11)
    arrow(ax, surface, (1.86, 2.1), color=COLORS[1])
    ax.text(1.82, 1.88, "$\\mathbf{t}_{i1}$", fontsize=11)
    arrow(ax, surface, (1.67, 2.72), color=COLORS[0], lw=2)
    ax.text(1.5, 2.85, "$\\mathbf{j}_i$", fontsize=12, color="0.2")
    arrow(ax, (2.15, 2.95), (2.15, 1.05), color=COLORS[2], lw=1.5)
    ax.text(2.3, 1.35, "$M_b\\mathbf{g}$", fontsize=11)
    arrow(ax, (3.5, 2.75), (4.18, 2.75), lw=1.4)
    ax.text(3.84, 2.99, "$\\mathbf{F}_{\\rm ext}$", fontsize=11, ha="center")
    ax.text(
        2.05,
        0.5,
        "Contact impulse $\\mathbf{j}_i$: N s\nGravity / external force: N",
        ha="center",
        fontsize=9,
        linespacing=1.6,
    )
    save_figure(fig, "contact_geometry")


if __name__ == "__main__":
    main()
