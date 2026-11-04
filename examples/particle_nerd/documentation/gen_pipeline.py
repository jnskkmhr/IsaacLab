# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Render an authored schematic of single-intruder Particle NeRD coupling; no empirical data."""

import matplotlib.pyplot as plt
from figure_style import COLORS, save_figure
from matplotlib.patches import Rectangle


def main() -> None:
    """Draw dependencies, distinguishing old geometry from predicted material velocity."""
    fig, ax = plt.subplots(figsize=(7.3, 3.6))
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.98)
    ax.set(xlim=(0, 10), ylim=(0, 3.6))
    ax.axis("off")
    boxes = [
        (0.1, 1.9, 2.0, 1.1, "Current state\nParticle history\nCollider state"),
        (2.7, 1.9, 1.8, 1.1, "Particle Transolver\nDisplacement\n$\\Delta\\widehat{\\mathbf{x}}$"),
        (5.2, 1.9, 2.1, 1.1, "Analytical contact\nMaterial-on-body\nforce + COM torque"),
        (8.0, 1.9, 1.9, 1.1, "Newton MJWarp\nRigid substeps\nNext rigid state"),
    ]
    for x, y, w, h, label in boxes:
        ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor="0.2", linewidth=1.0))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", linespacing=1.65, fontsize=9)
    for start, end in ((2.1, 2.7), (4.5, 5.2), (7.3, 8.0)):
        ax.annotate("", (end, 2.45), (start, 2.45), arrowprops={"arrowstyle": "->", "lw": 1.3})
    ax.text(4.85, 2.85, "$\\Delta\\widehat{\\mathbf{x}}/\\Delta t$", ha="center", fontsize=10)
    ax.plot([1.1, 1.1, 6.25, 6.25], [3.0, 3.42, 3.42, 3.08], color="0.35", lw=1.0)
    ax.annotate("", (6.25, 3.0), (6.25, 3.14), arrowprops={"arrowstyle": "->", "color": "0.35"})
    ax.text(
        4.0,
        3.42,
        "Interval-start particle positions + rigid state",
        ha="center",
        va="center",
        backgroundcolor="white",
        fontsize=9,
    )
    ax.annotate("", (3.6, 1.15), (3.6, 1.9), arrowprops={"arrowstyle": "->", "color": COLORS[0]})
    ax.text(3.6, 0.93, "Append predicted positions\nto particle history", ha="center", va="center", fontsize=9)
    ax.plot([9.72, 9.72, 1.1, 1.1], [1.9, 0.25, 0.25, 1.72], color=COLORS[0], lw=1.2)
    ax.annotate("", (1.1, 1.9), (1.1, 1.7), arrowprops={"arrowstyle": "->", "color": COLORS[0]})
    ax.plot([3.6, 3.6], [0.62, 0.25], color=COLORS[0], lw=1.2)
    ax.text(
        6.15, 0.25, "State feedback at the next interval", ha="center", va="center", backgroundcolor="white", fontsize=9
    )
    ax.annotate("", (8.65, 1.9), (8.65, 1.18), arrowprops={"arrowstyle": "->", "color": "0.35"})
    ax.text(8.65, 0.93, "Rigid-ground\ncontact", ha="center", va="center", fontsize=9)
    ax.text(6.15, 0.93, "Known loads\n+ gravity", ha="center", va="center", fontsize=9)
    ax.annotate("", (6.25, 1.9), (6.25, 1.18), arrowprops={"arrowstyle": "->", "color": "0.35"})
    ax.annotate("", (8.2, 1.9), (6.65, 1.18), arrowprops={"arrowstyle": "->", "color": "0.35"})
    save_figure(fig, "pipeline")


if __name__ == "__main__":
    main()
