# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Draw an authored particle-selection schematic; positions and sets are illustrative."""

import matplotlib.pyplot as plt
from figure_style import COLORS, save_figure
from matplotlib.patches import Rectangle


def main() -> None:
    """Show active, sampled-background, and unsampled particle roles in one material bed."""
    fig, ax = plt.subplots(figsize=(7.3, 3.2))
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.03, top=0.97)
    ax.set(xlim=(-0.1, 10.4), ylim=(-0.45, 4.05), aspect="equal")
    ax.axis("off")
    ax.add_patch(Rectangle((1.45, 2.98), 1.65, 0.85, facecolor="none", edgecolor="0.2", lw=1.2))
    ax.text(2.275, 3.405, "Collider\nbody", ha="center", va="center", fontsize=10)
    # Authored layout, not a simulated particle state or an execution of the selector.
    active = {(0, 2), (0, 3), (0, 4), (0, 5), (1, 2), (1, 3), (1, 4), (1, 5), (2, 3), (2, 4)}
    background = {(0, 0), (0, 8), (1, 7), (2, 1), (2, 9), (3, 5), (4, 2), (4, 10)}
    for row in range(5):
        for col in range(11):
            x = 0.25 + 0.55 * col + (0.08 if row % 2 else 0)
            y = 2.68 - 0.51 * row
            if (row, col) in active:
                ax.plot(x, y, "o", ms=6.5, color=COLORS[0], markeredgecolor="0.3", markeredgewidth=0.5)
            elif (row, col) in background:
                ax.plot(x, y, "s", ms=7, markerfacecolor="white", markeredgecolor=COLORS[2], markeredgewidth=1.5)
            else:
                ax.plot(x, y, "o", ms=6.5, markerfacecolor="white", markeredgecolor="0.65", markeredgewidth=1)

    entries = (
        (3.25, "o", COLORS[0], COLORS[0], "Active particles", "Selected by priority score\nTapered predicted motion"),
        (1.95, "s", "white", COLORS[2], "Background samples", "Sampled for material context\nFull predicted motion"),
        (0.65, "o", "white", "0.65", "Inactive, unsampled", "Outside both selected sets\nZero neural motion update"),
    )
    for y, marker, fill, edge, title, detail in entries:
        ax.plot(6.45, y, marker, ms=7, markerfacecolor=fill, markeredgecolor=edge, markeredgewidth=1.3)
        ax.text(6.75, y, title, fontsize=10, weight="bold", va="center")
        ax.text(6.75, y - 0.27, detail, fontsize=9, va="top", linespacing=1.5)
    ax.text(2.9, -0.12, "Material particles", ha="center", va="center", fontsize=10)
    save_figure(fig, "particle_selection")


if __name__ == "__main__":
    main()
