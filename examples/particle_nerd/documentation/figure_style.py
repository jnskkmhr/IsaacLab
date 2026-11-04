# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared print style and output locations for the Particle NeRD methodology figures."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
COLORS = ("#76B900", "#606060", "#B7791F")
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Liberation Sans", "Arial", "DejaVu Sans"],
        "font.size": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "savefig.dpi": 300,
    }
)


def save_figure(fig: plt.Figure, name: str) -> None:
    """Write a small documentation JPG and a vector PDF build artifact."""
    preview = ROOT / "examples/particle_nerd/documentation/figures"
    vector = ROOT / "docs/_build/particle_nerd_figures"
    preview.mkdir(parents=True, exist_ok=True)
    vector.mkdir(parents=True, exist_ok=True)
    fig.savefig(vector / f"{name}.pdf", bbox_inches="tight", pad_inches=0.08)
    fig.savefig(preview / f"{name}.jpg", bbox_inches="tight", pad_inches=0.08, pil_kwargs={"quality": 88})
    plt.close(fig)
