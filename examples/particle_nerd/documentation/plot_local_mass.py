# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Plot the analytical effective-mass ratios in Eqs. (24)–(28)."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.patches import Rectangle


def main():
    """Export paired heatmaps and their reproducible numerical samples."""
    out = Path(__file__).with_name("figures")
    cfg = json.loads((out / "local_mass_parameters.json").read_text())
    angle = np.linspace(0, 180, 181)
    packing = np.linspace(*cfg["packing_ratio_range"], 201)
    beta, rho = np.meshgrid(np.deg2rad(angle), packing)
    normals = np.stack([np.sin(beta), np.zeros_like(beta), np.cos(beta)], axis=-1)
    velocity = np.array(cfg["relative_velocity"], dtype=float)
    direction = velocity / np.linalg.norm(velocity)
    mu = np.clip(cfg["friction"], 0.25, 1)
    sin_phi = mu / np.sqrt(1 + mu**2)
    kp = (1 + sin_phi) / max(1 - sin_phi, 0.15)
    cos_beta = normals[..., 2]
    sin_gamma = -direction[2]
    attack = np.clip(normals @ direction, 0, 1)
    alpha_n = (
        kp ** (0.5 * (cos_beta + sin_gamma))
        * (cfg["c1"] + (1 - cfg["c1"]) * np.maximum(cos_beta, 0) ** 2 * kp)
        * (0.5 + 0.5 * attack)
    )
    alpha_t = kp ** np.maximum(cos_beta, 0)
    excess = max(cfg["base_scale"], 1) * rho**2 * (1 + cfg["compression_coefficient"] * cfg["compression"])
    ratios = [1 + excess * alpha_n, 1 + excess * alpha_t]
    assert all(np.allclose(a[0], 1) and np.all(a >= 1) for a in ratios)
    np.savetxt(
        out / "local_mass_scaling.csv",
        np.column_stack([np.rad2deg(beta).ravel(), rho.ravel(), ratios[0].ravel(), ratios[1].ravel()]),
        delimiter=",",
        header="beta_deg,packing_ratio,normal_mass_ratio,tangent_mass_ratio",
        comments="",
    )
    plt.rcParams.update({"font.family": "serif", "font.size": 11, "mathtext.fontset": "stix", "svg.fonttype": "none"})
    fig = plt.figure(figsize=(9, 5.0), layout="constrained")
    grid = fig.add_gridspec(2, 2, height_ratios=[1.1, 3.2], hspace=0.02)
    axes = [fig.add_subplot(grid[1, 0])]
    axes.append(fig.add_subplot(grid[1, 1], sharey=axes[0]))
    axes[1].tick_params(labelleft=False)
    for column in range(2):
        icons = fig.add_subplot(grid[0, column])
        icons.set(xlim=(0, 1), ylim=(0, 1))
        icons.axis("off")
        icons.text(0, 0.98, ["(a)", "(b)"][column], va="top")
        for center, name, offset in zip(
            [0.12, 0.5, 0.88], ["Bottom", "Side", "Top"], [(0, -0.24), (-0.125, 0), (0, 0.24)]
        ):
            # The bed surrounds the box; the dark particle identifies the queried face.
            for dx in [-0.13, -0.065, 0, 0.065, 0.13]:
                for y in [0.24, 0.38, 0.52, 0.66, 0.8]:
                    if abs(dx) > 0.085 or y < 0.35 or y > 0.69:
                        icons.plot(center + dx, y, "o", ms=3, color="#b6d7a8")
            icons.add_patch(
                Rectangle((center - 0.085, 0.35), 0.17, 0.34, facecolor="#e8eef3", edgecolor="#183957", lw=1)
            )
            px, py = center + offset[0], 0.52 + offset[1]
            icons.plot(px, py, "o", ms=5, color="#348242", zorder=4)
            # Inward normal points from the particle toward the box.
            end = (center + 0.6 * offset[0], 0.52 + 0.6 * offset[1])
            icons.annotate(
                "", xy=end, xytext=(px, py), arrowprops={"arrowstyle": "->", "color": "#183957", "lw": 1}, zorder=5
            )
            icons.text(center, 0.02, name, ha="center", va="bottom", fontsize=10)
    cmap = LinearSegmentedColormap.from_list("mass", ["#fafcf7", "#b6d7a8", "#348242", "#183957"])
    norm = LogNorm(1, max(a.max() for a in ratios))
    for ax, ratio in zip(axes, ratios):
        im = ax.pcolormesh(angle, packing, ratio, cmap=cmap, norm=norm, shading="auto", rasterized=True)
        ax.set_xticks([0, 90, 180], ["0°", "90°", "180°"])
        ax.set_xlabel(r"Surface orientation $\beta$")
        ax.set_yticks([0, 0.5, 1, 1.5, 2])
        ax.axhline(1, color="white", lw=0.8, ls="--")
    axes[0].set_ylabel(r"Packing ratio $\phi_i^{\mathrm{wend}}/\phi_{\mathrm{ref}}$")
    cb = fig.colorbar(im, ax=axes, shrink=0.94, pad=0.03, ticks=[1, 2, 5, 10, 20, 50, 100])
    cb.ax.set_yticklabels(["1", "2", "5", "10", "20", "50", "100"])
    cb.set_label(r"Effective mass / particle mass")
    for ext in ["svg", "pdf", "png"]:
        fig.savefig(out / f"local_mass_scaling.{ext}", dpi=300)
    print("Mass ratio ranges:", [(float(a.min()), float(a.max())) for a in ratios])


if __name__ == "__main__":
    main()
