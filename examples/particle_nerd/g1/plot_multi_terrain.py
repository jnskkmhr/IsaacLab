# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Publication-quality PNG figures for the 6-terrain MPM vs. Particle Transolver benchmark.

Follows the paper-figure skill specification:
- Serif / STIX typography, 300 DPI PNG output, tight bounding box
- Despined top/right axes, no internal titles or chart junk
- Colorblind-safe Okabe-Ito palette with distinct line styles
- Unmixed legends: shared top legend holds ONLY the two primary solvers (+ rigid approach shade);
  bar plots, velocity command, and terrain profile each use their own dedicated subplot legend
- Gray shaded band (0-2 s) marks when the robot is still walking on the rigid approach platform
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

UPSTREAM_ISAACLAB = Path("/home/jkamohara/isaac/IsaacLab")
BROOT = UPSTREAM_ISAACLAB / "source/isaaclab_tasks/isaaclab_tasks/contrib/g1_29dof_mpm/benchmark"
MPM_CAMPAIGN_DATA = BROOT / "data/four_policy_20261022_v3"

FONT_SIZE = 11
DPI = 300

# Colorblind-safe PRGn (Purple / Green) palette — avoids default matplotlib blue/orange
C_MPM = "#7B3294"  # Deep Purple (Implicit MPM Reference)
C_TOPK = "#1B7837"  # Dark Forest Green (Particle NeRD Top-K, Ours)
C_FULL = "#A6DBA0"  # Light Mint Green (Particle NeRD Full)
C_RIGID = "#E2E6EA"  # Light gray shade for rigid approach platform (0 - 2.0 s)
C_SAND = "#D8C3A5"  # Sand layer fill
C_SURF = "#8C6D46"  # Sand surface line

TERRAIN_SHORT = {
    "flat": "Flat Bed\n(6×2 m)",
    "ascending": "Ascending\n(+21.8°)",
    "descending": "Descending\n(−21.8°)",
    "dune": "Sinusoidal\nDunes",
    "standing_tilt": "Standing Tilt\n(±15°)",
    "walking_tilt": "Walking Tilt\n(±10°)",
}

TERRAIN_ROW_LABELS = {
    "flat": "Flat Bed\n(6×2 m, 118k)",
    "ascending": "Ascending Ramp\n(+21.8°, 118k)",
    "descending": "Descending Ramp\n(−21.8°, 118k)",
    "dune": "Sinusoidal Dunes\n(h=0.3 m, 118k)",
    "standing_tilt": "Standing Tilt\n(±15°, 59k)",
    "walking_tilt": "Walking Tilt\n(±10°, 156k)",
}


def apply_publication_style():
    matplotlib.rcParams.update(
        {
            "font.size": FONT_SIZE,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.labelsize": FONT_SIZE,
            "axes.titlesize": FONT_SIZE,
            "xtick.labelsize": FONT_SIZE - 1,
            "ytick.labelsize": FONT_SIZE - 1,
            "legend.fontsize": FONT_SIZE - 1,
            "figure.dpi": DPI,
            "savefig.dpi": DPI,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.06,
            "axes.grid": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.9,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "text.usetex": False,
            "mathtext.fontset": "stix",
        }
    )


def terrain_surface_profile(terrain: str, x: np.ndarray) -> np.ndarray:
    u = np.clip(x - (-3.0), 0.0, 6.0)
    if terrain == "ascending":
        return 0.4 * u
    if terrain == "descending":
        return -0.4 * u
    if terrain == "dune":
        return 0.5 * 0.3 * (1.0 - np.cos(2.0 * np.pi * 2.0 * u / 6.0))
    return np.zeros_like(x)


def smooth(y: np.ndarray, k: int = 3) -> np.ndarray:
    if k <= 1 or len(y) < k:
        return y
    kernel = np.ones(k, dtype=np.float64) / k
    return np.convolve(y, kernel, mode="same")


def generate_all_figures(
    out_root: Path = Path("/home/jkamohara/isaac/IsaacLab-particle-nerd/video/multi_terrain"),
    terrains: list[str] = ("flat", "ascending", "descending", "dune", "standing_tilt", "walking_tilt"),
    material: str = "mild_sand",
    policy: str = "mpm_rough",
    seed: int = 42,
):
    apply_publication_style()
    plots_dir = out_root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    for old_f in list(plots_dir.glob("*.pdf")) + list(plots_dir.glob("*.svg")):
        old_f.unlink()

    speed_path = out_root / "speed_profile_graphed.json"
    speed_map = {}
    if speed_path.is_file():
        for r in json.loads(speed_path.read_text()):
            speed_map[(r["terrain"], r["mode"])] = r

    summary_rows = []
    traces = {}
    for terrain in terrains:
        local_mpm_dir = out_root / terrain / "mpm"
        mpm_dir = (
            local_mpm_dir
            if (local_mpm_dir / "trace.npz").is_file()
            else (MPM_CAMPAIGN_DATA / terrain / f"{material}__{policy}__seed{seed}")
        )
        nerd_dir = out_root / terrain
        if not (mpm_dir / "trace.npz").is_file() or not (nerd_dir / "trace.npz").is_file():
            continue
        mpm_meta = json.loads((mpm_dir / "metadata.json").read_text())
        nerd_meta = json.loads((nerd_dir / "metadata.json").read_text())
        mpm_tr = np.load(mpm_dir / "trace.npz")
        nerd_tr = np.load(nerd_dir / "trace.npz")
        traces[terrain] = (mpm_tr, nerd_tr, mpm_meta, nerd_meta)

        is_tilt = terrain in ("standing_tilt", "walking_tilt")
        t_cut = 0.5 if is_tilt else 2.0
        m_mask = mpm_tr["time"] >= t_cut
        n_mask = nerd_tr["time"] >= t_cut
        m_fz = mpm_tr["force"][:, :, 2].sum(axis=1)
        n_fz = nerd_tr["force"][:, :, 2].sum(axis=1)
        m_fx = mpm_tr["force"][:, :, 0].sum(axis=1)
        n_fx = nerd_tr["force"][:, :, 0].sum(axis=1)

        t_common = mpm_tr["time"][mpm_tr["time"] <= min(mpm_tr["time"][-1], nerd_tr["time"][-1])]
        rp_mpm_i = np.stack([np.interp(t_common, mpm_tr["time"], mpm_tr["root_position"][:, a]) for a in range(3)], -1)
        rp_nerd_i = np.stack(
            [np.interp(t_common, nerd_tr["time"], nerd_tr["root_position"][:, a]) for a in range(3)], -1
        )
        pos_diff = np.linalg.norm(rp_nerd_i - rp_mpm_i, axis=-1)

        m_vx_rmse = float(np.sqrt(np.mean((mpm_tr["linear_velocity"][m_mask, 0] - mpm_tr["command"][m_mask, 0]) ** 2)))
        n_vx_rmse = float(
            np.sqrt(np.mean((nerd_tr["linear_velocity"][n_mask, 0] - nerd_tr["command"][n_mask, 0]) ** 2))
        )

        sp_mpm = speed_map.get((terrain, "mpm"), {})
        sp_full = speed_map.get((terrain, "transolver_full"), {})
        sp_topk = speed_map.get((terrain, "transolver_topk"), {})

        def _true_coupled_step_ms(sp: dict) -> float | None:
            raw_c = sp.get("coupled_physics_step_ms")
            raw_s = sp.get("sand_solver_step_ms")
            if raw_c is None:
                return None
            # speed_profile_graphed.json divided a single 5 ms physics step (set_decimation(1)) by 4;
            # restore the true un-divided coupled physics step time (MJWarp + Material + Coupling).
            return float(raw_c * 4.0) if (raw_s is not None and raw_c < raw_s) else float(raw_c)

        mpm_phys_ms = _true_coupled_step_ms(sp_mpm)
        full_phys_ms = _true_coupled_step_ms(sp_full)
        topk_phys_ms = _true_coupled_step_ms(sp_topk)

        row_dict = dict(
            terrain=terrain,
            mpm_outcome=mpm_meta["outcome"],
            nerd_outcome=nerd_meta["outcome"],
            mpm_duration_s=float(mpm_meta["duration_s"]),
            nerd_duration_s=float(nerd_meta["duration_s"]),
            mpm_mean_fz_N=float(m_fz[m_mask].mean()),
            nerd_mean_fz_N=float(n_fz[n_mask].mean()),
            mpm_std_fz_N=float(m_fz[m_mask].std()),
            nerd_std_fz_N=float(n_fz[n_mask].std()),
            mpm_mean_fx_N=float(m_fx[m_mask].mean()),
            nerd_mean_fx_N=float(n_fx[n_mask].mean()),
            mpm_vx_rmse=m_vx_rmse,
            nerd_vx_rmse=n_vx_rmse,
            mean_root_path_deviation_m=float(pos_diff.mean()),
            end_root_z_mpm=float(mpm_tr["root_position"][-1, 2]),
            end_root_z_nerd=float(nerd_tr["root_position"][-1, 2]),
            mpm_phys_ms=mpm_phys_ms,
            full_phys_ms=full_phys_ms,
            topk_phys_ms=topk_phys_ms,
        )
        # #region agent log
        with open("/home/jkamohara/isaac/IsaacLab-particle-nerd/.cursor/debug-2657b4.log", "a") as _df:
            _df.write(
                json.dumps(
                    {
                        "sessionId": "2657b4",
                        "runId": "green_purple_plots",
                        "hypothesisId": "H_palette_and_bench",
                        "location": "plot_multi_terrain.py:198",
                        "message": "Terrain summary row with green/purple palette",
                        "data": row_dict,
                        "timestamp": int(time.time() * 1000),
                    }
                )
                + "\n"
            )
        # #endregion
        summary_rows.append(row_dict)

    # Shared top-level legend handles (ONLY the two solvers + rigid approach platform shade)
    shared_solver_handles = [
        plt.Line2D([], [], color=C_MPM, lw=2.2, label="Implicit MPM (Reference)"),
        plt.Line2D([], [], color=C_TOPK, lw=2.2, ls="--", label="Particle Transolver Top-$K$ (Ours)"),
        plt.Rectangle(
            (0, 0), 1, 1, facecolor=C_RIGID, edgecolor="0.75", alpha=0.85, label="Rigid Approach Platform ($0$–$2$ s)"
        ),
    ]

    # =========================================================================
    # FIGURE 1: grf_comparison_all_terrains.png (Simulation Fidelity + Timing)
    # =========================================================================
    n_t = len(summary_rows)
    fig = plt.figure(figsize=(13.5, 11.8), layout="constrained")
    gs = fig.add_gridspec(n_t + 1, 2, height_ratios=[1.0] * n_t + [1.35], hspace=0.14, wspace=0.14)

    for idx, row in enumerate(summary_rows):
        t_name = row["terrain"]
        is_tilt = t_name in ("standing_tilt", "walking_tilt")
        mpm_tr, nerd_tr, _, _ = traces[t_name]
        m_fz = smooth(mpm_tr["force"][:, :, 2].sum(axis=1), 5)
        n_fz = smooth(nerd_tr["force"][:, :, 2].sum(axis=1), 5)
        m_fx = smooth(mpm_tr["force"][:, :, 0].sum(axis=1), 5)
        n_fx = smooth(nerd_tr["force"][:, :, 0].sum(axis=1), 5)

        ax_z = fig.add_subplot(gs[idx, 0])
        ax_x = fig.add_subplot(gs[idx, 1])

        if not is_tilt:
            ax_z.axvspan(0.0, 2.0, color=C_RIGID, alpha=0.8, zorder=0)
            ax_x.axvspan(0.0, 2.0, color=C_RIGID, alpha=0.8, zorder=0)

        h_mg = ax_z.axhline(327.0, color="0.45", ls=":", lw=1.1, label="Robot Weight $mg = 327$ N", zorder=1)
        ax_z.plot(mpm_tr["time"], m_fz, color=C_MPM, lw=1.4, zorder=2)
        ax_z.plot(nerd_tr["time"], n_fz, color=C_TOPK, lw=1.4, ls="--", zorder=3)
        ax_z.set_ylim(-40, 820)
        ax_z.set_yticks([0, 327, 650])
        ax_z.set_ylabel(f"{TERRAIN_ROW_LABELS[t_name]}\n$F_z$ (N)", fontsize=9.5)
        if idx == 0:
            ax_z.legend(handles=[h_mg], loc="upper left", frameon=False, fontsize=8.8)
        ax_z.text(
            0.985,
            0.86,
            f"Sand Mean $F_z$: MPM {row['mpm_mean_fz_N']:.0f} N | Ours {row['nerd_mean_fz_N']:.0f} N",
            transform=ax_z.transAxes,
            ha="right",
            va="top",
            fontsize=8.8,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.85),
        )

        ax_x.axhline(0.0, color="0.45", ls=":", lw=1.0, zorder=1)
        ax_x.plot(mpm_tr["time"], m_fx, color=C_MPM, lw=1.4, zorder=2)
        ax_x.plot(nerd_tr["time"], n_fx, color=C_TOPK, lw=1.4, ls="--", zorder=3)
        ax_x.set_ylim(-240, 240)
        ax_x.set_yticks([-150, 0, 150])
        ax_x.set_ylabel("$F_x$ (N)", fontsize=9.5)
        ax_x.text(
            0.985,
            0.86,
            f"Sand Mean $F_x$: MPM {row['mpm_mean_fx_N']:+.1f} N | Ours {row['nerd_mean_fx_N']:+.1f} N",
            transform=ax_x.transAxes,
            ha="right",
            va="top",
            fontsize=8.8,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.85),
        )

        if idx == n_t - 1:
            ax_z.set_xlabel("Episode Time (s)")
            ax_x.set_xlabel("Episode Time (s)")

    # Bottom Row: Single Wall-Clock Time per Physics Step Benchmark
    ax_b = fig.add_subplot(gs[n_t, :])
    x_pos = np.arange(n_t)
    w = 0.25
    short_names = [TERRAIN_SHORT[r["terrain"]] for r in summary_rows]

    v_mpm = [r["mpm_phys_ms"] or 0.0 for r in summary_rows]
    v_full = [r["full_phys_ms"] or 0.0 for r in summary_rows]
    v_topk = [r["topk_phys_ms"] or 0.0 for r in summary_rows]
    b1 = ax_b.bar(x_pos - w, v_mpm, width=w, color=C_MPM, label="Implicit MPM")
    b2 = ax_b.bar(x_pos, v_full, width=w, color=C_FULL, edgecolor="#4D9221", linewidth=0.7, label="Particle NeRD Full")
    b3 = ax_b.bar(x_pos + w, v_topk, width=w, color=C_TOPK, label="Particle NeRD Top-$K$ (Ours)")
    ax_b.set_xticks(x_pos, short_names, fontsize=9.2)
    ax_b.set_ylabel("Wall-Clock Time /\nPhysics Step (ms)", fontsize=9.5)
    ymax = max(max(v_full), max(v_mpm)) * 1.32
    ax_b.set_ylim(0, ymax)
    ax_b.legend(loc="upper left", ncol=3, frameon=False, fontsize=9.0)
    for bar_m, bar_f, bar_k in zip(b1, b2, b3):
        hm, hf, hk = bar_m.get_height(), bar_f.get_height(), bar_k.get_height()
        ax_b.text(
            bar_m.get_x() + bar_m.get_width() / 2.0,
            hm + 0.015 * ymax,
            f"{hm:.2f}",
            ha="center",
            va="bottom",
            fontsize=8.0,
            color=C_MPM,
        )
        ax_b.text(
            bar_f.get_x() + bar_f.get_width() / 2.0,
            hf + 0.015 * ymax,
            f"{hf:.2f}",
            ha="center",
            va="bottom",
            fontsize=8.0,
            color="0.35",
        )
        if hk > 0:
            sp = hm / hk
            ax_b.text(
                bar_k.get_x() + bar_k.get_width() * 0.58,
                hk + 0.02 * ymax,
                f"{sp:.1f}×\n({hk:.2f})",
                ha="center",
                va="bottom",
                fontsize=8.0,
                fontweight="bold",
                color=C_TOPK,
            )

    fig.legend(handles=shared_solver_handles, loc="outside upper center", ncol=3, frameon=False, fontsize=10.5)
    fig.savefig(plots_dir / "grf_comparison_all_terrains.png", dpi=DPI)
    plt.close(fig)

    # =========================================================================
    # FIGURE 2: root_trajectory_comparison_all_terrains.png (Policy Response + Timing)
    # =========================================================================
    fig = plt.figure(figsize=(14.5, 11.8), layout="constrained")
    gs_t = fig.add_gridspec(n_t + 1, 3, height_ratios=[1.0] * n_t + [1.35], hspace=0.14, wspace=0.14)
    for idx, row in enumerate(summary_rows):
        t_name = row["terrain"]
        is_tilt = t_name in ("standing_tilt", "walking_tilt")
        mpm_tr, nerd_tr, _, _ = traces[t_name]

        # Col 0: Forward Root Position x(t) (1-to-1 clean comparison, no secondary lines)
        ax0 = fig.add_subplot(gs_t[idx, 0])
        if not is_tilt:
            ax0.axvspan(0.0, 2.0, color=C_RIGID, alpha=0.8, zorder=0)
        ax0.plot(mpm_tr["time"], mpm_tr["root_position"][:, 0], color=C_MPM, lw=1.6, zorder=2)
        ax0.plot(nerd_tr["time"], nerd_tr["root_position"][:, 0], color=C_TOPK, lw=1.6, ls="--", zorder=3)
        ax0.set_ylabel(f"{TERRAIN_ROW_LABELS[t_name]}\nRoot $x(t)$ (m)", fontsize=9.5)
        ax0.text(
            0.03,
            0.86,
            f"Mean 3D path diff: {row['mean_root_path_deviation_m']:.2f} m",
            transform=ax0.transAxes,
            ha="left",
            va="top",
            fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.85),
        )

        # Col 1: Sagittal Elevation z(x) or Pelvis Height z(t)
        ax1 = fig.add_subplot(gs_t[idx, 1])
        if not is_tilt:
            ax1.axvspan(-4.5, -3.0, color=C_RIGID, alpha=0.8, zorder=0)
            xs = np.linspace(-3.0, 3.0, 200)
            zs = terrain_surface_profile(t_name, xs)
            h_sand = ax1.fill_between(xs, zs - 0.30, zs, color=C_SAND, alpha=0.5, label="0.3 m Sand Bed", zorder=1)
            ax1.plot(xs, zs, color=C_SURF, lw=1.2, zorder=1)
            ax1.plot(mpm_tr["root_position"][:, 0], mpm_tr["root_position"][:, 2], color=C_MPM, lw=1.7, zorder=2)
            ax1.plot(
                nerd_tr["root_position"][:, 0], nerd_tr["root_position"][:, 2], color=C_TOPK, lw=1.7, ls="--", zorder=3
            )
            ax1.set_ylabel("Elevation $z(x)$ (m)", fontsize=9.5)
            if idx == 0:
                ax1.legend(handles=[h_sand], loc="lower right", frameon=False, fontsize=8.5)
            if idx == 3:
                ax1.set_xlabel("World Position $x$ (m)", fontsize=9.5)
        else:
            ax1.plot(mpm_tr["time"], mpm_tr["root_position_reference"][:, 2], color=C_MPM, lw=1.6, zorder=2)
            ax1.plot(nerd_tr["time"], nerd_tr["root_position_reference"][:, 2], color=C_TOPK, lw=1.6, ls="--", zorder=3)
            ax1.set_ylabel("Pelvis Height $z(t)$ (m)", fontsize=9.5)
            if idx == n_t - 1:
                ax1.set_xlabel("Episode Time (s)", fontsize=9.5)

        # Col 2: Forward Velocity Tracking vx(t) with Dedicated Command Legend
        ax2 = fig.add_subplot(gs_t[idx, 2])
        if not is_tilt:
            ax2.axvspan(0.0, 2.0, color=C_RIGID, alpha=0.8, zorder=0)
        (h_cmd,) = ax2.plot(
            mpm_tr["time"],
            mpm_tr["command"][:, 0],
            color="0.2",
            ls=":",
            lw=1.3,
            label="Command $v_x^{\\mathrm{cmd}}$",
            zorder=1,
        )
        ax2.plot(mpm_tr["time"], smooth(mpm_tr["linear_velocity"][:, 0], 3), color=C_MPM, lw=1.4, zorder=2)
        ax2.plot(nerd_tr["time"], smooth(nerd_tr["linear_velocity"][:, 0], 3), color=C_TOPK, lw=1.4, ls="--", zorder=3)
        ax2.set_ylabel("Vel $v_x$ (m/s)", fontsize=9.5)
        if idx in (0, 5):
            ax2.legend(handles=[h_cmd], loc="upper left", frameon=False, fontsize=8.5)
        ax2.text(
            0.98,
            0.14,
            f"$v_x$ RMSE: MPM {row['mpm_vx_rmse']:.2f} | Ours {row['nerd_vx_rmse']:.2f} m/s",
            transform=ax2.transAxes,
            ha="right",
            va="bottom",
            fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.85),
        )

        if idx == n_t - 1:
            ax0.set_xlabel("Episode Time (s)", fontsize=9.5)
            ax2.set_xlabel("Episode Time (s)", fontsize=9.5)

    # Bottom Row of Figure 2: Single Wall-Clock Time / Physics Step + Velocity Tracking Error
    ax_tb1 = fig.add_subplot(gs_t[n_t, :2])
    ax_tb2 = fig.add_subplot(gs_t[n_t, 2])
    x_pos = np.arange(n_t)
    w = 0.25
    short_names = [TERRAIN_SHORT[r["terrain"]] for r in summary_rows]

    v_mpm = [r["mpm_phys_ms"] or 0.0 for r in summary_rows]
    v_full = [r["full_phys_ms"] or 0.0 for r in summary_rows]
    v_topk = [r["topk_phys_ms"] or 0.0 for r in summary_rows]
    b1 = ax_tb1.bar(x_pos - w, v_mpm, width=w, color=C_MPM, label="Implicit MPM")
    b2 = ax_tb1.bar(
        x_pos, v_full, width=w, color=C_FULL, edgecolor="#4D9221", linewidth=0.7, label="Particle NeRD Full"
    )
    b3 = ax_tb1.bar(x_pos + w, v_topk, width=w, color=C_TOPK, label="Particle NeRD Top-$K$ (Ours)")
    ax_tb1.set_xticks(x_pos, short_names, fontsize=8.8)
    ax_tb1.set_ylabel("Wall-Clock Time /\nPhysics Step (ms)", fontsize=9.5)
    ymax = max(max(v_full), max(v_mpm)) * 1.34
    ax_tb1.set_ylim(0, ymax)
    ax_tb1.legend(loc="upper left", ncol=3, frameon=False, fontsize=8.5)
    for bar_m, bar_f, bar_k in zip(b1, b2, b3):
        hm, hf, hk = bar_m.get_height(), bar_f.get_height(), bar_k.get_height()
        ax_tb1.text(
            bar_m.get_x() + bar_m.get_width() / 2.0,
            hm + 0.015 * ymax,
            f"{hm:.2f}",
            ha="center",
            va="bottom",
            fontsize=7.8,
            color=C_MPM,
        )
        ax_tb1.text(
            bar_f.get_x() + bar_f.get_width() / 2.0,
            hf + 0.015 * ymax,
            f"{hf:.2f}",
            ha="center",
            va="bottom",
            fontsize=7.8,
            color="0.35",
        )
        if hk > 0:
            ax_tb1.text(
                bar_k.get_x() + bar_k.get_width() * 0.58,
                hk + 0.02 * ymax,
                f"{hm / hk:.1f}×\n({hk:.2f})",
                ha="center",
                va="bottom",
                fontsize=7.8,
                fontweight="bold",
                color=C_TOPK,
            )

    v_rmse_mpm = [r["mpm_vx_rmse"] for r in summary_rows]
    v_rmse_topk = [r["nerd_vx_rmse"] for r in summary_rows]
    ax_tb2.bar(x_pos - 0.17, v_rmse_mpm, width=0.32, color=C_MPM, label="Implicit MPM")
    ax_tb2.bar(x_pos + 0.17, v_rmse_topk, width=0.32, color=C_TOPK, label="Top-$K$ (Ours)")
    ax_tb2.set_xticks(x_pos, short_names, fontsize=8.2)
    ax_tb2.set_ylabel("Forward Velocity\nRMSE (m/s)", fontsize=9.5)
    ax_tb2.set_ylim(0, max(max(v_rmse_mpm), max(v_rmse_topk)) * 1.30)
    ax_tb2.legend(loc="upper right", ncol=2, frameon=False, fontsize=8.2)

    fig.legend(handles=shared_solver_handles, loc="outside upper center", ncol=3, frameon=False, fontsize=10.5)
    fig.savefig(plots_dir / "root_trajectory_comparison_all_terrains.png", dpi=DPI)
    plt.close(fig)

    # =========================================================================
    # FIGURE 3: speed_benchmark_all_terrains.png (Dedicated Speed Comparison)
    # =========================================================================
    fig, ax_b = plt.subplots(1, 1, figsize=(8.2, 4.2), layout="constrained")
    x_pos = np.arange(n_t)
    w = 0.25
    short_names = [TERRAIN_SHORT[r["terrain"]] for r in summary_rows]

    v_mpm = [r["mpm_phys_ms"] or 0.0 for r in summary_rows]
    v_full = [r["full_phys_ms"] or 0.0 for r in summary_rows]
    v_topk = [r["topk_phys_ms"] or 0.0 for r in summary_rows]
    b1 = ax_b.bar(x_pos - w, v_mpm, width=w, color=C_MPM, label="Implicit MPM (CUDA Graph)")
    b2 = ax_b.bar(
        x_pos,
        v_full,
        width=w,
        color=C_FULL,
        edgecolor="#4D9221",
        linewidth=0.7,
        label="Particle NeRD Full (CUDA Graph)",
    )
    b3 = ax_b.bar(x_pos + w, v_topk, width=w, color=C_TOPK, label="Particle NeRD Top-$K$ (Ours, CUDA Graph)")
    ax_b.set_xticks(x_pos, short_names, fontsize=9.5)
    ax_b.set_ylabel("Wall-Clock Time / Physics Step (ms)", fontsize=10.5)
    ymax = max(max(v_full), max(v_mpm)) * 1.32
    ax_b.set_ylim(0, ymax)
    ax_b.legend(loc="upper left", frameon=False, fontsize=9.0)
    for bar_m, bar_f, bar_k in zip(b1, b2, b3):
        hm, hf, hk = bar_m.get_height(), bar_f.get_height(), bar_k.get_height()
        ax_b.text(
            bar_m.get_x() + bar_m.get_width() / 2.0,
            hm + 0.015 * ymax,
            f"{hm:.2f}",
            ha="center",
            va="bottom",
            fontsize=8.2,
            color=C_MPM,
        )
        ax_b.text(
            bar_f.get_x() + bar_f.get_width() / 2.0,
            hf + 0.015 * ymax,
            f"{hf:.2f}",
            ha="center",
            va="bottom",
            fontsize=8.2,
            color="0.35",
        )
        if hk > 0:
            sp = hm / hk
            ax_b.text(
                bar_k.get_x() + bar_k.get_width() * 0.58,
                hk + 0.015 * ymax,
                f"{sp:.1f}×\n({hk:.2f})",
                ha="center",
                va="bottom",
                fontsize=8.4,
                fontweight="bold",
                color=C_TOPK,
            )

    fig.savefig(plots_dir / "speed_benchmark_all_terrains.png", dpi=DPI)
    plt.close(fig)

    # =========================================================================
    # FIGURE 4: Per-Terrain 2x3 Detailed Figures (<terrain>_mpm_vs_transolver.png)
    # =========================================================================
    for row in summary_rows:
        t_name = row["terrain"]
        is_tilt = t_name in ("standing_tilt", "walking_tilt")
        mpm_tr, nerd_tr, _, _ = traces[t_name]
        m_fz = smooth(mpm_tr["force"][:, :, 2].sum(axis=1), 5)
        n_fz = smooth(nerd_tr["force"][:, :, 2].sum(axis=1), 5)
        m_fx = smooth(mpm_tr["force"][:, :, 0].sum(axis=1), 5)
        n_fx = smooth(nerd_tr["force"][:, :, 0].sum(axis=1), 5)

        fig, axes = plt.subplots(2, 3, figsize=(13.8, 6.5), layout="constrained")

        # (0, 0): Total Vertical GRF Fz(t)
        ax = axes[0, 0]
        if not is_tilt:
            ax.axvspan(0.0, 2.0, color=C_RIGID, alpha=0.8, zorder=0)
        h_mg = ax.axhline(327.0, color="0.45", ls=":", lw=1.1, label="Weight $mg = 327$ N", zorder=1)
        ax.plot(mpm_tr["time"], m_fz, color=C_MPM, lw=1.5, zorder=2)
        ax.plot(nerd_tr["time"], n_fz, color=C_TOPK, lw=1.5, ls="--", zorder=3)
        ax.set_xlabel("Episode Time (s)")
        ax.set_ylabel("Total Vertical GRF $F_z$ (N)")
        ax.set_ylim(-40, 820)
        ax.legend(handles=[h_mg], loc="upper left", frameon=False, fontsize=8.8)
        ax.text(
            0.97,
            0.92,
            f"Mean $F_z$: MPM {row['mpm_mean_fz_N']:.1f} N | Ours {row['nerd_mean_fz_N']:.1f} N",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8.8,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.85),
        )

        # (0, 1): Longitudinal Shear GRF Fx(t)
        ax = axes[0, 1]
        if not is_tilt:
            ax.axvspan(0.0, 2.0, color=C_RIGID, alpha=0.8, zorder=0)
        ax.axhline(0.0, color="0.45", ls=":", lw=1.0, zorder=1)
        ax.plot(mpm_tr["time"], m_fx, color=C_MPM, lw=1.5, zorder=2)
        ax.plot(nerd_tr["time"], n_fx, color=C_TOPK, lw=1.5, ls="--", zorder=3)
        ax.set_xlabel("Episode Time (s)")
        ax.set_ylabel("Longitudinal Shear GRF $F_x$ (N)")
        ax.set_ylim(-240, 240)
        ax.text(
            0.97,
            0.92,
            f"Mean $F_x$: MPM {row['mpm_mean_fx_N']:+.1f} N | Ours {row['nerd_mean_fx_N']:+.1f} N",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8.8,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.85),
        )

        # (0, 2): Wall-Clock Time / Physics Step (ms) Comparison
        ax = axes[0, 2]
        m_val = row["mpm_phys_ms"] or 0.0
        f_val = row["full_phys_ms"] or 0.0
        k_val = row["topk_phys_ms"] or 0.0
        bars = ax.bar(
            [0, 1, 2],
            [m_val, f_val, k_val],
            width=0.52,
            color=[C_MPM, C_FULL, C_TOPK],
        )
        ax.set_xticks([0, 1, 2], ["Implicit MPM", "NeRD Full", "NeRD Top-$K$\n(Ours)"])
        ax.set_ylabel("Wall-Clock Time / Physics Step (ms)")
        ym = max(m_val, f_val, k_val) * 1.35
        ax.set_ylim(0, ym)
        ax.text(
            bars[0].get_x() + bars[0].get_width() / 2,
            m_val + 0.02 * ym,
            f"{m_val:.2f} ms",
            ha="center",
            va="bottom",
            fontsize=8.2,
            color=C_MPM,
        )
        ax.text(
            bars[1].get_x() + bars[1].get_width() / 2,
            f_val + 0.02 * ym,
            f"{f_val:.2f} ms",
            ha="center",
            va="bottom",
            fontsize=8.2,
            color="0.35",
        )
        if k_val > 0:
            ax.text(
                bars[2].get_x() + bars[2].get_width() / 2,
                k_val + 0.02 * ym,
                f"{m_val / k_val:.1f}× ({k_val:.2f} ms)",
                ha="center",
                va="bottom",
                fontsize=8.4,
                fontweight="bold",
                color=C_TOPK,
            )

        # (1, 0): Forward Root Base Position x(t)
        ax = axes[1, 0]
        if not is_tilt:
            ax.axvspan(0.0, 2.0, color=C_RIGID, alpha=0.8, zorder=0)
        ax.plot(mpm_tr["time"], mpm_tr["root_position"][:, 0], color=C_MPM, lw=1.6, zorder=2)
        ax.plot(nerd_tr["time"], nerd_tr["root_position"][:, 0], color=C_TOPK, lw=1.6, ls="--", zorder=3)
        ax.set_xlabel("Episode Time (s)")
        ax.set_ylabel("Root Base Position $x(t)$ (m)")
        ax.text(
            0.03,
            0.92,
            f"Mean 3D path diff: {row['mean_root_path_deviation_m']:.2f} m",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.8,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.85),
        )

        # (1, 1): Sagittal Elevation z(x) or Pelvis Height z(t)
        ax = axes[1, 1]
        if not is_tilt:
            ax.axvspan(-4.5, -3.0, color=C_RIGID, alpha=0.8, zorder=0)
            xs = np.linspace(-3.0, 3.0, 200)
            zs = terrain_surface_profile(t_name, xs)
            h_sand = ax.fill_between(xs, zs - 0.30, zs, color=C_SAND, alpha=0.5, label="0.3 m Sand Bed", zorder=1)
            ax.plot(xs, zs, color=C_SURF, lw=1.3, zorder=1)
            ax.plot(mpm_tr["root_position"][:, 0], mpm_tr["root_position"][:, 2], color=C_MPM, lw=1.8, zorder=2)
            ax.plot(
                nerd_tr["root_position"][:, 0], nerd_tr["root_position"][:, 2], color=C_TOPK, lw=1.8, ls="--", zorder=3
            )
            ax.set_xlabel("World Position $x$ (m)")
            ax.set_ylabel("Root Elevation $z(x)$ (m)")
            ax.legend(handles=[h_sand], loc="lower right", frameon=False, fontsize=8.8)
        else:
            ax.plot(mpm_tr["time"], mpm_tr["root_position_reference"][:, 2], color=C_MPM, lw=1.6, zorder=2)
            ax.plot(nerd_tr["time"], nerd_tr["root_position_reference"][:, 2], color=C_TOPK, lw=1.6, ls="--", zorder=3)
            ax.set_xlabel("Episode Time (s)")
            ax.set_ylabel("Pelvis Height $z(t)$ (m)")

        # (1, 2): Forward Velocity Tracking vx(t) with Dedicated Command Legend
        ax = axes[1, 2]
        if not is_tilt:
            ax.axvspan(0.0, 2.0, color=C_RIGID, alpha=0.8, zorder=0)
        (h_cmd,) = ax.plot(
            mpm_tr["time"],
            mpm_tr["command"][:, 0],
            color="0.2",
            ls=":",
            lw=1.3,
            label="Command $v_x^{\\mathrm{cmd}}$",
            zorder=1,
        )
        ax.plot(mpm_tr["time"], smooth(mpm_tr["linear_velocity"][:, 0], 3), color=C_MPM, lw=1.5, zorder=2)
        ax.plot(nerd_tr["time"], smooth(nerd_tr["linear_velocity"][:, 0], 3), color=C_TOPK, lw=1.5, ls="--", zorder=3)
        ax.set_xlabel("Episode Time (s)")
        ax.set_ylabel("Forward Velocity $v_x$ (m/s)")
        ax.legend(handles=[h_cmd], loc="upper left", frameon=False, fontsize=8.8)
        ax.text(
            0.97,
            0.14,
            f"$v_x$ RMSE: MPM {row['mpm_vx_rmse']:.2f} | Ours {row['nerd_vx_rmse']:.2f} m/s",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8.8,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.85),
        )

        handles_det = shared_solver_handles[:2] if is_tilt else shared_solver_handles
        fig.legend(handles=handles_det, loc="outside upper center", ncol=len(handles_det), frameon=False, fontsize=10.2)
        fig.savefig(plots_dir / f"{t_name}_mpm_vs_transolver.png", dpi=DPI)
        plt.close(fig)

    # =========================================================================
    # FIGURE 5: env_scaling_benchmark.png (Multi-Environment Scaling 1..256 Envs)
    # =========================================================================
    scaling_path = out_root / "env_scaling.json"
    if scaling_path.is_file():
        s_rows = json.loads(scaling_path.read_text())
        env_counts = [1, 4, 16, 64, 128, 256]
        by_key = {(r["num_envs"], r["mode"]): r for r in s_rows}

        fig, (ax_tot, ax_per) = plt.subplots(1, 2, figsize=(12.8, 4.3), layout="constrained")
        x_idx = np.arange(len(env_counts))
        w = 0.26
        x_labels = [f"{e}\n({e * 16.4:.0f}k)" if e < 64 else f"{e}\n({e * 16.384 / 1000:.1f}M)" for e in env_counts]

        for ax_sc, per_env, ylabel_sc in [
            (ax_tot, False, "Total Wall-Clock Time / Physics Step (ms)"),
            (ax_per, True, "Amortized Wall-Clock Time / Env / Physics Step (ms)"),
        ]:
            vals_m, vals_f, vals_k = [], [], []
            for e in env_counts:
                for m, dst in [("mpm", vals_m), ("transolver_full", vals_f), ("transolver_topk", vals_k)]:
                    r = by_key.get((e, m))
                    if r and r.get("status") == "OK" and r.get("step_ms") is not None:
                        val = float(r["step_per_env_ms"]) if per_env else float(r["step_ms"])
                        dst.append(val)
                    else:
                        dst.append(0.0)

            bm = ax_sc.bar(x_idx - w, vals_m, width=w, color=C_MPM, label="Implicit MPM")
            bf = ax_sc.bar(
                x_idx, vals_f, width=w, color=C_FULL, edgecolor="#4D9221", linewidth=0.7, label="Particle NeRD Full"
            )
            bk = ax_sc.bar(x_idx + w, vals_k, width=w, color=C_TOPK, label="Particle NeRD Top-$K$ (Ours)")
            ax_sc.set_xticks(x_idx, x_labels, fontsize=9.2)
            ax_sc.set_xlabel("Number of Parallel Environments (Total Particles)")
            ax_sc.set_ylabel(ylabel_sc)
            ymax = max(max(vals_m), max(vals_f), max(vals_k)) * 1.30
            ax_sc.set_ylim(0, ymax)
            ax_sc.legend(loc="upper left" if not per_env else "upper right", frameon=False, fontsize=8.8)

            for i, e in enumerate(env_counts):
                hm, hf, hk = vals_m[i], vals_f[i], vals_k[i]
                if hm > 0:
                    ax_sc.text(
                        bm[i].get_x() + bm[i].get_width() / 2,
                        hm + 0.015 * ymax,
                        f"{hm:.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=7.8,
                        color=C_MPM,
                    )
                if hf > 0:
                    ax_sc.text(
                        bf[i].get_x() + bf[i].get_width() / 2,
                        hf + 0.015 * ymax,
                        f"{hf:.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=7.8,
                        color="0.35",
                    )
                if hk > 0:
                    sp_str = f"\n({hm / hk:.1f}×)" if hm > 0 else ""
                    ax_sc.text(
                        bk[i].get_x() + bk[i].get_width() * (0.62 if hm > 0 else 0.5),
                        hk + 0.015 * ymax,
                        f"{hk:.2f}{sp_str}",
                        ha="center",
                        va="bottom",
                        fontsize=8.0,
                        fontweight="bold",
                        color=C_TOPK,
                    )

        fig.savefig(plots_dir / "env_scaling_benchmark.png", dpi=DPI)
        plt.close(fig)

    (out_root / "summary.json").write_text(json.dumps(summary_rows, indent=2) + "\n")
    print(f"Generated publication-grade PNG figures in {plots_dir}", flush=True)


if __name__ == "__main__":
    generate_all_figures()
