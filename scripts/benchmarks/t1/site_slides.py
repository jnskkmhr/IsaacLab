# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# ruff: noqa: E501
"""Build the T1 slides (``site/T1``) and the result tables from the benchmark JSON/NPZ files.

Reads what ``run_sweeps.py`` and ``bench_g1_lemniscate.py`` wrote and shows whatever exists; a criterion without
data is shown as "not done". Rerun after new results. Needs only numpy and matplotlib.

    uv run python scripts/benchmarks/t1/site_slides.py --logs logs/t1/rtx5090 \
        --site ~/RSS_27/RSS_MPM_Project_Skill/site/T1 --tables logs/t1/rtx5090/tables.md
"""

from __future__ import annotations

import argparse
import glob
import html
import json
import os
import subprocess

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

TERRAINS = ("full", "stream", "stream_cache")
SLEEPS = ("off", "on")
MODES = [(terrain, sleep) for terrain in TERRAINS for sleep in SLEEPS]
TERRAIN_LABEL = {"full": "full", "stream": "stream without cache", "stream_cache": "stream with cache"}
COLOR = {"full": "#333333", "stream": "#D55E00", "stream_cache": "#0072B2"}
LENGTHS = (3.2, 6.4, 12.8, 25.6, 51.2, 102.4)
ENV_COUNTS = (1, 2, 4, 8, 16, 32, 64, 128, 256)
CELL = 0.04
CONTROL_DT = 0.02


def label(terrain: str, sleep: str) -> str:
    return f"{TERRAIN_LABEL[terrain]}, sleep {sleep}"


def load(path: str):
    try:
        with open(path) as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


# --------------------------------------------------------------------------------------
# Cost sweeps (P1 and P2)
# --------------------------------------------------------------------------------------
def collect(logs: str, prefix: str, key: str, values) -> dict:
    """``{(terrain, sleep): [(x, plain, profile), ...]}`` for the runs that exist, in sweep order."""
    data = {}
    for terrain, sleep in MODES:
        rows = []
        for value in values:
            plain = load(os.path.join(logs, f"{prefix}_{terrain}_sleep_{sleep}_{key}{value}_plain.json"))
            if plain is None:
                continue
            profile = load(os.path.join(logs, f"{prefix}_{terrain}_sleep_{sleep}_{key}{value}_profile.json"))
            rows.append((value, plain, profile if profile and profile.get("status") == "ok" else None))
        data[(terrain, sleep)] = rows
    return data


def cost_rows(data: dict) -> list[dict]:
    rows = []
    for (terrain, sleep), runs in data.items():
        for value, plain, profile in runs:
            row = {
                "terrain": terrain,
                "sleep": sleep,
                "x": value,
                "status": plain["status"],
                "error": plain.get("error", ""),
            }
            if plain["status"] == "ok":
                envs = plain["num_envs"]
                util = plain["gpu_utilization_percent"]
                row.update(
                    envs=envs,
                    particles=plain["particle_count"],
                    vram=plain["peak_process_MiB"],
                    vram_env=plain["peak_process_MiB"] / envs,
                    step=plain["step_ms"]["mean"],
                    p95=plain["step_ms"]["p95"],
                    step_env=plain["step_ms"]["mean"] / envs,
                    steps=plain["step_ms"]["n"],
                    util=util.get("mean"),
                    util_max=util.get("max"),
                    util_n=util.get("n"),
                    idle=plain.get("gpu_utilization_idle_percent", {}).get("mean"),
                    physics=profile["physics_ms"]["mean"] if profile else None,
                    physics_env=profile["physics_ms"]["mean"] / envs if profile else None,
                    activity=plain.get("activity_asleep_support_awake_last"),
                    cache=plain.get("cache_usage"),
                    cache_cfg=(plain.get("geometry") or {}).get("cache"),
                )
            rows.append(row)
    return rows


def cost_figure(rows: list[dict], xlabel: str, per_env: bool, path: str) -> None:
    panels = (
        ("vram_env" if per_env else "vram", "peak VRAM per env [MiB]" if per_env else "peak VRAM [MiB]", True),
        ("step_env" if per_env else "step", "step time per env [ms]" if per_env else "step time [ms]", True),
        (
            "physics_env" if per_env else "physics",
            "physics time per env [ms]" if per_env else "physics time per step [ms]",
            True,
        ),
        ("util", "GPU utilization, mean [%]", False),
    )
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.4))
    for axis, (key, ylabel, logy) in zip(axes, panels):
        for terrain, sleep in MODES:
            points = [
                (r["x"], r[key])
                for r in rows
                if r["terrain"] == terrain and r["sleep"] == sleep and r["status"] == "ok" and r.get(key) is not None
            ]
            if not points:
                continue
            xs, ys = zip(*points)
            axis.plot(
                xs,
                ys,
                marker="o" if sleep == "off" else "s",
                linestyle="-" if sleep == "off" else "--",
                color=COLOR[terrain],
                label=label(terrain, sleep),
                markersize=5,
                linewidth=1.6,
                markerfacecolor=COLOR[terrain] if sleep == "off" else "white",
            )
            failed = [r for r in rows if r["terrain"] == terrain and r["sleep"] == sleep and r["status"] != "ok"]
            if failed and key in ("vram", "vram_env"):
                axis.axvline(
                    failed[0]["x"],
                    color=COLOR[terrain],
                    linestyle=":" if sleep == "off" else "-.",
                    linewidth=1.0,
                    alpha=0.7,
                )
        axis.set_xscale("log", base=2)
        # Label the x axis with the swept values themselves, not powers of two.
        values = sorted({r["x"] for r in rows})
        axis.set_xticks(values)
        axis.set_xticklabels([f"{v:g}" for v in values])
        axis.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        if logy:
            axis.set_yscale("log")
            # Plain numbers at 1, 2 and 5 per decade, so every panel shows several labelled values.
            axis.yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
            axis.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
            axis.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        else:
            axis.set_ylim(0, 105)
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        axis.grid(True, which="both", alpha=0.25)
    axes[0].legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def terrain_figure(path: str) -> None:
    """Top view of the corridors of the terrain-size sweep: the width is fixed and the length doubles."""
    from matplotlib.patches import Rectangle

    width, pool = 1.6, 1.6
    fig, axis = plt.subplots(figsize=(15, 4.6))
    gap = 2.6
    for row, length in enumerate(LENGTHS):
        y = -row * gap
        # full simulates the whole corridor; the streaming pool is the square at the box
        axis.add_patch(Rectangle((0, y), length, width, facecolor="#e6d3a3", edgecolor=COLOR["full"], linewidth=1.2))
        axis.add_patch(
            Rectangle((0, y), pool, width, facecolor=COLOR["stream_cache"], alpha=0.35, edgecolor=COLOR["stream_cache"])
        )
        axis.add_patch(Rectangle((0.7, y + 0.7), 0.2, 0.2, facecolor="#2b4c9b", edgecolor="none"))
        axis.annotate(
            "",
            xy=(length - 0.8, y + 0.8),
            xytext=(0.95, y + 0.8),
            arrowprops={"arrowstyle": "->", "color": "#444444", "lw": 1.0},
        )
        axis.text(
            length + 1.0,
            y + 0.8,
            f"length_x = {length:g} m   (box travels {length - 1.6:g} m)",
            va="center",
            fontsize=9,
        )
    axis.text(-1.0, 0.8, "1.6 m", rotation=90, va="center", ha="right", fontsize=9)
    axis.plot([-0.6, -0.6], [0, width], color="#444444", linewidth=1.0)
    axis.set_xlim(-3, 150)
    axis.set_ylim(-(len(LENGTHS) - 1) * gap - 1.0, width + 1.4)
    axis.set_xlabel("x [m]  (to scale; the vertical direction is enlarged about 4 times)")
    axis.set_yticks([])
    for side in ("left", "right", "top"):
        axis.spines[side].set_visible(False)
    axis.set_title(
        "Terrain of the size sweep, seen from above: width fixed at 1.6 m, length doubling. "
        "Sand colour: simulated by full. Blue square: the 1.6 m streaming pool, which follows the box.",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def envs_figure(path: str) -> None:
    """Top view of one environment of the environment-count sweep, and how the environments are laid out."""
    from matplotlib.patches import Rectangle

    counts = (1, 2, 4, 8, 16, 32, 64)
    fig = plt.figure(figsize=(15, 4.6))
    grid = fig.add_gridspec(1, len(counts) + 3)
    axis = fig.add_subplot(grid[0, :3])
    axis.add_patch(Rectangle((-4, -4), 8, 8, facecolor="#e6d3a3", edgecolor=COLOR["full"], linewidth=1.2))
    axis.add_patch(
        Rectangle((-4.0, -0.8), 1.6, 1.6, facecolor=COLOR["stream_cache"], alpha=0.35, edgecolor=COLOR["stream_cache"])
    )
    axis.add_patch(Rectangle((-3.3, -0.1), 0.2, 0.2, facecolor="#2b4c9b", edgecolor="none"))
    axis.annotate("", xy=(3.2, 0.0), xytext=(-3.0, 0.0), arrowprops={"arrowstyle": "->", "color": "#444444", "lw": 1.0})
    axis.text(0.0, 0.35, "box travels 6.4 m", ha="center", fontsize=9)
    axis.text(0.0, -4.6, "8 m", ha="center", fontsize=9)
    axis.text(-4.6, 0.0, "8 m", rotation=90, va="center", fontsize=9)
    axis.set_xlim(-5.2, 4.6)
    axis.set_ylim(-5.2, 4.6)
    axis.set_aspect("equal")
    axis.axis("off")
    axis.set_title("One environment, to scale", fontsize=10)
    for index, count in enumerate(counts):
        axis = fig.add_subplot(grid[0, index + 3])
        columns = int(np.ceil(np.sqrt(count)))
        rows = int(np.ceil(count / columns))
        for env in range(count):
            x, y = env % columns, rows - 1 - env // columns
            axis.add_patch(Rectangle((x, y), 0.9, 0.9, facecolor="#e6d3a3", edgecolor=COLOR["full"], linewidth=0.6))
            axis.add_patch(Rectangle((x, y + 0.36), 0.18, 0.18, facecolor=COLOR["stream_cache"], edgecolor="none"))
        axis.set_xlim(-0.1, 8.0)
        axis.set_ylim(-0.1, 8.0)
        axis.set_aspect("equal")
        axis.axis("off")
        axis.set_title(f"{count} env" + ("s" if count > 1 else ""), fontsize=10)
    fig.suptitle(
        "Terrain of the environment sweep, seen from above: every environment has its own 8 x 8 m of sand. "
        "Sand colour: simulated by full. Blue square: the 1.6 m streaming pool, which follows the box.",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def fmt(value, digits=1):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:,.{digits}f}"
    return f"{value:,}" if isinstance(value, int) else str(value)


def cost_table(rows: list[dict], xname: str, per_env: bool) -> tuple[list[str], list[list[str]]]:
    headers = ["terrain", "sleep", xname, "status", "particles", "peak VRAM [MiB]"]
    headers += ["VRAM per env [MiB]"] if per_env else []
    headers += ["step mean [ms]", "step p95 [ms]"]
    headers += ["step per env [ms]", "physics per env [ms]"] if per_env else ["physics [ms]"]
    headers += ["GPU util mean / max [%]", "timed steps", "asleep / support / awake"]
    table = []
    for r in rows:
        line = [TERRAIN_LABEL[r["terrain"]], r["sleep"], fmt(r["x"])]
        if r["status"] != "ok":
            table.append([*line, f"FAILED ({r['status']})"] + [""] * (len(headers) - 4))
            continue
        line += ["ok", fmt(r["particles"]), fmt(r["vram"], 0)]
        line += [fmt(r["vram_env"], 0)] if per_env else []
        line += [fmt(r["step"]), fmt(r["p95"])]
        line += [fmt(r["step_env"], 2), fmt(r["physics_env"], 2)] if per_env else [fmt(r["physics"])]
        util = f"{r['util']:.0f} / {r['util_max']:.0f}" if r["util"] is not None else ""
        activity = " / ".join(f"{v:,}" for v in r["activity"]) if r["activity"] else ""
        table.append([*line, util, fmt(r["steps"]), activity])
    return headers, table


def cache_table(rows: list[dict], xname: str) -> tuple[list[str], list[list[str]]]:
    headers = [
        xname,
        "sleep",
        "slots: peak / allocated",
        "records: peak / allocated",
        "tiles: peak / allocated",
        "slot multiplier",
    ]
    table = []
    for r in rows:
        if r["terrain"] != "stream_cache" or r["status"] != "ok" or not r.get("cache"):
            continue
        c, cfg = r["cache"], r["cache_cfg"]
        table.append([
            fmt(r["x"]), r["sleep"], f"{c['slots_peak']:,} / {c['slot_capacity']:,}", f"{c['records_peak']:,} / {c['record_capacity']:,}",
            f"{c['tiles_peak']:,} / {c['tile_capacity']:,}", fmt(cfg["simulation_capacity_multiplier"], 2),
        ])  # fmt: skip
    return headers, table


def failures(rows: list[dict], xname: str) -> list[str]:
    return [
        f"{label(r['terrain'], r['sleep'])} at {xname} = {fmt(r['x'])}: `{r['error'][:260]}`"
        for r in rows
        if r["status"] != "ok"
    ]


def largest(rows: list[dict]) -> list[list[str]]:
    table = []
    for terrain, sleep in MODES:
        ok = [r["x"] for r in rows if r["terrain"] == terrain and r["sleep"] == sleep and r["status"] == "ok"]
        bad = [r for r in rows if r["terrain"] == terrain and r["sleep"] == sleep and r["status"] != "ok"]
        table.append(
            [
                TERRAIN_LABEL[terrain],
                sleep,
                fmt(max(ok)) if ok else "none",
                f"{fmt(bad[0]['x'])} ({bad[0]['status']})" if bad else "not reached",
            ]
        )
    return table


# --------------------------------------------------------------------------------------
# Terrain history (P3)
# --------------------------------------------------------------------------------------
def track_mask(shape, lower, points, radius: float) -> np.ndarray:
    """Cells of a height map within ``radius`` of any path point."""
    x = lower[0] + (np.arange(shape[0]) + 0.5) * CELL
    y = lower[1] + (np.arange(shape[1]) + 0.5) * CELL
    xx, yy = np.meshgrid(x, y, indexing="ij")
    mask = np.zeros(shape, dtype=bool)
    for px, py in points[:: max(1, len(points) // 400)]:
        mask |= (xx - px) ** 2 + (yy - py) ** 2 <= radius**2
    return mask


def fill_unsimulated(grid: np.ndarray, level: float) -> np.ndarray:
    """A mode without a cache has no sand state outside its pool; that terrain comes back at the fresh level."""
    filled = grid.copy()
    filled[~np.isfinite(filled)] = level
    return filled


def history_box(logs: str, fig_dir: str, sleep: str) -> dict | None:
    runs = {}
    for terrain in TERRAINS:
        meta = load(os.path.join(logs, f"history_{terrain}_sleep_{sleep}.json"))
        if meta and meta["status"] == "ok":
            runs[terrain] = (meta, np.load(os.path.join(logs, f"history_{terrain}_sleep_{sleep}.npz")))
    if "full" not in runs:
        return None
    repeat = load(os.path.join(logs, "history_full_sleep_off_repeat.json")) if sleep == "off" else None
    if repeat and repeat["status"] == "ok":
        runs["repeat"] = (repeat, np.load(os.path.join(logs, "history_full_sleep_off_repeat.npz")))
    out = {"runs": runs, "sleep": sleep, "height": [], "force": [], "levels": {}}

    def relative(name: str, key: str) -> np.ndarray:
        meta, data = runs[name]
        level = meta["surface_z_after_presettle_m"]
        return fill_unsimulated(data[key], level) - level

    meta_full, data_full = runs["full"]
    marks = meta_full["phase_end_steps"]
    lower = data_full["terrain_lower"]
    origin = data_full["env_origin"][:2]
    path = data_full["target"][:, :2] + origin
    lap1 = slice(marks["settle_in"], marks["lap_1"])
    lap2 = slice(marks["lap_1"], marks["lap_2"])
    shape = data_full["height_terrain_lap_1"].shape
    track = track_mask(shape, lower, path[lap1], 0.2)
    for name, (meta, _) in runs.items():
        out["levels"][name] = meta["surface_z_after_presettle_m"]

    # height maps: depth relative to each mode's own settled surface
    order = [n for n in ("full", "stream_cache", "stream") if n in runs]
    for key, title in (("height_terrain_lap_1", "end of lap 1"), ("height_terrain_lap_2", "end of lap 2")):
        fig, axes = plt.subplots(2, len(order), figsize=(5.2 * len(order), 6.6), squeeze=False)
        extent = [lower[0], lower[0] + shape[0] * CELL, lower[1], lower[1] + shape[1] * CELL]
        reference = relative("full", key)
        for column, name in enumerate(order):
            grid = relative(name, key)
            image = axes[0, column].imshow(100 * grid.T, origin="lower", extent=extent, cmap="viridis", vmin=-8, vmax=4)
            axes[0, column].set_title(f"{TERRAIN_LABEL[name]}: height relative to its settled surface [cm]", fontsize=9)
            fig.colorbar(image, ax=axes[0, column], shrink=0.8)
            if name == "full":
                axes[1, column].imshow(track.T, origin="lower", extent=extent, cmap="Greys", vmin=0, vmax=2)
                axes[1, column].set_title("track band used for the statistics (within 0.2 m of the path)", fontsize=9)
            else:
                diff = 100 * (grid - reference)
                image = axes[1, column].imshow(diff.T, origin="lower", extent=extent, cmap="RdBu_r", vmin=-6, vmax=6)
                axes[1, column].set_title(f"{TERRAIN_LABEL[name]} minus full [cm]", fontsize=9)
                fig.colorbar(image, ax=axes[1, column], shrink=0.8)
                on_track = diff[track & np.isfinite(diff)]
                out["height"].append(
                    [
                        title,
                        TERRAIN_LABEL[name],
                        fmt(float(np.median(on_track)), 2),
                        fmt(float(np.sqrt(np.mean(on_track**2))), 2),
                        fmt(float(np.percentile(np.abs(on_track), 95)), 2),
                    ]
                )
        if "repeat" in runs:
            diff = 100 * (relative("repeat", key) - reference)
            on_track = diff[track & np.isfinite(diff)]
            out["height"].append(
                [
                    title,
                    "full, repeated run",
                    fmt(float(np.median(on_track)), 2),
                    fmt(float(np.sqrt(np.mean(on_track**2))), 2),
                    fmt(float(np.percentile(np.abs(on_track), 95)), 2),
                ]
            )
        for axis in axes.flat:
            axis.set_xlabel("x [m]")
            axis.set_ylabel("y [m]")
        fig.suptitle(f"Box on the lemniscate, sleep {sleep}: terrain height at the {title}", fontsize=11)
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, f"history_box_heights_sleep_{sleep}_{key[-5:]}.png"), dpi=95)
        plt.close(fig)

    # forces
    time = np.arange(len(data_full["force_world"])) * CONTROL_DT
    fig, axes = plt.subplots(4, 1, figsize=(15, 10.5), sharex=True)
    for name in order:
        _, data = runs[name]
        force = data["force_world"]
        velocity = np.gradient(data["target"][:, :2], CONTROL_DT, axis=0)
        speed = np.maximum(np.linalg.norm(velocity, axis=1), 1.0e-6)
        along = np.sum(force[:, :2] * velocity, axis=1) / speed
        series = (force[:, 0], force[:, 1], force[:, 2], along)
        for axis, values in zip(axes, series):
            kernel = np.ones(25) / 25.0
            axis.plot(
                time[: len(values)],
                np.convolve(values, kernel, mode="same"),
                color=COLOR[name],
                linewidth=1.1,
                label=TERRAIN_LABEL[name],
            )
    for axis, name in zip(axes, ("Fx [N]", "Fy [N]", "Fz [N]", "force along the motion [N]")):
        axis.axvspan(lap2.start * CONTROL_DT, lap2.stop * CONTROL_DT, color="#999999", alpha=0.15)
        axis.set_ylabel(name)
        axis.grid(True, alpha=0.25)
    axes[0].legend(fontsize=9, ncol=3, loc="upper left")
    axes[0].set_title(
        f"Box on the lemniscate, sleep {sleep}: force from the sand, 0.5 s moving average; lap 2 (revisited track) is shaded",
        fontsize=10,
    )
    axes[-1].set_xlabel("time [s]")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, f"history_box_forces_sleep_{sleep}.png"), dpi=95)
    plt.close(fig)

    reference = data_full["force_world"]
    for lap_name, lap in (("lap 1 (fresh, except at the crossing)", lap1), ("lap 2 (revisited)", lap2)):
        for name in [n for n in ("full", "stream_cache", "stream", "repeat") if n in runs]:
            _, data = runs[name]
            force = data["force_world"]
            velocity = np.gradient(data["target"][:, :2], CONTROL_DT, axis=0)
            speed = np.maximum(np.linalg.norm(velocity, axis=1), 1.0e-6)
            along = (np.sum(force[:, :2] * velocity, axis=1) / speed)[lap]
            row = [
                lap_name,
                "full, repeated run" if name == "repeat" else TERRAIN_LABEL[name],
                fmt(float(along.mean())),
                fmt(float(force[lap, 2].mean())),
            ]
            if name == "full":
                row += ["", "", ""]
            else:
                error = force[lap] - reference[lap]
                row += [fmt(float(np.sqrt(np.mean(error[:, i] ** 2)))) for i in range(3)]
            out["force"].append(row)
    return out


def history_g1(logs: str, fig_dir: str) -> dict | None:
    runs = {}
    for terrain in TERRAINS:
        meta = load(os.path.join(logs, f"g1_lemniscate_{terrain}.json"))
        if meta and meta["status"] == "ok":
            runs[terrain] = (meta, np.load(os.path.join(logs, f"g1_lemniscate_{terrain}.npz")))
    if "full" not in runs:
        return None
    out = {"runs": runs, "height": [], "summary": [], "levels": {}}

    _, data_full = runs["full"]
    lower = data_full["terrain_lower"]
    shape = data_full["height_terrain_end"].shape
    near_path = track_mask(shape, lower, data_full["path"], 0.3)
    x = lower[0] + (np.arange(shape[0]) + 0.5) * CELL
    y = lower[1] + (np.arange(shape[1]) + 0.5) * CELL
    xx, yy = np.meshgrid(x, y, indexing="ij")
    levels = {}

    def relative(name: str, key: str) -> np.ndarray:
        """Height relative to the mode's settled, untrodden surface.

        That surface is measured at the end of the run inside the simulated window around the robot, away from
        the path. The first frame cannot be used: the streaming modes have not settled their sand yet.
        """
        _, data = runs[name]
        if name not in levels:
            end, root = data["height_terrain_end"], data["root"][-1]
            window = (np.abs(xx - root[0]) < 0.55) & (np.abs(yy - root[1]) < 0.55)
            levels[name] = float(np.nanmedian(end[window & ~near_path & np.isfinite(end)]))
        return fill_unsimulated(data[key], levels[name]) - levels[name]

    order = [n for n in ("full", "stream_cache", "stream") if n in runs]
    extent = [lower[0], lower[0] + shape[0] * CELL, lower[1], lower[1] + shape[1] * CELL]
    track = near_path
    fig, axes = plt.subplots(2, len(order), figsize=(5.2 * len(order), 6.6), squeeze=False)
    reference = relative("full", "height_terrain_end")
    for column, name in enumerate(order):
        meta, data = runs[name]
        grid = relative(name, "height_terrain_end")
        image = axes[0, column].imshow(100 * grid.T, origin="lower", extent=extent, cmap="viridis", vmin=-10, vmax=5)
        axes[0, column].plot(data["root"][:, 0], data["root"][:, 1], color="white", linewidth=0.6, alpha=0.8)
        axes[0, column].set_title(
            f"{TERRAIN_LABEL[name]}: height relative to its settled surface [cm]; pelvis path in white", fontsize=8
        )
        fig.colorbar(image, ax=axes[0, column], shrink=0.8)
        if name == "full":
            axes[1, column].imshow(track.T, origin="lower", extent=extent, cmap="Greys", vmin=0, vmax=2)
            axes[1, column].set_title("track band used for the statistics (within 0.3 m of the path)", fontsize=9)
        else:
            diff = 100 * (grid - reference)
            image = axes[1, column].imshow(diff.T, origin="lower", extent=extent, cmap="RdBu_r", vmin=-8, vmax=8)
            axes[1, column].set_title(f"{TERRAIN_LABEL[name]} minus full [cm]", fontsize=9)
            fig.colorbar(image, ax=axes[1, column], shrink=0.8)
            on_track = diff[track & np.isfinite(diff)]
            out["height"].append(
                [TERRAIN_LABEL[name], fmt(float(np.median(on_track)), 2), fmt(float(np.sqrt(np.mean(on_track**2))), 2)]
            )
        own = 100 * grid[track & np.isfinite(grid)]
        out["summary"].append([
            TERRAIN_LABEL[name], fmt(meta["steps_run"]), "yes" if meta["completed"] else "no", fmt(meta["resets"]),
            fmt(100 * meta["path_deviation_m"]["mean"]), fmt(100 * meta["path_deviation_m"]["max"]),
            fmt(float(np.percentile(own, 5)), 2), fmt(float(np.mean(own < -2.0) * 100), 0),
        ])  # fmt: skip
    for axis in axes.flat:
        axis.set_xlabel("x [m]")
        axis.set_ylabel("y [m]")
    out["levels"] = levels
    fig.suptitle("G1 on the lemniscate: terrain height after two laps", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "history_g1_heights.png"), dpi=95)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------------------
# Fresh terrain (step 5, debug check; tables only)
# --------------------------------------------------------------------------------------
def fresh_tables(logs: str) -> dict:
    out = {"drag": [], "indent": []}
    runs = {}
    for terrain in TERRAINS:
        for kind in ("drag", "indent"):
            meta = load(os.path.join(logs, f"fresh_{terrain}_{kind}.json"))
            if meta and meta["status"] == "ok":
                runs[(terrain, kind)] = (meta, np.load(os.path.join(logs, f"fresh_{terrain}_{kind}.npz")))
    for terrain in TERRAINS:
        if (terrain, "drag") in runs:
            meta, data = runs[(terrain, "drag")]
            marks = meta["phase_end_steps"]
            begin = marks["settle_in"] + int(0.4 * (marks["out"] - marks["settle_in"]))
            force = data["force_world"][begin : marks["out"]]
            row = [
                TERRAIN_LABEL[terrain],
                fmt(100 * meta["surface_z_after_presettle_m"]),
                fmt(float(force[:, 0].mean())),
                fmt(float(force[:, 2].mean())),
            ]
            if terrain != "full" and ("full", "drag") in runs:
                reference = runs[("full", "drag")][1]["force_world"][begin : marks["out"]]
                row += [
                    fmt(float(np.sqrt(np.mean((force[:, 0] - reference[:, 0]) ** 2)))),
                    fmt(float(np.sqrt(np.mean((force[:, 2] - reference[:, 2]) ** 2)))),
                ]
            else:
                row += ["", ""]
            out["drag"].append(row)
        if (terrain, "indent") in runs:
            meta, data = runs[(terrain, "indent")]
            marks = meta["phase_end_steps"]
            force = data["force_world"]
            depth = meta["surface_z_after_presettle_m"] - (data["joint_pos"][:, 2] - 0.05)
            press = slice(marks["presettle"], marks["press"])
            row = [TERRAIN_LABEL[terrain], fmt(100 * meta["surface_z_after_presettle_m"])]
            for target in (0.04, 0.07, 0.10):
                index = int(np.argmin(np.abs(depth[press] - target)))
                row.append(fmt(float(force[press][index, 2])))
            row.append(fmt(float(force[marks["press"] : marks["hold"], 2].mean())))
            out["indent"].append(row)
    return out


# --------------------------------------------------------------------------------------
# HTML
# --------------------------------------------------------------------------------------
CSS = """
:root{--bg:#fff;--fg:#1c1e21;--mut:#5f6670;--line:#d9dde3;--card:#f5f6f8;--warn:#9a5b00;--warnbg:#fff4dd}
@media (prefers-color-scheme:dark){:root{--bg:#15171a;--fg:#e6e8eb;--mut:#9aa2ad;--line:#33383f;--card:#1e2126;--warn:#f0b860;--warnbg:#3a2c12}}
*{box-sizing:border-box}html{scroll-behavior:smooth}
body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
section{padding:32px max(16px,5vw) 40px;border-bottom:1px solid var(--line)}
h1{font-size:2rem;margin:0 0 8px}h2{font-size:1.5rem;margin:0 0 4px}h3{font-size:1.05rem;margin:24px 0 8px}
p,li{max-width:90ch}.kicker{color:var(--mut);font-size:.85rem;letter-spacing:.04em;text-transform:uppercase}
.badge{display:inline-block;padding:1px 8px;border-radius:999px;font-size:.75rem;font-weight:600;vertical-align:middle;
background:var(--warnbg);color:var(--warn);border:1px solid var(--warn);margin-left:8px}
.none{background:var(--card);color:var(--mut);border-color:var(--line)}
.chip{display:inline-block;width:.75em;height:.75em;border-radius:2px;margin-right:6px}
table{border-collapse:collapse;margin:8px 0;font-size:.85rem;font-variant-numeric:tabular-nums}
th,td{padding:3px 10px;border-bottom:1px solid var(--line);text-align:left}td.n,th.n{text-align:right}th{color:var(--mut);font-weight:600}
.wrap{overflow-x:auto}figure{margin:16px 0}img{max-width:100%;height:auto;background:#fff;border:1px solid var(--line)}
figcaption,.src{color:var(--mut);font-size:.8rem;max-width:110ch}
video{width:100%;background:#000;display:block}.vgrid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,360px),1fr));gap:12px}
.vcell{background:var(--card);padding:8px;border:1px solid var(--line)}.vcell .t{font-weight:600;margin-bottom:4px}
.missing{color:var(--mut);font-size:.95rem;padding:16px;background:var(--card);border-left:4px solid var(--line);max-width:90ch}
.take{background:var(--card);border:1px solid var(--line);padding:10px 16px;margin:20px 0 0;max-width:110ch}
.take-title{font-weight:700;font-size:.85rem;letter-spacing:.04em;text-transform:uppercase;color:var(--mut)}
.take ul{margin:6px 0 0;padding-left:20px}
.callout{background:var(--card);border-left:4px solid var(--warn);padding:8px 16px;margin:12px 0;max-width:110ch}
code{font-size:.85em}a{color:inherit}
"""


def esc(value) -> str:
    return html.escape(str(value))


def table(headers, rows, numeric_from=2) -> str:
    head = "".join(f'<th class="{"n" if i >= numeric_from else ""}">{esc(h)}</th>' for i, h in enumerate(headers))
    body = "".join(
        "<tr>"
        + "".join(f'<td class="{"n" if i >= numeric_from else ""}">{esc(c)}</td>' for i, c in enumerate(r))
        + "</tr>"
        for r in rows
    )
    return f'<div class="wrap"><table><tr>{head}</tr>{body}</table></div>'


def markdown_table(headers, rows) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def slide(title, kicker, body, badge="unchecked") -> str:
    b = f'<span class="badge {"none" if badge == "not done" else ""}">{esc(badge)}</span>' if badge else ""
    return f'<section><div class="kicker">{esc(kicker)}</div><h2>{esc(title)}{b}</h2>{body}</section>'


FIGURE_COUNT = [0]


def figure(rel, caption) -> str:
    FIGURE_COUNT[0] += 1
    label = f"<b>Figure {FIGURE_COUNT[0]}.</b> "
    return f'<figure><img src="{esc(rel)}" alt=""><figcaption>{label}{caption}</figcaption></figure>'


def takeaway(lines) -> str:
    items = "".join(f"<li>{line}</li>" for line in lines)
    return f'<div class="take"><div class="take-title">Takeaway</div><ul>{items}</ul></div>'


def pick(rows, terrain, sleep, x):
    """Row of one mode at one sweep value, or None if it did not run."""
    for row in rows:
        if row["terrain"] == terrain and row["sleep"] == sleep and row["x"] == x and row["status"] == "ok":
            return row
    return None


def largest_ok(rows, terrain, sleep):
    done = [row for row in rows if row["terrain"] == terrain and row["sleep"] == sleep and row["status"] == "ok"]
    return max(done, key=lambda row: row["x"]) if done else None


def missing(text) -> str:
    return f'<div class="missing">{text}</div>'


def callouts(lines) -> str:
    return "".join(
        f'<div class="callout">{esc(line).replace("`", "<code>", 1).replace("`", "</code>", 1)}</div>' for line in lines
    )


def encode_video(source: str, target: str) -> bool:
    """Write a smaller copy of a recording for the slides (960 px wide, original speed)."""
    command = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        source,
        "-vf",
        "scale=960:-2",
        "-c:v",
        "libx264",
        "-crf",
        "30",
        "-preset",
        "fast",
        "-an",
        target,
    ]
    return subprocess.run(command).returncode == 0 and os.path.exists(target)


def videos(logs: str, media: str, pattern: str, names: dict) -> str:
    cells = ""
    for terrain in ("full", "stream_cache", "stream"):
        found = sorted(glob.glob(os.path.join(logs, "video", pattern.format(terrain=terrain))))
        if not found:
            continue
        target = os.path.join(media, names[terrain])
        if not os.path.exists(target) or os.path.getmtime(target) < os.path.getmtime(found[-1]):
            encode_video(found[-1], target)
        cells += (
            f'<div class="vcell"><div class="t"><span class="chip" style="background:{COLOR[terrain]}"></span>{esc(TERRAIN_LABEL[terrain])}</div>'
            f'<video controls preload="metadata" src="media/{esc(names[terrain])}"></video>'
            f'<div class="src">Original: <code>{esc(os.path.abspath(found[-1]))}</code>, 1x speed.</div></div>'
        )
    return f'<div class="vgrid">{cells}</div>' if cells else ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs", required=True)
    parser.add_argument("--site", required=True)
    parser.add_argument("--tables", default=None, help="Also write the result tables as Markdown to this file.")
    args = parser.parse_args()
    logs = os.path.abspath(args.logs)
    fig_dir, media = os.path.join(args.site, "fig"), os.path.join(args.site, "media")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(media, exist_ok=True)
    provenance = load(os.path.join(logs, "provenance", "summary.json")) or {}
    checkouts = provenance.get("checkouts", {})
    commit = "; ".join(
        f"{name} {info['commit'][:9]} ({info['dirty_paths']} uncommitted paths)" for name, info in checkouts.items()
    )
    source = (
        f"Script <code>scripts/benchmarks/t1/site_slides.py</code>; raw data <code>{esc(logs)}</code>; code {esc(commit)}; "
        f"uncommitted changes archived in <code>{esc(logs)}/provenance/</code>; GPU {esc(provenance.get('gpu', '?'))}."
    )
    md: list[str] = []

    # ---- P1
    rows1 = cost_rows(collect(logs, "terrain", "x", LENGTHS))
    if rows1:
        cost_figure(rows1, "terrain length length_x [m]", False, os.path.join(fig_dir, "cost_vs_length.png"))
        headers, body = cost_table(rows1, "length_x [m]", False)
        cache_headers, cache_body = cache_table(rows1, "length_x [m]")
        full_counts = ", ".join(
            f"{fmt(r['x'])} m: {r['particles']:,}"
            for r in rows1
            if r["terrain"] == "full" and r["sleep"] == "off" and r["status"] == "ok"
        )
        terrain_figure(os.path.join(fig_dir, "terrain_size_sweep.png"))
        s1 = figure(
            "fig/terrain_size_sweep.png",
            f"How the terrain grows in this sweep. The corridor is as wide as the pool's total width (1.6 m), so full and the streaming modes cover the same strip. {source}",
        ) + figure(
            "fig/cost_vs_length.png",
            f"Six modes against corridor length (width 1.6 m, one environment). Solid: sleep off; dashed: sleep on. Vertical lines in the first panel mark a mode's first failed length. Particles simulated by full: {esc(full_counts)}. {source}",
        )
        s1 += (
            table(headers, body)
            + "<h3>Cache allocation against use (stream with cache)</h3>"
            + table(cache_headers, cache_body)
        )
        far = LENGTHS[-1]
        full, on = pick(rows1, "full", "off", far), pick(rows1, "full", "on", far)
        stream, cached = pick(rows1, "stream", "off", far), pick(rows1, "stream_cache", "off", far)
        if full and on and stream and cached:
            s1 += takeaway(
                [
                    f"Streaming MPM removes the memory bound of large terrain. At {far:g} m, full needs {full['vram'] / 1024:.1f} GB and {full['step']:.0f} ms per step; stream without cache needs {stream['vram'] / 1024:.1f} GB and {stream['step']:.0f} ms at every length, and stream with cache {cached['vram'] / 1024:.1f} GB and {cached['step']:.0f} ms (it grows only by the sand it saves).",
                    f"Particle sleeping cuts the compute time of full on large terrain ({on['step']:.0f} ms against {full['step']:.0f} ms at {far:g} m), but it uses more memory ({on['vram'] / 1024:.1f} GB against {full['vram'] / 1024:.1f} GB) for its bookkeeping of which sand is awake.",
                    "On the streaming modes sleeping adds nothing: the region kept awake is as large as the pool, so there is little sand left to put to sleep.",
                ]
            )
        md += [
            "### P1: scaling with terrain size",
            "",
            markdown_table(headers, body),
            "",
            "Cache allocation against use (stream with cache):",
            "",
            markdown_table(cache_headers, cache_body),
            "",
            "Failures:",
            *[f"- {line}" for line in failures(rows1, "length_x")],
            "",
        ]
    else:
        s1 = missing("Not run yet.")
    slide1 = slide(
        "P1. Does it scale with terrain size?",
        "Step 2 · cost against length_x",
        s1,
        "unchecked" if rows1 else "not done",
    )

    # ---- P2
    rows2 = cost_rows(collect(logs, "envs", "n", ENV_COUNTS))
    if rows2:
        cost_figure(rows2, "number of environments", True, os.path.join(fig_dir, "cost_vs_envs.png"))
        headers, body = cost_table(rows2, "envs", True)
        cache_headers, cache_body = cache_table(rows2, "envs")
        fit = largest(rows2)
        envs_figure(os.path.join(fig_dir, "envs_sweep.png"))
        s2 = figure(
            "fig/envs_sweep.png",
            f"How the terrain grows in this sweep. Each environment has its own 8 x 8 m of sand and its own box; the environment grids are schematic, not to scale. {source}",
        ) + figure(
            "fig/cost_vs_envs.png",
            f"Six modes against the number of environments, each on 8 x 8 m of terrain. Per-environment values divide by the environment count. Vertical lines in the first panel mark a mode's first failed count. {source}",
        )
        s2 += "<h3>Largest environment count that ran, and the next count</h3>" + table(
            ["terrain", "sleep", "largest count that ran", "first failure"], fit
        )
        s2 += (
            table(headers, body)
            + "<h3>Cache allocation against use (stream with cache)</h3>"
            + table(cache_headers, cache_body)
        )
        best = {mode: largest_ok(rows2, *mode) for mode in MODES}
        if all(best.values()):
            full, stream, cached = best[("full", "off")], best[("stream", "off")], best[("stream_cache", "off")]
            s2 += takeaway(
                [
                    f"Streaming MPM fits many more environments on one GPU: {stream['x']} without cache and {cached['x']} with cache, against {full['x']} for full (32 GB card, 8 x 8 m of terrain each).",
                    f"Per environment, full needs {full['vram_env'] / 1024:.1f} GB and {full['step_env']:.0f} ms per step; stream without cache {stream['vram_env']:.0f} MiB and {stream['step_env']:.1f} ms; stream with cache {cached['vram_env']:.0f} MiB and {cached['step_env']:.1f} ms.",
                    f"Particle sleeping speeds up full only, and its extra memory lowers the count that fits: full {full['x']} to {best[('full', 'on')]['x']}, stream with cache {cached['x']} to {best[('stream_cache', 'on')]['x']}.",
                ]
            )
        md += [
            "### P2: scaling with the number of environments",
            "",
            markdown_table(["terrain", "sleep", "largest count that ran", "first failure"], fit),
            "",
            markdown_table(headers, body),
            "",
            "Cache allocation against use (stream with cache):",
            "",
            markdown_table(cache_headers, cache_body),
            "",
            "Failures:",
            *[f"- {line}" for line in failures(rows2, "envs")],
            "",
        ]
    else:
        s2 = missing("Not run yet.")
    slide2 = slide(
        "P2. Does it scale with the number of environments?",
        "Step 3 · cost against num_envs at the training terrain size",
        s2,
        "unchecked" if rows2 else "not done",
    )

    # ---- P3 box
    s3 = ""
    for sleep in SLEEPS:
        result = history_box(logs, fig_dir, sleep)
        if result is None:
            continue
        height_headers = [
            "when",
            "mode against full",
            "median difference [cm]",
            "rms difference [cm]",
            "95th percentile of |difference| [cm]",
        ]
        force_headers = [
            "lap",
            "mode",
            "mean force along the motion [N]",
            "mean Fz [N]",
            "rms(Fx - full) [N]",
            "rms(Fy - full) [N]",
            "rms(Fz - full) [N]",
        ]
        levels = ", ".join(
            f"{'full (repeat)' if n == 'repeat' else TERRAIN_LABEL[n]} {100 * z:.1f} cm"
            for n, z in result["levels"].items()
        )
        # The slide shows the sleep-off runs only; the sleep-on repeat of the same analysis stays in the tables.
        if sleep == "off":
            s3 += figure(
                f"fig/history_box_heights_sleep_{sleep}_lap_1.png",
                f"Terrain after lap 1: this is what the box meets on lap 2. Heights are relative to each mode's own settled surface ({esc(levels)} below the nominal surface). Without a cache the terrain outside the pool has no state; it is drawn at the fresh level, which is what the pool restores there. {source}",
            )
            s3 += figure(f"fig/history_box_heights_sleep_{sleep}_lap_2.png", f"Terrain after lap 2. {source}")
            s3 += "<h3>Height on the track: each streaming mode minus full, after lap 1 and after lap 2</h3>" + table(
                height_headers, result["height"]
            )
            s3 += figure(
                f"fig/history_box_forces_sleep_{sleep}.png",
                f"Force on the box (world frame) and its component along the direction of motion. {source}",
            )
            s3 += "<h3>Force on the box, lap 1 and lap 2</h3>" + table(force_headers, result["force"])
        md += [
            f"### P3: terrain history, box, sleep {sleep}",
            "",
            f"Settled surface below the nominal surface: {levels}.",
            "",
            markdown_table(height_headers, result["height"]),
            "",
            markdown_table(force_headers, result["force"]),
            "",
        ]
    clips = videos(
        logs, media, "probe_{terrain}_sleep_off_lemniscate_*.mp4", {t: f"box_lemniscate_{t}.mp4" for t in TERRAINS}
    )
    if clips:
        s3 += (
            "<h3>Videos</h3>"
            + clips
            + '<p class="src">RTX visualizer, ground and walls hidden. Each mode draws the sand it holds as particles: full its whole bed; stream without cache its pool (simulated window and kinematic ring); stream with cache its pool plus the sand saved in the cache.</p>'
        )
    if s3:
        s3 += takeaway(
            [
                "With the cache, terrain history is preserved: on the second lap the track height matches full to within the difference between two identical runs of full, and the drop in force on the revisited track is reproduced to within 10% (sleep off). That 10% is a real residual: two identical runs of full differ by about 1%.",
                "Without the cache, history is lost: the second lap meets fresh sand, so its force is the same as on the first lap.",
                "The comparison needs the same starting sand in every mode. The cache's settled tile template makes fresh sand denser and stiffer than a full bed, so these runs skip it.",
                "With sleep on the heights still agree, but the forces do not: on fresh sand the cache is 31% below full.",
            ]
        )
    slide3 = slide(
        "P3. Does it preserve terrain history? Box on a lemniscate",
        "Step 4 · two laps, lobes larger than the patch",
        s3 or missing("Not run yet."),
        "unchecked" if s3 else "not done",
    )

    # ---- P3 G1
    result = history_g1(logs, fig_dir)
    if result:
        summary_headers = [
            "mode",
            "control steps",
            "two laps completed",
            "resets",
            "path deviation mean [cm]",
            "path deviation max [cm]",
            "5th percentile height on the track [cm]",
            "track cells deeper than 2 cm [%]",
        ]
        height_headers = ["mode against full", "median difference [cm]", "rms difference [cm]"]
        s4 = figure(
            "fig/history_g1_heights.png",
            f"Heights are relative to each mode's own settled surface, measured at the end of the run inside the simulated window around the robot, away from the path. The robot's motion differs between modes, so the maps are compared, not the forces. {source}",
        )
        s4 += table(summary_headers, result["summary"], 1) + table(height_headers, result["height"], 1)
        md += [
            "### P3: terrain history, G1",
            "",
            markdown_table(summary_headers, result["summary"]),
            "",
            markdown_table(height_headers, result["height"]),
            "",
        ]
        clips = videos(
            logs, media, "g1_{terrain}_sleep_off_lemniscate_*.mp4", {t: f"g1_lemniscate_{t}.mp4" for t in TERRAINS}
        )
        s4 += ("<h3>Videos</h3>" + clips) if clips else ""
        s4 += takeaway(
            [
                "The robot's footprints are kept with the cache and lost without it, and the robot completes both laps in every mode.",
            ]
        )
    else:
        s4 = missing("Not run yet.")
    slide4 = slide(
        "P3. Does it preserve terrain history? G1 on the same lemniscate",
        "Step 4 · height maps and video only",
        s4,
        "unchecked" if result else "not done",
    )

    # ---- step 5 (tables for the output only)
    fresh = fresh_tables(logs)
    if fresh["drag"] or fresh["indent"]:
        md += [
            "### Step 5: fresh terrain (debug check)",
            "",
            "Straight drag, 4.8 m at 0.5 m/s, underside 6 cm deep; means over the last 60% of the pass:",
            "",
            markdown_table(
                [
                    "mode",
                    "settled surface [cm]",
                    "mean Fx [N]",
                    "mean Fz [N]",
                    "rms(Fx - full) [N]",
                    "rms(Fz - full) [N]",
                ],
                fresh["drag"],
            ),
            "",
            "Indentation at 0.05 m/s to 10 cm:",
            "",
            markdown_table(
                [
                    "mode",
                    "settled surface [cm]",
                    "Fz at 4 cm [N]",
                    "Fz at 7 cm [N]",
                    "Fz at 10 cm [N]",
                    "Fz during the hold [N]",
                ],
                fresh["indent"],
            ),
            "",
        ]

    # ---- last slide and title
    rules = (
        "<h3>Measurement rules</h3><ol>"
        "<li>Step time includes the action, the MDP terms, all four physics substeps, tile paging and device synchronization; it excludes rendering and the first 25 control steps (which include the hover that precedes contact).</li>"
        "<li>GPU utilization is <code>utilization.gpu</code> from <code>nvidia-smi</code>, sampled every 0.25 s (about 0.3 s in practice) over the timed window of the same run.</li>"
        "<li>Peak VRAM is the process peak from <code>nvidia-smi</code>, so it includes the CUDA context.</li>"
        "<li>Physics time comes from a second, shorter pass (at most 400 control steps) in which the physics step is timed separately.</li>"
        "<li>All modes share the sand layer (depth 0.25 m, voxel 0.03 m, 1.25 particles per cell edge, material <code>poppy_seed_loose</code>), the timestep (1/200 s, 4 substeps per control step), the box (0.2 x 0.2 x 0.1 m, 10 kg, PD gantry) and the trajectory. Full is bounded by a rigid floor and walls and starts from the sampled particles; the streaming modes are bounded by a ring of kinematic particles. The cache starts its tiles from a template settled for 2.5 s, except in the P3 box runs, where the tiles start from the sampled particles like the other two modes. Every run hovers for 2 s before contact.</li>"
        "<li>One run per configuration.</li></ol>"
    )
    values = (
        "<h3>Open values as used</h3><ul>"
        "<li>Sleep on: strain activation inside a 1.2 x 1.2 x 3.0 m region tracked around the box, in all three terrain modes.</li>"
        "<li>Tracked body: <code>box</code> for the pool and the region; <code>pelvis</code> for G1.</li>"
        "<li>length_x: 3.2 to 102.4 m, doubling; width 1.6 m; the box travels length_x - 1.6 m at 1.0 m/s, underside 6 cm deep.</li>"
        "<li>Training terrain: 8 x 8 m per environment; the box travels 6.4 m.</li>"
        "<li>Lemniscate: x = a sin t, y = (a/2) sin 2t, a = 2.4 m, two laps; box at 0.5 m/s, G1 at 0.3 m/s nominal.</li>"
        "<li>Cache: sized per run from the planned path (tiles the pool loads plus a one-tile border; records for the sand outside the pool plus the strips saved in one update); 1.10 simulation slots per pool particle.</li>"
        "<li>Sparse-grid reserve (P1 and P2): sized per run to the bounding box of the sand a mode simulates (the bed for full, the pool for streaming), one leaf of headroom in z, times 1.25.</li></ul>"
    )
    notes = (
        "<ul>"
        "<li>One run per configuration. The only estimate of run-to-run variation is one repeat of full on the lemniscate.</li>"
        "<li>Nothing is committed: the code state is archived as patches, and the probe script changed during the runs.</li>"
        "<li>The GPU also drives the desktop (about 1.7 GB in use, 1 to 8% utilization when idle).</li>"
        "<li>Environment counts were doubled, so the largest count that fits is only bracketed between the last success and the first failure.</li>"
        "<li>The physics time comes from a shorter pass (first 400 steps), so with sleep on it includes more of the falling-asleep transient than the step time does.</li>"
        "<li>With sleep off, the cache is within about 10% of full in mean force on both laps (58.7 against 63.9 N on lap 1, 29.4 against 26.7 N on lap 2); two identical runs of full differ by 0.2% and 1.4%, so the residual is real. With sleep on the forces do not agree (lap 1: 50.7 against 74.0 N).</li>"
        "<li>Without a cache, terrain outside the pool has no state; the maps draw it at the fresh level.</li>"
        "<li>Full runs without the task's pre-settled bed snapshot (it went non-finite on the 12.8 m bed).</li>"
        "<li>P3: the cache runs skip the cache's template settling, so all three modes start from the same sampling (surfaces after the hover: -4.3, -4.1 and -3.8 cm). With the task's template (settled for 2.5 s in a separate small simulation) the fresh sand is about 1 cm denser and the lap 1 force is 92 N against 64 N for full; those runs are kept in <code>history_settled_template/</code>. Settling full for longer (up to 12 s) does not close that gap. The same applies to the G1 run, the fresh-terrain check and the videos. Only the P1 and P2 runs use the task's settled template; their cost does not depend on it.</li>"
        "<li>With sleep on, the streaming modes gain no speed: the region kept awake (1.2 x 1.2 m) is as large as the pool's simulated window, so little or nothing sleeps, while sleep adds 8 to 19% VRAM (17 to 19% at 32 and 64 environments).</li>"
        "<li>The earlier P1 and P2 runs used the task's default sparse-grid reserve (8,192 leaf nodes per environment, about 290 MiB each). With it the streaming modes stopped at 32 environments; those runs are kept in <code>default_grid_reserve/</code>.</li>"
        "</ul>"
    )
    slide_caveats = slide("Not done and caveats", "What weakens the results", notes, "")
    slide5 = slide(
        "Measurement rules and open values",
        "Conditions for every result",
        rules + values + f'<p class="src">{source}</p>',
        "",
    )
    title = (
        '<section><div class="kicker">RSS MPM project · task T1</div><h1>Streaming MPM benchmark: scaling and terrain history (Q1)</h1>'
        "<p>A rigid box on a PD-driven gantry is moved through MPM sand in six modes: sleep off or on, times three ways of managing the terrain.</p><ul>"
        f'<li><span class="chip" style="background:{COLOR["full"]}"></span>full: every particle of the terrain is simulated.</li>'
        f'<li><span class="chip" style="background:{COLOR["stream"]}"></span>stream without cache: a 1.2 x 1.2 m pool (1.6 m with its boundary) follows the box; terrain that re-enters is fresh.</li>'
        f'<li><span class="chip" style="background:{COLOR["stream_cache"]}"></span>stream with cache: the same pool, and sand that leaves it is saved and restored on return.</li></ul>'
        f'<p class="src">{source} Every result is unchecked.</p></section>'
    )
    page = f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>T1 slides</title><style>{CSS}</style></head><body>{title}{slide1}{slide2}{slide3}{slide4}{slide_caveats}{slide5}</body></html>'
    with open(os.path.join(args.site, "index.html"), "w") as handle:
        handle.write(page)
    if args.tables:
        with open(args.tables, "w") as handle:
            handle.write("\n".join(md) + "\n")
    print("slides written to", os.path.join(args.site, "index.html"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
