# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run the T1 box-probe sweeps, one ``bench_probe.py`` process per configuration.

Run from the repository root:

    python3 scripts/benchmarks/t1/run_sweeps.py <sweep> [<sweep> ...] --out logs/t1/<machine>

Sweeps:

* ``terrain`` -- corridor length ``length_x``, one environment, six modes (step 2).
* ``envs``    -- number of environments on 8 x 8 m of terrain each, six modes (step 3).
* ``history`` -- two laps of the lemniscate with whole-terrain height maps (step 4, box).
* ``fresh``   -- an indentation and a straight drag on fresh terrain (step 5).
* ``video``   -- the lemniscate recorded with the RTX visualizer (step 4, box).

The cost sweeps run each configuration twice: a plain pass (step time, GPU utilization, peak VRAM) and a short
pass with the physics step timed separately. A mode stops growing its sweep after its first failure, which is
recorded in its JSON. Finished JSON files are skipped, so the script can be restarted.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
MODES = [(terrain, sleep) for terrain in ("full", "stream", "stream_cache") for sleep in ("off", "on")]
TERRAINS = ("full", "stream", "stream_cache")
LENGTHS = (3.2, 6.4, 12.8, 25.6, 51.2, 102.4)
ENV_COUNTS = (1, 2, 4, 8, 16, 32, 64, 128, 256)
CACHE_SLOTS = 1.10
"""Simulation slots per particle of one pool. The calibration runs peaked at 1.04 (59,109 of 57,024)."""


def run(out: str, tag: str, options: list[str], timeout_s: float = 4.0 * 3600.0) -> bool:
    """Run one configuration unless its result exists; return whether it succeeded."""
    path = os.path.join(out, f"{tag}.json")
    if not os.path.exists(path):
        if "stream_cache" in options:
            options = [*options, "--cache_slots", str(CACHE_SLOTS)]
        command = ["uv", "run", "--no-sync", "python", os.path.join(HERE, "bench_probe.py"), *options, "--out", path]
        print(f"run  {tag}", flush=True)
        begin = time.time()
        with open(os.path.join(out, f"{tag}.log"), "w") as log:
            try:
                subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=timeout_s)
            except subprocess.TimeoutExpired:
                with open(path, "w") as handle:
                    json.dump({"status": "error", "error": f"timeout after {timeout_s:.0f} s", "argv": options}, handle)
        print(f"     {time.time() - begin:.0f} s", flush=True)
    else:
        print(f"skip {tag} (done)", flush=True)
    try:
        with open(path) as handle:
            result = json.load(handle)
    except (OSError, ValueError):
        result = {"status": "missing"}
    print(f"     -> {result['status']} {result.get('error', '')[:160]}", flush=True)
    return result["status"] == "ok"


def mode_options(terrain: str, sleep: str) -> list[str]:
    return ["--terrain", terrain, "--sleep", sleep]


def cost_pair(out: str, tag: str, options: list[str]) -> bool:
    """Plain pass, then a short pass that times the physics step; false if the plain pass failed."""
    if not run(out, f"{tag}_plain", options):
        return False
    run(out, f"{tag}_profile", [*options, "--profile_physics", "--max_steps", "400"])
    return True


def sweep_terrain(out: str) -> None:
    for terrain, sleep in MODES:
        for length in LENGTHS:
            options = [*mode_options(terrain, sleep), "--trajectory", "drag", "--length_x", str(length)]
            if not cost_pair(out, f"terrain_{terrain}_sleep_{sleep}_x{length}", options):
                break


def sweep_envs(out: str) -> None:
    for terrain, sleep in MODES:
        for count in ENV_COUNTS:
            options = [
                *mode_options(terrain, sleep),
                *("--trajectory", "drag", "--length_x", "8.0", "--length_y", "8.0", "--num_envs", str(count)),
            ]
            if not cost_pair(out, f"envs_{terrain}_sleep_{sleep}_n{count}", options):
                break


LEMNISCATE = ["--trajectory", "lemniscate", "--lemniscate_a", "2.4", "--speed", "0.5", "--laps", "2"]


def sweep_history(out: str) -> None:
    for sleep in ("off", "on"):
        for terrain in TERRAINS:
            run(out, f"history_{terrain}_sleep_{sleep}", [*mode_options(terrain, sleep), *LEMNISCATE, "--terrain_map"])
    # One-time check of how much two identical runs differ.
    run(out, "history_full_sleep_off_repeat", [*mode_options("full", "off"), *LEMNISCATE, "--terrain_map"])


def sweep_fresh(out: str) -> None:
    for terrain in TERRAINS:
        mode = mode_options(terrain, "off")
        run(out, f"fresh_{terrain}_indent", [*mode, "--trajectory", "indent", "--speed", "0.05"])
        run(
            out,
            f"fresh_{terrain}_drag",
            [*mode, "--trajectory", "drag", "--length_x", "6.4", "--speed", "0.5", "--terrain_map"],
        )


def sweep_video(out: str) -> None:
    for terrain in TERRAINS:
        run(
            out,
            f"video_{terrain}_sleep_off",
            [*mode_options(terrain, "off"), *LEMNISCATE, "--video_dir", os.path.join(out, "video")],
        )


def save_provenance(out: str) -> None:
    """Record the commit and the uncommitted changes of every checkout the runs depend on."""
    import newton
    import warp

    folder = os.path.abspath(os.path.join(out, "provenance"))
    os.makedirs(folder, exist_ok=True)
    checkouts = {"isaaclab": ROOT, "newton": os.path.dirname(newton.__file__), "warp": os.path.dirname(warp.__file__)}
    summary = {}
    for name, path in checkouts.items():

        def git(*arguments, cwd=path):
            return subprocess.run(["git", *arguments], cwd=cwd, capture_output=True, text=True).stdout

        top = git("rev-parse", "--show-toplevel").strip()
        status = git("status", "--porcelain", cwd=top)
        summary[name] = {
            "path": top,
            "commit": git("rev-parse", "HEAD").strip(),
            "dirty_paths": len(status.splitlines()),
        }
        with open(os.path.join(folder, f"{name}_status.txt"), "w") as handle:
            handle.write(status)
        with open(os.path.join(folder, f"{name}_tracked_changes.patch"), "w") as handle:
            handle.write(git("diff", "HEAD", cwd=top))
        untracked = git("ls-files", "--others", "--exclude-standard", cwd=top).split()
        if untracked:
            subprocess.run(["tar", "czf", os.path.join(folder, f"{name}_untracked.tar.gz"), *untracked], cwd=top)
    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    with open(os.path.join(folder, "summary.json"), "w") as handle:
        json.dump({"checkouts": summary, "gpu": gpu, "time": time.strftime("%Y-%m-%d %H:%M:%S")}, handle, indent=2)


SWEEPS = {
    "terrain": sweep_terrain,
    "envs": sweep_envs,
    "history": sweep_history,
    "fresh": sweep_fresh,
    "video": sweep_video,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sweeps", nargs="+", choices=sorted(SWEEPS))
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if not os.path.exists(os.path.join(args.out, "provenance", "summary.json")):
        save_provenance(args.out)
    for name in args.sweeps:
        print(f"=== sweep {name}", flush=True)
        SWEEPS[name](args.out)
    print("finished", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
