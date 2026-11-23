# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run the collection of one terrain on one node: enumerate the work, run one collector per GPU, summarize the files.

The work is split into units, each one collector process: an object group (shape, size level, bed variant), a chunk of
bare-bed episodes, one free-fall episode, or a robot (per bed variant). Object and sand-only episodes use the same
collector and scene. Collectors skip finished files, so a retry or a resubmission resumes.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Queue

from .episodes import (
    OBJECT_CATEGORIES,
    PER_MOTION,
    PRODUCTION_ROBOTS,
    PRODUCTION_TERRAINS,
    SAND_CATEGORIES,
    episodes,
    robot_episode,
)

SELECTIONS = {
    "lateral": ("lateral_push",),
    "vertical_impact": ("vertical_push", "impact"),
    "rest_sand": ("rest", "freefall", "rest_bed"),
    "sand": ("freefall", "rest_bed"),
    "robots": ("franka", "g1"),
    "franka": ("franka",),
    "g1": ("g1",),
}
"""Task groups: each one writes roughly 30-50 GB for a full terrain."""
SAND_CHUNK = 15
COST = {  # rough seconds per episode, used only to start the long units first
    "lateral_push": 45,
    "vertical_push": 45,
    "impact": 45,
    "rest": 70,
    "freefall": 45,
    "rest_bed": 70,
    "franka": 60,
    "g1": 55,
}


def build_units(terrain: str, select: list[str], per_motion: int, g1_checkpoint: str | None) -> list[dict]:
    """Enumerate the collector invocations of one terrain."""
    units = []
    for name in select:
        for item in SELECTIONS[name]:
            if item in OBJECT_CATEGORIES:
                records = episodes(terrain, item, per_motion)
                groups = sorted({(r["shape"], r["size_level"], r["terrain_variant"]) for r in records})
                for shape, level, variant in groups:
                    count = sum(
                        1
                        for r in records
                        if (r["shape"], r["size_level"], r["terrain_variant"]) == (shape, level, variant)
                    )
                    units.append(
                        {
                            "label": f"{terrain}-{item}-{shape}-s{level}-{variant}",
                            "module": "collect_objects",
                            "args": [
                                *("--terrain", terrain, "--category", item, "--shape", shape),
                                *("--size-level", str(level), "--variant", variant, "--per-motion", str(per_motion)),
                            ],
                            "cost": 60 + count * COST[item],
                        }
                    )
            elif item in SAND_CATEGORIES:
                # Same collector and scene as the object episodes. A free-fall process holds one block (its particle
                # count is fixed when the scene is built); bare beds are chunked per bed variant.
                records = episodes(terrain, item, per_motion)
                chunks = []
                if item == "freefall":
                    chunks = [(r["terrain_variant"], [r["index"]]) for r in records]
                else:
                    for variant in sorted({r["terrain_variant"] for r in records}):
                        indices = [r["index"] for r in records if r["terrain_variant"] == variant]
                        chunks += [
                            (variant, indices[start : start + SAND_CHUNK])
                            for start in range(0, len(indices), SAND_CHUNK)
                        ]
                for variant, chunk in chunks:
                    units.append(
                        {
                            "label": f"{terrain}-{item}-{variant}-{chunk[0]:04d}",
                            "module": "collect_objects",
                            "args": [
                                *("--terrain", terrain, "--category", item, "--variant", variant),
                                *("--per-motion", str(per_motion), "--indices", *map(str, chunk)),
                            ],
                            "cost": 60 + len(chunk) * COST[item],
                        }
                    )
            elif item in PRODUCTION_ROBOTS:
                variants = sorted({robot_episode(item, i, terrain)["terrain_variant"] for i in range(per_motion)})
                for variant in variants:
                    count = sum(
                        1 for i in range(per_motion) if robot_episode(item, i, terrain)["terrain_variant"] == variant
                    )
                    extra = ["--checkpoint", g1_checkpoint] if item == "g1" and g1_checkpoint else []
                    units.append(
                        {
                            "label": f"{terrain}-robot-{item}-{variant}",
                            "module": "collect_robot",
                            "args": [
                                *("--robot", item, "--terrain", terrain, "--variant", variant),
                                *("--per-motion", str(per_motion), *extra),
                            ],
                            "cost": 90 + count * COST[item],
                        }
                    )
            else:
                raise ValueError(f"Unknown item {item!r}")
    return sorted(units, key=lambda u: -u["cost"])


def run_unit(unit: dict, gpu: str, output: Path, log_dir: Path, attempts: int) -> dict:
    log_path = log_dir / f"{unit['label']}.log"
    command = [
        sys.executable,
        "-m",
        f"particle_nerd_benchmark.dataset.{unit['module']}",
        *unit["args"],
        "--output",
        str(output),
    ]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu, "PYTHONUNBUFFERED": "1"}
    start = time.perf_counter()
    status = "failed"
    for attempt in range(1, attempts + 1):
        with open(log_path, "a") as log:
            log.write(f"=== attempt {attempt}: {' '.join(command)}\n")
            log.flush()
            code = subprocess.run(
                command, env=env, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL
            ).returncode
        if code == 0:
            status = "ok"
            break
        print(f"[{unit['label']}] attempt {attempt} exited with {code}", flush=True)
        tail = log_path.read_text().splitlines()[-25:]
        print("\n".join(f"[{unit['label']}]   | {line[:200]}" for line in tail), flush=True)
    print(f"[{unit['label']}] {status} in {time.perf_counter() - start:.0f} s on GPU {gpu}", flush=True)
    return {"label": unit["label"], "status": status, "seconds": round(time.perf_counter() - start, 1)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--terrain", choices=PRODUCTION_TERRAINS, required=True)
    parser.add_argument("--select", nargs="+", choices=tuple(SELECTIONS), default=list(SELECTIONS))
    parser.add_argument("--per-motion", type=int, default=PER_MOTION, help="Episodes per motion (test run: 1)")
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--attempts", type=int, default=2)
    parser.add_argument("--g1-checkpoint", help="G1 policy checkpoint; training_config.json must sit beside it")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--list", action="store_true", help="Print the units and exit")
    parser.add_argument("--only", nargs="+", help="Run only the units with these labels (to redo failed ones)")
    args = parser.parse_args()

    units = build_units(args.terrain, args.select, args.per_motion, args.g1_checkpoint)
    if args.only:
        units = [u for u in units if u["label"] in set(args.only)]
    if args.list:
        for unit in units:
            print(f"{unit['cost']:6d}  {unit['label']}")
        print(f"{len(units)} units")
        return
    log_dir = args.output / "logs" / args.terrain
    log_dir.mkdir(parents=True, exist_ok=True)
    gpus: Queue = Queue()
    for index in range(args.gpus):
        gpus.put(str(index))

    def work(unit):
        gpu = gpus.get()
        try:
            return run_unit(unit, gpu, args.output, log_dir, args.attempts)
        finally:
            gpus.put(gpu)

    print(f"{len(units)} units on {args.gpus} GPU(s), terrain {args.terrain}, selection {args.select}", flush=True)
    with ThreadPoolExecutor(max_workers=args.gpus) as pool:
        results = list(pool.map(work, units))
    failed = [r for r in results if r["status"] != "ok"]
    name = f"{args.terrain}-{'+'.join(args.select)}"
    (args.output / "logs" / f"{name}.units.json").write_text(json.dumps(results, indent=2))
    print(f"{len(results) - len(failed)} units ok, {len(failed)} failed: {[r['label'] for r in failed]}", flush=True)

    from .summarize import summarize_tree

    summarize_tree(args.output, args.output / "logs" / f"{name}.files.jsonl", args.terrain)
    # Always exit 0 so that the task outputs are uploaded; failures are listed above and in the units file.


if __name__ == "__main__":
    main()
