# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Write the manifest of the planned dataset: every episode with its split, path and sampled description."""

import argparse
import json
from collections import Counter
from pathlib import Path

from .episodes import (
    CATEGORY_MOTIONS,
    PER_MOTION,
    PRODUCTION_ROBOTS,
    PRODUCTION_TERRAINS,
    episodes,
    robot_episode,
)

KEYS = ("id", "split", "terrain", "terrain_variant", "category", "motion", "motion_type", "shape", "size_level", "k")


def planned(per_motion: int = PER_MOTION):
    """Yield (relative path, record) for every planned episode."""
    for terrain in PRODUCTION_TERRAINS:
        for category in CATEGORY_MOTIONS:
            for record in episodes(terrain, category, per_motion):
                yield f"{record['split']}/{terrain}/{category}/{record['index']:04d}.hdf5", record
        for robot in PRODUCTION_ROBOTS:
            for index in range(per_motion):
                record = robot_episode(robot, index, terrain)
                yield f"{record['split']}/{terrain}/robot/{robot}/{index:04d}.hdf5", record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, required=True, help="Directory that receives manifest.jsonl and splits.json"
    )
    parser.add_argument("--per-motion", type=int, default=PER_MOTION)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    counts: Counter = Counter()
    with open(args.out / "manifest.jsonl", "w") as handle:
        for path, record in planned(args.per_motion):
            handle.write(json.dumps({"path": path, **{k: record.get(k) for k in KEYS}}) + "\n")
            counts[(record["split"], record["terrain"])] += 1
            counts[(record["split"], "all")] += 1
    summary = {
        split: {t: counts[(split, t)] for t in (*PRODUCTION_TERRAINS, "all")}
        for split in ("train", "validation", "test")
    }
    (args.out / "splits.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
