# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Collect disjoint dynamic-body episodes for a small, reproducible learning study."""

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--simulation-python", type=Path, required=True)
    parser.add_argument(
        "--expanded", action="store_true", help="Additional training combinations with independently varied mass."
    )
    parser.add_argument("--deadline", help="Optional timezone-aware ISO stop time.")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    deadline = datetime.fromisoformat(args.deadline) if args.deadline else None
    if deadline is not None and deadline.tzinfo is None:
        parser.error("Deadline must include a UTC offset.")
    # No exact material/mass combination crosses a dataset split. Geometric
    # families occur in all splits; this does not test unseen shape families.
    settings = {
        "train": [(800, 0.25, 1e5, 1), (1600, 0.48, 1e7, 2), (2400, 0.7, 1e15, 4), (1600, 0.7, 1e5, 1)],
        "validation": [(1200, 0.4, 1e6, 1.5)],
        "test": [(2000, 0.6, 1e8, 3), (1600, 0.48, 1e15, 2)],
    }
    if args.expanded:
        settings = {
            "train": [
                (density, friction, modulus, mass)
                for density, friction, modulus in ((800, 0.25, 1e5), (1600, 0.48, 1e7), (2400, 0.7, 1e15))
                for mass in (1, 2, 4)
            ],
            "validation": [],
            "test": [],
        }
    cases = []
    # Validation is collected early so training can start while remaining data
    # are generated. Test data are never read by the optimizer or model selector.
    for split in ("train", "validation", "test"):
        for j, (density, friction, modulus, mass) in enumerate(settings[split]):
            for i, shape in enumerate(("box", "sphere", "cylinder")):
                index = len(cases)
                cases.append(
                    dict(
                        name=f"{split}_{'extra_' if args.expanded else ''}{j}_{shape}",
                        split=split,
                        shape=shape,
                        density=density,
                        friction=friction,
                        young_modulus=modulus,
                        mass=mass,
                        seed=(204000 if args.expanded else 104000) + index * 31,
                        linear=[0.12 * (-1) ** j, 0.04 * (-1) ** i, -0.1],
                        angular=[[0, 0, 0], [3, 0, 2], [0, -3, 2]][j % 3],
                    )
                )
    # Generate one train group and validation, then finish training/test groups.
    cases.sort(
        key=lambda c: 0
        if c["name"].startswith("train_0")
        else 1
        if c["split"] == "validation"
        else 2
        if c["split"] == "train"
        else 3
    )
    report = {"deadline": args.deadline, "cases": cases, "completed": [], "failed": []}
    path = args.output_dir / ("collection_expanded.json" if args.expanded else "collection.json")
    for case in cases:
        if deadline is not None and datetime.now(timezone.utc) >= deadline:
            break
        output = args.output_dir / (case["name"] + ".hdf5")
        if output.exists():
            raise FileExistsError(f"Refusing ambiguous collection restart: {output}")
        command = [
            str(args.simulation_python),
            str(Path(__file__).with_name("collect.py")),
            "--output",
            str(output),
            "--scene",
            "small",
            "--voxel-size",
            ".02",
            "--steps",
            "300",
            "--num_envs",
            "2",
            "--episodes_per_env",
            "2" if case["split"] == "train" else "1",
            "--seed",
            str(case["seed"]),
            "--split",
            case["split"],
            "--body-shape",
            case["shape"],
            "--material-density",
            str(case["density"]),
            "--material-friction",
            str(case["friction"]),
            "--young-modulus",
            str(case["young_modulus"]),
            "--body-mass",
            str(case["mass"]),
            "--initial-linear-velocity",
            *map(str, case["linear"]),
            "--initial-angular-velocity",
            *map(str, case["angular"]),
        ]
        if args.expanded:
            command.extend(["--body-size", ".10", ".10", ".10"])
        print(json.dumps({"starting": case}), flush=True)
        with (args.output_dir / (case["name"] + ".log")).open("w") as log:
            try:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=600)
                success = result.returncode == 0
            except subprocess.TimeoutExpired:
                success = False
        report["completed" if success else "failed"].append(case["name"])
        report["updated_at"] = datetime.now(timezone.utc).isoformat()
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(path)
        print(json.dumps({"finished": case["name"], "success": success}), flush=True)
    return bool(report["failed"])


if __name__ == "__main__":
    sys.exit(main())
