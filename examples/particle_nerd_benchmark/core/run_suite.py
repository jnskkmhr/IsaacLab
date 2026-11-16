# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Record and validate every case in separate simulation processes."""

import argparse
import subprocess
import sys
from pathlib import Path

CASES = [
    (motion, "flat")
    for motion in ("horizontal", "vertical", "circle", "rolling", "projectile", "franka", "quadruped", "g1")
]
CASES += [(motion, terrain) for motion in ("horizontal", "rolling") for terrain in ("ascend", "descend", "wave")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True, help="G1 policy with training_config.json alongside")
    parser.add_argument("--duration", type=float, default=6.0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    failed = []
    for motion, terrain in CASES:
        output = args.output / f"{motion}-{terrain}"
        command = [
            sys.executable,
            "-m",
            "particle_nerd_benchmark.core.run",
            "--motion",
            motion,
            "--terrain",
            terrain,
            "--duration",
            str(args.duration),
            "--output",
            str(output),
        ]
        if motion == "g1":
            command.extend(["--checkpoint", str(args.checkpoint)])
        with (args.output / f"{motion}-{terrain}.log").open("x") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode == 0:
                result = subprocess.run(
                    [sys.executable, "-m", "particle_nerd_benchmark.core.evaluate", str(output)],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
        if result.returncode:
            failed.append(output.name)
        print(f"{output.name}: {'FAILED' if result.returncode else 'validated'}", flush=True)
    if failed:
        raise SystemExit("Failed cases: " + ", ".join(failed))


if __name__ == "__main__":
    main()
