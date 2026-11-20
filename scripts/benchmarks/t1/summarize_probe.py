# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Print one line per T1 probe result: status, cost, and the mean box force of each pass.

Usage:
    python3 scripts/benchmarks/t1/summarize_probe.py logs/t1/smoke/*.json
"""

import json
import os
import sys

import numpy as np

for path in sys.argv[1:]:
    with open(path) as handle:
        result = json.load(handle)
    line = [f"{result['terrain']:12s} sleep {result['sleep']:3s}", result["status"]]
    if result["status"] != "ok":
        print(*line, "|", result.get("stage"), "|", result.get("error", "")[:200])
        continue
    step, util = result["step_ms"], result["gpu_utilization_percent"]
    line += [
        f"| particles {result['particle_count']:,}",
        f"| step ms mean {step['mean']:.1f} p95 {step['p95']:.1f}",
        f"| GPU util mean {util.get('mean', float('nan')):.0f}% max {util.get('max', float('nan')):.0f}%",
        f"| VRAM {result['peak_process_MiB']:.0f} MiB",
        f"| surface {100 * result['surface_z_after_presettle_m']:.1f} cm",
    ]
    data = np.load(os.path.splitext(path)[0] + ".npz")
    marks, force = result["phase_end_steps"], data["force_world"]
    names = sorted(marks, key=marks.get)
    for name in ("out", "back", "lap_1", "lap_2", "hold"):
        if name in marks:
            begin = marks[names[names.index(name) - 1]]
            # Skip the entry transient: the last 60% of the pass.
            part = force[begin + int(0.4 * (marks[name] - begin)) : marks[name]]
            line.append(f"| {name} F = ({part[:, 0].mean():.0f}, {part[:, 1].mean():.0f}, {part[:, 2].mean():.0f}) N")
    if result.get("activity_asleep_support_awake_last"):
        line.append(f"| asleep/support/awake {result['activity_asleep_support_awake_last']}")
    if "cache_saved_records_env0" in result:
        line.append(
            f"| saved {result['cache_saved_records_env0']:,} records, {result['cache_visited_tiles_env0']} tiles"
        )
    print(*line)
