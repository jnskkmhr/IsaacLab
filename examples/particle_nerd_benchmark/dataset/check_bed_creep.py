# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compare how much a bare bed keeps settling under different implicit MPM step settings.

The bed is settled for 2 s (as in the dataset) and then followed for 10 s. The vertical change of the top 3 cm, a
mid layer and the floor layer is reported after 2, 5 and 10 s.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import warp as wp

from .bench_mpm import build
from .episodes import DT

SETTLE_S = 2.0
REPORT_S = (2.0, 5.0, 10.0)


def creep(substeps: int, collider_basis: str | None, terrain: str, device: str) -> dict:
    model, solver, _ = build(1, terrain, device, collider_basis)
    s0, s1 = model.state(), model.state()

    def advance(frames):
        nonlocal s0, s1
        for _ in range(frames):
            for _ in range(substeps):
                solver.step(s0, s1, None, None, DT / substeps)
                s0, s1 = s1, s0

    initial = s0.particle_q.numpy().copy()
    advance(round(SETTLE_S / DT))
    start = s0.particle_q.numpy().copy()
    top = np.percentile(start[:, 2], 99.5)
    layers = {
        "top3": start[:, 2] > top - 0.03,
        "mid": (start[:, 2] < top - 0.10) & (start[:, 2] > top - 0.18),
        "floor": start[:, 2] < top - 0.20,
    }
    row = {
        "substeps": substeps,
        "collider_basis": collider_basis or "default",
        "terrain": terrain,
        "settle_drop_of_top_mm": round(float(np.percentile(initial[:, 2], 99.5) - top) * 1e3, 2),
    }
    done = 0.0
    for seconds in REPORT_S:
        advance(round((seconds - done) / DT))
        done = seconds
        now = s0.particle_q.numpy()
        for name, mask in layers.items():
            row[f"{name}_dz_mm@{seconds:g}s"] = round(float((now[mask, 2] - start[mask, 2]).mean()) * 1e3, 2)
    row["finite"] = bool(np.isfinite(s0.particle_q.numpy()).all())
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--terrain", default="flat")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    device = "cuda:0"
    wp.init()
    wp.set_device(device)
    rows = []
    for substeps, basis in ((2, None), (1, None), (1, "pic27"), (2, "pic27")):
        row = creep(substeps, basis, args.terrain, device)
        rows.append(row)
        print(json.dumps(row), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
