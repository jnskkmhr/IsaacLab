# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Audit downloaded collection metadata without copying the full particle dataset."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected", type=int, default=1000)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    records = [json.loads(p.read_text()) for p in sorted(args.data.rglob("*.campaign.json"))]
    ids = [r["index"] for r in records]
    assert len(ids) == len(set(ids)), "Duplicate accepted episode IDs"
    assert all(0 <= i < args.expected for i in ids)
    if args.require_complete:
        assert set(ids) == set(range(args.expected)), f"Only {len(ids)}/{args.expected} episodes present"
    assert all(r["quality"]["post_settle_frames"] == 251 for r in records)
    canonical = json.dumps(sorted(records, key=lambda r: r["index"]), sort_keys=True, separators=(",", ":"))
    result = dict(
        accepted_metadata_sha256=hashlib.sha256(canonical.encode()).hexdigest(),
        completed=len(records),
        expected=args.expected,
        complete=len(records) == args.expected,
        splits=dict(Counter(r["split"] for r in records)),
        cases=dict(Counter(r["case"] for r in records)),
        mass_min_kg=min((r["mass_kg"] for r in records), default=None),
        mass_max_kg=max((r["mass_kg"] for r in records), default=None),
        stored_bytes=sum(r["quality"]["bytes"] for r in records),
        total_particle_frames=sum(r["quality"]["total_frames"] for r in records),
        post_settle_particle_frames=sum(r["quality"]["post_settle_frames"] for r in records),
        episodes_with_unaccounted_body_force_over_0_01_N=sum(
            r["quality"]["max_unaccounted_body_force_N"] > 0.01 for r in records
        ),
        rejected_seed_attempts_before_acceptance=sum(r["attempt"] for r in records),
        missing_episode_ids=sorted(set(range(args.expected)) - set(ids)),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "missing_episode_ids"}, indent=2))


if __name__ == "__main__":
    main()
