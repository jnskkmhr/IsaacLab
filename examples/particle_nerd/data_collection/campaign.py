# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Collect deterministic, episode-separated shards using the validated MPM scene."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import h5py
import numpy as np


def specification(index, attempt=0):
    seed = 510000 + index * 10 + attempt
    rng = np.random.default_rng(seed)
    # Each consecutive group of ten contains two controls, four pushes, four drags.
    case = ("rest", "long-push", "continuous-drag", "long-push", "continuous-drag")[index % 5]
    split_number = int(hashlib.sha256(f"particle-gnn-v1:{index}".encode()).hexdigest()[:8], 16) % 10
    return dict(
        index=index,
        seed=seed,
        attempt=attempt,
        case=case,
        mass_kg=float(rng.uniform(5.0, 10.0)),
        split="test" if split_number == 0 else "validation" if split_number == 1 else "train",
    )


def validate(path):
    with h5py.File(path, "r") as f:
        m = json.loads(f.attrs["manifest"])
        assert m["status"] == "complete" and m["particle_count"] == 16384
        assert m["scene"]["voxel_size_m"] == 0.04 and not m["scene"]["container_walls"]
        assert np.count_nonzero(f["frame_steps"][:] >= m["settled_step"]) == 251
        assert np.all(np.diff(f["frame_steps"][:]) == 4)
        for key in f:
            for start in range(0, len(f[key]), 16):
                if not np.isfinite(f[key][start : start + 16]).all():
                    raise ValueError(f"Nonfinite {key}")
        if m["max_fraction_particle_centers_below_ground_by_more_than_grid"] > 0:
            raise ValueError("Particle center below rigid ground by more than 0.04 m")
        applied = f["applied_feedback_wrench"][:]
        np.testing.assert_allclose(applied[1:], f["material_on_body_wrench"][:-1], atol=0.01, rtol=1e-4)
        residual = m["body_mass_kg"] * np.diff(f["body_velocity"][:, :3], axis=0) / m["dt"]
        residual -= applied[:, :3] + f["external_wrench"][:, :3] + [0.0, 0.0, -9.81 * m["body_mass_kg"]]
        if m.get("release_step") is not None:
            residual = residual[m["release_step"] :]
        # Direct ground/box rigid contact can contribute a force absent from
        # the MPM recorder. Keep such an episode, but label the residual honestly.
        return dict(
            post_settle_frames=251,
            total_frames=len(f["frame_steps"]),
            max_unaccounted_body_force_N=float(np.linalg.norm(residual, axis=1).max()),
            bytes=path.stat().st_size,
        )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--episodes", type=int, default=1000)
    p.add_argument("--geometry", choices=("box", "shovel"), default="box")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--shards", type=int, default=1)
    args = p.parse_args()
    assert 0 <= args.shard < args.shards and args.episodes > 0
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(exist_ok=True)
    started = time.time()
    records = []
    for index in range(args.shard, args.episodes, args.shards):
        name = f"episode-{index:05d}"
        completed = out / f"{name}.campaign.json"
        if completed.exists():
            record = json.loads(completed.read_text())
            validate(out / record["file"])
            records.append(record)
            continue
        for attempt in range(3):
            spec = specification(index, attempt)
            if args.geometry == "shovel":
                from particle_nerd.shovel.asset import ASSET_DIR

                spec["mass_kg"] = json.loads((ASSET_DIR / "metadata.json").read_text())["body_mass_kg"]
                spec["geometry"] = "shovel"
            attempt_name = f"{name}-attempt-{attempt}"
            path = out / f"{attempt_name}.hdf5"
            if not path.exists():
                command = [
                    sys.executable,
                    "-m",
                    "particle_nerd.data_collection.collect",
                    "--geometry",
                    args.geometry,
                    "--case",
                    spec["case"],
                    "--body-mass",
                    str(spec["mass_kg"]),
                    "--seed",
                    str(spec["seed"]),
                    "--randomize-motion",
                    "--name",
                    attempt_name,
                    "--output",
                    str(out),
                ]
                with (out / "logs" / f"{attempt_name}.log").open("w") as log:
                    run = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
                if run.returncode:
                    print(
                        json.dumps(
                            dict(
                                event="rejected",
                                **spec,
                                reason="simulation failed; see log",
                            )
                        ),
                        flush=True,
                    )
                    continue
            try:
                quality = validate(path)
            except (ValueError, AssertionError) as error:
                print(
                    json.dumps(dict(event="rejected", **spec, reason=str(error)[:500])),
                    flush=True,
                )
                continue
            record = dict(**spec, file=path.name, quality=quality)
            completed.write_text(json.dumps(record, indent=2) + "\n")
            records.append(record)
            print(
                json.dumps(dict(event="episode_complete", completed=len(records), **record)),
                flush=True,
            )
            break
        else:
            raise RuntimeError(f"Episode {index} failed three seeds; inspect logs before resubmission")
        status = dict(
            shard=args.shard,
            shards=args.shards,
            total_requested=args.episodes,
            completed=len(records),
            elapsed_s=time.time() - started,
            episodes=records,
        )
        temporary = out / "progress.partial.json"
        temporary.write_text(json.dumps(status, indent=2) + "\n")
        os.replace(temporary, out / "progress.json")
    (out / "COMPLETE.json").write_text(json.dumps(dict(shard=args.shard, episodes=records), indent=2) + "\n")


if __name__ == "__main__":
    main()
