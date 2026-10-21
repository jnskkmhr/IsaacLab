# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fail before training on missing, duplicate, nonfinite, or malformed G1 records."""

import argparse
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import h5py
import numpy as np

FRAMES = (
    "time_s",
    "particle_position",
    "particle_velocity",
    "body_pose",
    "body_velocity",
    "foot_material_wrench",
    "joint_position",
    "joint_velocity",
    "applied_generalized_force",
    "action",
    "command",
)
STATIC = (
    "particle_mass",
    "particle_radius",
    "body_com_local",
    "collider_body",
    "collider_local_pose",
    "collider_half_size",
    "collider_friction",
    "foot_body_ids",
)


def selected_trials(schedule, limit):
    trials = schedule["trials"]
    if not limit:
        return trials
    grouped = {}
    for trial in trials:
        grouped.setdefault((trial["group"], trial["split"]), []).append(trial)
    selected = []
    while len(selected) < min(limit, len(trials)):
        for group in grouped.values():
            if group and len(selected) < limit:
                selected.append(group.pop(0))
    return selected


def audit_episode(path, expected):
    if ".partial." in path.name:
        raise ValueError(f"Incomplete trajectory remains: {path}")
    with h5py.File(path, "r") as f:
        manifest = json.loads(f.attrs["manifest"])
        outcome = json.loads(f.attrs["outcome"])
        trial = manifest["trial"]
        trial_id = trial["trial_id"]
        if trial_id not in expected:
            raise ValueError(f"Unexpected trial ID {trial_id}: {path}")
        if trial != expected[trial_id] or manifest["split"] != trial["split"]:
            raise ValueError(f"Trial metadata/split differs from immutable schedule: {path}")
        if manifest["schema"] != "g1_particle_nerd_v1":
            raise ValueError(f"Wrong schema: {path}")
        if not np.isclose(manifest["physics_dt_s"], 0.005) or manifest["record_hz"] != 200:
            raise ValueError(f"Wrong physics or recording rate: {path}")
        if not np.isclose(manifest["policy_dt_s"], 0.02):
            raise ValueError(f"Wrong policy rate: {path}")
        if not np.allclose(manifest["bed_dimensions_m"], [3, 3, 0.3]) or not np.isclose(
            manifest["grid_spacing_m"], 0.04
        ):
            raise ValueError(f"Wrong terrain dimensions or MPM grid spacing: {path}")
        times = f["time_s"][:]
        frames = len(times)
        if frames < 5 or not np.isclose(times[0], 0) or not np.allclose(np.diff(times), 0.005, atol=1e-9, rtol=0):
            raise ValueError(f"Insufficient or nonuniform 200 Hz frames: {path}")
        expected_frames = 1 + round(manifest["requested_duration_s"] / 0.005)
        if outcome["frames"] != frames or not np.isclose(outcome["duration_s"], times[-1]):
            raise ValueError(f"Outcome timing mismatch: {path}")
        if outcome["outcome"] == "duration":
            if frames != expected_frames:
                raise ValueError(f"Full-duration record is truncated: {path}")
        elif outcome["outcome"] == "native_termination":
            if frames > expected_frames:
                raise ValueError(f"Native termination exceeds requested duration: {path}")
        else:
            raise ValueError(f"Unexpected termination status: {path}")
        for name in FRAMES:
            ds = f[name]
            if len(ds) != frames:
                raise ValueError(f"Frame count mismatch {name}: {path}")
            # Read bounded chunks, including every particle coordinate and velocity.
            for start in range(0, frames, 8):
                if not np.isfinite(ds[start : start + 8]).all():
                    raise ValueError(f"Nonfinite {name} near frame {start}: {path}")
        for name in STATIC:
            if not np.isfinite(f[name][:]).all():
                raise ValueError(f"Nonfinite static geometry {name}: {path}")
        if len(f["collider_body"]) != 7 or len(f["foot_body_ids"]) != 2:
            raise ValueError(f"Expected two robot feet and five boundary boxes: {path}")
        particles = f["particle_position"].shape[1]
        bodies = f["body_pose"].shape[1]
        if f["particle_position"].shape != (frames, particles, 3) or f["particle_velocity"].shape != (
            frames,
            particles,
            3,
        ):
            raise ValueError(f"Particle state shape mismatch: {path}")
        if f["body_pose"].shape != (frames, bodies, 7) or f["body_velocity"].shape != (frames, bodies, 6):
            raise ValueError(f"Rigid state shape mismatch: {path}")
        for name in ("particle_mass", "particle_radius"):
            values = f[name][:]
            if values.shape != (particles,) or (values <= 0).any():
                raise ValueError(f"Invalid {name}: {path}")
        for name in ("collider_body", "foot_body_ids"):
            ids = f[name][:]
            if ((ids < 0) | (ids >= bodies)).any():
                raise ValueError(f"Body index out of range in {name}: {path}")
        # Native MPM reference-trial12/metadata.json records body_label[0]
        # as /World/envs/env_0/Robot/Geometry/pelvis. The HDF5 schema omitted
        # labels, so additionally verify the same native topology/asset here.
        robot_asset = manifest["config"]["scene"]["robot"]["spawn"]["usd_path"]
        if (
            bodies != 40
            or list(f["foot_body_ids"][:]) != [6, 13]
            or not robot_asset.endswith("g1_29dof_rev_1_0_box_foot_improved_collision.usda")
        ):
            raise ValueError(f"Cannot verify native pelvis body index for this topology: {path}")
        indices = np.arange(0, frames, 20)
        pelvis = f["body_pose"][indices, 0, :3]
        pelvis_path = dict(
            body_index=0,
            body_label="Robot/Geometry/pelvis",
            label_verification=(
                "Native MPM reference-trial12/metadata.json body_labels[0]; matching asset,40 bodies,foot IDs6/13"
            ),
            label_reference_sha256="d71a79f37ae6c58dbdf92fbf39135436b73b9ddd511c4947713d717a64a9b7d1",
            position_reference="Pelvis body-frame origin, not COM; environment-local XYZ in metres",
            height_reference="Environment-local z=0 is the initial flat sand surface",
            sampling_interval_s=0.1,
            frame_indices=indices.tolist(),
            time_s=times[indices].tolist(),
            position_m=pelvis.tolist(),
            initial_height_m=float(f["body_pose"][0, 0, 2]),
            final_height_m=float(f["body_pose"][-1, 0, 2]),
            final_time_s=float(times[-1]),
        )
        record = dict(
            trial_id=trial_id,
            split=trial["split"],
            group=trial["group"],
            frames=frames,
            duration_s=float(times[-1]),
            outcome=outcome["outcome"],
            particles=particles,
            file_bytes=path.stat().st_size,
            policy_sha256=manifest["policy_sha256"],
            file=str(path),
            pelvis_trajectory=pelvis_path,
        )
        return record


def audit(data, schedule, limit, workers=1):
    expected = {t["trial_id"]: t for t in selected_trials(schedule, limit)}
    records = []
    seen = set()
    paths = sorted(data.rglob("episode-*.hdf5"))
    if not paths:
        raise ValueError("No HDF5 trajectories found")
    if workers < 1:
        raise ValueError("Workers must be positive")

    def add(record):
        trial_id = record["trial_id"]
        if trial_id in seen:
            raise ValueError(f"Duplicate trial ID {trial_id}")
        seen.add(trial_id)
        records.append(record)
        print("AUDIT_EPISODE", json.dumps({k: v for k, v in record.items() if k != "pelvis_trajectory"}), flush=True)

    if workers == 1:
        for path in paths:
            add(audit_episode(path, expected))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            pending = [pool.submit(audit_episode, path, expected) for path in paths]
            try:
                for future in as_completed(pending):
                    add(future.result())
            except BaseException:
                for future in pending:
                    future.cancel()
                raise
    records.sort(key=lambda record: record["trial_id"])
    if seen != expected.keys():
        raise ValueError(f"Missing trial IDs: {sorted(expected.keys() - seen)}")
    if len({r["policy_sha256"] for r in records}) != 1:
        raise ValueError("Inconsistent policy checkpoints across trajectories")
    return dict(
        status="passed",
        trajectory_count=len(records),
        split_counts=dict(Counter(r["split"] for r in records)),
        outcome_counts=dict(Counter(r["outcome"] for r in records)),
        min_frames=min(r["frames"] for r in records),
        max_frames=max(r["frames"] for r in records),
        total_frames=sum(r["frames"] for r in records),
        total_bytes=sum(r["file_bytes"] for r in records),
        records=records,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--expected-limit", type=int)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.data, json.loads(args.schedule.read_text()), args.expected_limit, args.workers)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print("DATASET_AUDIT_PASSED", json.dumps({k: v for k, v in result.items() if k != "records"}), flush=True)


if __name__ == "__main__":
    main()
