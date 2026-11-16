# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Summarize matched rollouts with equal trajectory weights and explicit failures."""

import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np

CORE_FIELDS = (
    "global_rmse_mm",
    "near_contact_rmse_mm",
    "moving_rmse_mm",
    "surface_mean_mm",
    "surface_p95_mm",
    "surface_mean_mm_30mm_grid",
    "surface_p95_mm_30mm_grid",
    "mass_discrepancy",
    "velocity_rmse_m_s",
    "velocity_reference_mass_coverage",
    "reference_mass_kg",
    "predicted_mass_kg",
    "reference_ke_j",
    "predicted_ke_j",
    "reference_below_body_mass_kg",
    "predicted_below_body_mass_kg",
    "reference_inside_body_mass_kg",
    "predicted_inside_body_mass_kg",
)


def table(path):
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    return {
        key: np.asarray([float(row[key]) if row.get(key) else np.nan for row in rows])
        for key in (rows[0] if rows else ())
    }


def describe(values):
    """Statistics across trajectory summaries; unavailable values never become zero."""
    available = np.asarray([value for value in values if value is not None and np.isfinite(value)])
    return {
        "available_trajectories": len(available),
        "unavailable_trajectories": len(values) - len(available),
        "mean": float(available.mean()) if len(available) else None,
        "sample_std": float(available.std(ddof=1)) if len(available) > 1 else None,
        "min": float(available.min()) if len(available) else None,
        "max": float(available.max()) if len(available) else None,
    }


def episode_summary(path):
    meta = json.loads(path.read_text())
    complete = meta["failure"] is None and meta["completed_steps"] == meta["expected_steps"]
    result = {
        "id": meta["episode"]["id"],
        "shape": meta["episode"]["shape"],
        "motion_mode": meta["episode"]["mode"],
        "motion": meta["episode"]["motion"],
        "complete": complete,
        "completed_duration_s": meta["completed_steps"] * meta["native_dt_s"],
        "expected_duration_s": meta["expected_steps"] * meta["native_dt_s"],
        "failure": meta["failure"],
        "field_failure": meta.get("field_failure"),
        "surface_failures": meta["surface_failures"],
        "reference_sha256": meta["reference_sha256"],
        "has_force_reference": "force_reference_particles" in meta,
        "metrics": {key + suffix: None for key in CORE_FIELDS for suffix in ("_time_mean", "_end")},
        "valid_samples": {},
    }
    for name in ("particle_metrics", "field_metrics"):
        values = table(path.parent / (name + ".csv"))
        for key, samples in values.items():
            if key == "time_s":
                continue
            valid = np.isfinite(samples)
            scheduled = len(samples)
            if key.startswith("surface_"):
                quotient = values["time_s"] / meta["surface_metric_interval_s"]
                scheduled = int(np.isclose(quotient, np.round(quotient)).sum())
            result["valid_samples"][key] = {"available": int(valid.sum()), "scheduled": scheduled}
            result["metrics"][key + "_time_mean"] = float(samples[valid].mean()) if valid.any() else None
            result["metrics"][key + "_end"] = float(samples[-1]) if complete and np.isfinite(samples[-1]) else None
        if name == "field_metrics":
            for prefix in ("reference", "predicted"):
                key = prefix + "_below_body_mass_kg"
                if key in values and len(values[key]):
                    change = values[key][-1] - values[key][0]
                    result["metrics"][key + "_change_from_initial"] = (
                        float(change) if complete and np.isfinite(change) else None
                    )
            for key in ("surface_mean_mm", "surface_p95_mm"):
                if key in values and key + "_30mm_grid" in values:
                    difference = values[key + "_30mm_grid"] - values[key]
                    valid = np.isfinite(difference)
                    result["metrics"][key + "_grid_change_time_mean"] = (
                        float(difference[valid].mean()) if valid.any() else None
                    )
    for source in ("reference_particles", "predicted_particles"):
        if errors := meta.get("force_" + source):
            result["metrics"][source + "_force_vector_rmse_n"] = errors["vector_rmse_n"]
            for index, axis in enumerate("xyz"):
                for key in ("rmse", "bias"):
                    result["metrics"][source + f"_f{axis}_{key}_n"] = errors[f"component_{key}_n"][index]
            result[source + "_unconverged_force_steps"] = errors["unconverged_steps"]
    if result["has_force_reference"]:
        result["force_windows"] = {}
        with h5py.File(path.parent / "prediction.h5", "r") as data:
            for source in ("reference", "reference_particles", "predicted_particles"):
                forces = data["force_" + source][:]
                times = np.arange(1, len(forces) + 1) * meta["native_dt_s"]
                windows = []
                for start in np.arange(0, times[-1], 1.0):
                    end = min(start + 1.0, times[-1])
                    selected = forces[(times > start) & (times <= end)]
                    windows.append(
                        {
                            "start_exclusive_s": float(start),
                            "end_inclusive_s": float(end),
                            "samples": len(selected),
                            "mean_fx_fy_fz_n": selected.mean(0).tolist(),
                            "frame_std_fx_fy_fz_n": selected.std(0).tolist(),
                        }
                    )
                result["force_windows"][source] = windows
    return meta, result


def aggregate(records):
    names = sorted({key for record in records for key in record["metrics"]})
    return {
        "trajectories": len(records),
        "complete_trajectories": sum(record["complete"] for record in records),
        "force_reference_trajectories": sum(record["has_force_reference"] for record in records),
        "failed_trajectories": [record["id"] for record in records if not record["complete"]],
        "field_failure_trajectories": [record["id"] for record in records if record["field_failure"]],
        "surface_failure_trajectories": [record["id"] for record in records if record["surface_failures"]],
        "complete_rollouts": {
            key: describe([record["metrics"].get(key) for record in records if record["complete"]]) for key in names
        },
        "available_duration_including_failed_rollouts": {
            key: describe([record["metrics"].get(key) for record in records]) for key in names
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evaluation", type=Path)
    parser.add_argument("--expected", type=int, help="Fail if the expected number of trajectory results is missing")
    parser.add_argument(
        "--protocol-test",
        action="store_true",
        help="Require exactly the 60 (or 100 combined) held-out campaign IDs",
    )
    args = parser.parse_args()
    paths = sorted(args.evaluation.glob("*/evaluation.json"))
    if not paths or (args.expected is not None and len(paths) != args.expected):
        raise ValueError(f"Expected {args.expected or 'at least one'} trajectory results; found {len(paths)}")
    identities, records = set(), []
    checkpoints, implementations, evaluations = set(), set(), set()
    for path in paths:
        meta, record = episode_summary(path)
        if record["id"] in identities:
            raise ValueError("Duplicate episode ID in evaluation results")
        identities.add(record["id"])
        checkpoints.add(meta["checkpoint_sha256"])
        implementations.add(json.dumps(meta["network_source_sha256"], sort_keys=True))
        evaluations.add(json.dumps(meta["evaluation_source_sha256"], sort_keys=True))
        records.append(record)
    if len(checkpoints) != 1 or len(implementations) != 1 or len(evaluations) != 1:
        raise ValueError("Results from different checkpoints, network code, or evaluation code cannot be combined")
    if args.protocol_test:
        from .protocol import episodes

        expected_obj = {record["id"] for record in episodes(include_freefall=False) if record["split"] == "test"}
        expected_all = {record["id"] for record in episodes(include_freefall=True) if record["split"] == "test"}
        expected = expected_all if any(i.startswith("freefall-") for i in identities) else expected_obj
        if identities != expected:
            raise ValueError(
                f"Test-set mismatch: missing={sorted(expected - identities)}, extra={sorted(identities - expected)}"
            )
    groups = {"combined": records}
    for record in records:
        groups.setdefault(record["shape"], []).append(record)
        groups.setdefault(record["shape"] + "/" + record["motion_mode"], []).append(record)
    report = {
        "checkpoint_sha256": next(iter(checkpoints)),
        "network_source_sha256": json.loads(next(iter(implementations))),
        "evaluation_source_sha256": json.loads(next(iter(evaluations))),
        "aggregation": (
            "Compute each time mean within one trajectory, then give trajectories equal weight. "
            "Sample standard deviation measures variability across trajectory summaries, not frames. "
            "Complete rollouts and available-duration results including failures are reported separately. "
            "Unavailable measurements are counted explicitly; end values require a complete rollout."
        ),
        "groups": {key: aggregate(values) for key, values in groups.items()},
        "episodes": records,
    }
    (args.evaluation / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    metric_names = sorted({key for record in records for key in record["metrics"]})
    columns = ("id", "shape", "motion_mode", "motion", "complete", "completed_duration_s", "expected_duration_s")
    with (args.evaluation / "per_trajectory.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=[*columns, *metric_names])
        writer.writeheader()
        writer.writerows({**{key: record[key] for key in columns}, **record["metrics"]} for record in records)
    print(json.dumps({"trajectories": len(records), "complete": sum(record["complete"] for record in records)}))


if __name__ == "__main__":
    main()
