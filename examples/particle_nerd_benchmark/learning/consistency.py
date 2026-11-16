# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Measure sand drift during a verified fixed-object hold from saved paired states."""

import argparse
import json
import os
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .evaluate import below_body_mass, file_hash
from .metrics import CommonGrid, surface_distances


class TemporalGrid(CommonGrid):
    """Use the same CIC/isosurface rules with a larger, explicitly bounded CPU grid.

    Escaped particles remain included. This offline diagnostic permits 100 million
    nodes; the original paired evaluator's 20-million-node failures remain recorded.
    """

    def __init__(self, lower, upper, spacing):
        self.spacing = float(spacing)
        self.lower = np.floor(np.asarray(lower) / spacing) * spacing
        self.shape = tuple((np.ceil((np.asarray(upper) - self.lower) / spacing).astype(int) + 1).tolist())
        if np.prod(self.shape) > 100_000_000:
            raise ValueError("Temporal evaluation grid exceeds 100 million nodes; no particles have been removed")


def particle_drift(position, anchor, mass):
    """Physical-mass-weighted RMS displacement and signed mean vertical change, in mm."""
    change = np.asarray(position, dtype=np.float64) - anchor
    total = mass.sum()
    return {
        "rms_displacement_mm": float(1000 * np.sqrt(np.sum(mass * np.sum(change**2, axis=-1)) / total)),
        "mean_vertical_change_mm": float(1000 * np.sum(mass * change[:, 2]) / total),
    }


def fixed_body_check(states, anchor_frame):
    """Reject translating, rotating, or nonzero-velocity body inputs in the measurement interval."""
    selected = np.asarray(states[anchor_frame:], dtype=np.float64)
    translation = np.linalg.norm(selected[:, :3] - selected[0, :3], axis=1)
    quaternions = selected[:, 3:7] / np.linalg.norm(selected[:, 3:7], axis=1)[:, None]
    angle = 2 * np.arccos(np.clip(np.abs(quaternions @ quaternions[0]), 0, 1))
    linear = np.linalg.norm(selected[:, 7:10], axis=1)
    angular = np.linalg.norm(selected[:, 10:13], axis=1)
    result = {
        "maximum_translation_m": float(translation.max()),
        "maximum_rotation_rad": float(angle.max()),
        "maximum_linear_speed_m_s": float(linear.max()),
        "maximum_angular_speed_rad_s": float(angular.max()),
    }
    if translation.max() > 1e-7 or angle.max() > 1e-6 or linear.max() > 1e-7 or angular.max() > 1e-7:
        raise ValueError(f"Object is not fixed over the requested interval: {result}")
    return result


def plot(root, data, model_label="Particle NeRD"):
    """Plot both methods relative to their own state at the same measurement start."""
    plt.rcParams.update({"font.family": "serif", "font.size": 12, "axes.spines.top": False, "axes.spines.right": False})
    figure, axes = plt.subplots(3, 2, figsize=(13.2, 11.1))
    figure.subplots_adjust(top=0.95, bottom=0.23, wspace=0.27, hspace=0.42)
    keys = (
        ("particles", "rms_displacement_mm", "RMS particle drift (mm)"),
        ("particles", "mean_vertical_change_mm", "Mean vertical drift (mm)"),
        ("fields", "surface_mean_mm", "Sand-surface change (mm)"),
        ("fields", "temporal_mass_discrepancy", "Normalized grid-mass redistribution"),
        ("fields", "below_body_mass_kg", "Sand mass beneath object (kg)"),
        ("particles", "below_floor_mass_percent", "Mass below rigid tray floor (%)"),
    )
    for method, label, color in (("reference", "Implicit MPM", "#193c64"), ("predicted", model_label, "#248044")):
        for axis, (kind, key, ylabel) in zip(axes.flat, keys, strict=True):
            rows = data[method][kind]
            selected = [r for r in rows if key in r]
            axis.plot(
                [r["time_s"] for r in selected], [r[key] for r in selected], label=label, color=color, linewidth=1.3
            )
            axis.set_ylabel(ylabel)
            if key == "surface_mean_mm":
                axis.plot(
                    [r["time_s"] for r in selected],
                    [r["surface_p95_mm"] for r in selected],
                    color=color,
                    linestyle="--",
                    linewidth=1.1,
                )
    for index, axis in enumerate(axes.flat):
        axis.set_title(f"({chr(97 + index)})", loc="left")
        axis.set_xlabel("Simulation time (s)")
        axis.grid(alpha=0.15)
    figure.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.5, 1.01), ncol=2)
    figure.text(
        0.5,
        0.025,
        "Figure 7: Fixed-object hold, measured at 3–23 s. Both methods start from identical particles at 0 s.\n"
        "(a) Mass-weighted RMS drift from each method's own 3 s state; (b) "
        "mass-weighted mean vertical drift (negative down).\n"
        "(c) Distance to that method's 3 s material isosurface: solid mean, dashed "
        "95th percentile (includes sides/bottom).\n"
        "(d) Sum of absolute grid-cell mass changes from 3 s, divided by total "
        "physical mass; dimensionless, range 0–2.\n"
        "(e) Absolute mass in the rectangle enclosing the object's world x/y projection, extending 0.10 m below its\n"
        "lowest z and clipped above the tray floor at z = −0.30 m. (f) Absolute mass "
        "fraction with centers below that floor.\n"
        "The object stops at 2 s. The 3 s anchor does not reset particles or assert "
        "equilibrium. One paired run per shape.\n"
        "No alignment, gate, projection, filtering, or particle removal; MPM settling and leakage remain included.",
        ha="center",
        fontsize=10,
    )
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(root / f"spatiotemporal_consistency.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evaluation", type=Path)
    args = parser.parse_args()
    root = args.evaluation
    meta = json.loads((root / "evaluation.json").read_text())
    record = meta["episode"]
    if record.get("motion") != "hold" or "consistency_start_s" not in record:
        raise ValueError("This metric requires an explicitly declared fixed-hold reference")
    if meta["failure"] or meta["completed_steps"] != meta["expected_steps"]:
        raise ValueError("Do not report a complete hold metric for a failed or truncated rollout")
    reference_path = Path(meta["reference_path"])
    if file_hash(reference_path) != meta["reference_sha256"]:
        raise ValueError("Reference trajectory changed since evaluation")
    manifest = json.loads((reference_path.parent / "manifest.json").read_text())
    dt = meta["native_dt_s"]
    start = record["consistency_start_s"]
    if start != 3.0 or meta["completed_steps"] * dt != 23.0:
        raise ValueError("This declared protocol requires the 3–23 s measurement interval")
    anchor_frame = round(start / dt)
    fields_stride, surface_stride = round(0.1 / dt), round(0.5 / dt)
    result = {
        "episode": record["id"],
        "checkpoint_sha256": meta["checkpoint_sha256"],
        "reference_sha256": meta["reference_sha256"],
        "prediction_sha256": file_hash(root / "prediction.h5"),
        "definition": "Each method is compared to its own state at the shared 3 s anchor; neither state is reset.",
        "start_s": start,
        "end_s": meta["completed_steps"] * dt,
        "late_window_s": [13.0, 23.0],
        "native_dt_s": dt,
        "field_interval_s": 0.1,
        "surface_interval_s": 0.5,
        "surface_grid_m": 0.02,
        "surface_sigma_m": 0.032,
        "surface_density_kg_m3": 800.0,
        "surface_samples_per_direction": 4096,
        "surface_seed": 42,
        "surface_seed_schedule": "42 + native frame index",
        "original_paired_field_failure": meta.get("field_failure"),
        "floor_z_m": -0.30,
        "floor_center_tolerance_m": 1e-6,
        "maximum_grid_nodes": 100_000_000,
        "below_body_region": meta["below_body_region"],
        "surface_failures": [],
        "source_sha256": file_hash(Path(__file__)),
    }
    data = {}
    with h5py.File(reference_path, "r") as ref, h5py.File(root / "prediction.h5", "r") as pred:
        result["body_motion_check"] = fixed_body_check(ref["body_state"], anchor_frame)
        result["collider_input_check"] = fixed_body_check(ref["collider_state"], anchor_frame)
        if not np.array_equal(ref["body_state"][:], pred["body_state"][:]):
            raise ValueError("Predicted and reference objects do not share the recorded fixed pose")
        body = ref["body_state"][anchor_frame]
        mass = ref["particle_mass"][:].astype(np.float64)
        total = mass.sum()
        result["particle_count"], result["physical_mass_kg"] = len(mass), float(total)
        lower, upper = np.full(3, np.inf), np.full(3, -np.inf)
        for handle in (ref, pred):
            for first in range(anchor_frame, len(handle["particle_position"]), 64):
                chunk = handle["particle_position"][first : first + 64]
                lower = np.minimum(lower, chunk.min(axis=(0, 1)))
                upper = np.maximum(upper, chunk.max(axis=(0, 1)))
        grid = TemporalGrid(lower - 0.18, upper + 0.18, 0.02)
        result["grid_lower_m"], result["grid_shape"] = grid.lower.tolist(), grid.shape
        for method, handle in (("reference", ref), ("predicted", pred)):
            positions = handle["particle_position"]
            anchor = positions[anchor_frame].astype(np.float64)
            anchor_mass, _ = grid.deposit(anchor, mass)
            anchor_surface = grid.surface(anchor_mass)
            particles, fields = [], []
            previous = positions[anchor_frame - 1].astype(np.float64)
            for first in range(anchor_frame, len(positions), 32):
                for offset, raw in enumerate(positions[first : first + 32]):
                    frame = first + offset
                    position = raw.astype(np.float64)
                    velocity = (position - previous) / dt
                    row = {"time_s": frame * dt, **particle_drift(position, anchor, mass)}
                    row["kinetic_energy_j"] = float(np.sum(mass * np.sum(velocity**2, axis=1)) / 2)
                    row["rms_speed_m_s"] = float(np.sqrt(2 * row["kinetic_energy_j"] / total))
                    row["below_floor_mass_percent"] = float(100 * mass[position[:, 2] < -0.300001].sum() / total)
                    row["below_floor_1cm_mass_percent"] = float(100 * mass[position[:, 2] < -0.31].sum() / total)
                    row["minimum_particle_z_m"] = float(position[:, 2].min())
                    particles.append(row)
                    if frame % fields_stride == 0:
                        current_mass, _ = grid.deposit(position, mass)
                        field = {
                            "time_s": frame * dt,
                            "temporal_mass_discrepancy": float(np.abs(current_mass - anchor_mass).sum() / total),
                            "below_body_mass_kg": below_body_mass(
                                position, body, record["shape"], manifest["body_size_m"], mass
                            ),
                        }
                        if frame % surface_stride == 0:
                            try:
                                values = surface_distances(anchor_surface, grid.surface(current_mass), seed=42 + frame)
                                if not all(np.isfinite(value) for value in values.values()):
                                    raise ValueError("Nonfinite temporal surface distance")
                                field.update(values if frame != anchor_frame else {key: 0.0 for key in values})
                            except (ValueError, RuntimeError) as error:
                                result["surface_failures"].append(
                                    {"method": method, "time_s": frame * dt, "reason": str(error)}
                                )
                        fields.append(field)
                    previous = position
            late = [row for row in particles if 13.0 <= row["time_s"] <= 23.0]
            t = np.asarray([row["time_s"] for row in late])
            z = np.asarray([row["mean_vertical_change_mm"] for row in late])
            slope = np.dot(t - t.mean(), z - z.mean()) / np.sum((t - t.mean()) ** 2)
            baseline = fields[0]["below_body_mass_kg"]
            result[method] = {
                "final": {**particles[-1], **fields[-1]},
                "late_vertical_drift_rate_mm_s": float(slope),
                "late_mean_rms_speed_m_s": float(np.mean([row["rms_speed_m_s"] for row in late])),
                "late_mean_kinetic_energy_j": float(np.mean([row["kinetic_energy_j"] for row in late])),
                "late_samples": len(late),
                "initial_below_body_mass_kg": baseline,
                "final_below_body_mass_fraction": fields[-1]["below_body_mass_kg"] / baseline if baseline > 0 else None,
            }
            coarse = TemporalGrid(lower - 0.18, upper + 0.18, 0.03)
            a, _ = coarse.deposit(anchor, mass)
            b, _ = coarse.deposit(position, mass)
            try:
                result[method]["surface_final_30mm_grid"] = surface_distances(
                    coarse.surface(a), coarse.surface(b), seed=42 + frame
                )
            except (ValueError, RuntimeError) as error:
                result[method]["surface_final_30mm_grid"] = None
                result["surface_failures"].append(
                    {"method": method, "time_s": frame * dt, "grid_m": 0.03, "reason": str(error)}
                )
            data[method] = {"particles": particles, "fields": fields}
    (root / "consistency_summary.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    (root / "consistency_series.json").write_text(json.dumps(data, allow_nan=False))
    model_label = os.environ.get("BENCHMARK_MODEL_LABEL") or (
        "Particle World"
        if "cconv_warp.py" in meta.get("network_source_sha256", {})
        else ("Particle NeRD-Local" if len(meta.get("network_source_sha256", {})) == 2 else "Particle NeRD")
    )
    plot(root, data, model_label=model_label)
    names = ["consistency_summary.json", "consistency_series.json"] + [
        f"spatiotemporal_consistency.{s}" for s in ("png", "pdf", "svg")
    ]
    provenance = {
        key: result[key] for key in ("checkpoint_sha256", "reference_sha256", "prediction_sha256", "source_sha256")
    }
    provenance["files_sha256"] = {name: file_hash(root / name) for name in names}
    (root / "consistency_provenance.json").write_text(json.dumps(provenance, indent=2))
    print(json.dumps({"episode": record["id"], "metrics": {key: result[key] for key in ("reference", "predicted")}}))


if __name__ == "__main__":
    main()
