# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate recurrent object-model rollouts against matched native MPM recordings."""

import argparse
import csv
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path

import h5py
import numpy as np
import torch
import warp as wp

from .data import Episodes
from .forces import NeRDRecordedForceEstimator, RecordedForceEstimator
from .metrics import CommonGrid, body_distance, force_errors, particle_rmse, surface_distances
from .network import (
    CONTACT_SETTINGS,
    OBJECT_PIPELINE,
    ObjectParticleStepper,
    ParticleNeRDLocalNetwork,
    ParticleTransolver,
    WorldParticleNetwork,
    features,
    features_particle_nerd_local,
    features_world_particle,
    nerd_source_hashes,
    validate_object_checkpoint,
)


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def below_body_mass(points, state, shape, size, mass, depth=0.10):
    """Mass below the body's world-aligned bounding box, clipped at the tray floor.

    The horizontal region is the rectangle enclosing the body's world x/y
    projection. Its upper plane passes through the body's lowest world z.
    Sphere rotation therefore cannot rotate the region away from the sand below.
    """
    relative = points - state[:3]
    half = np.asarray(size) / 2
    q, w = state[3:6], state[6]
    axes = np.eye(3) + 2 * np.cross(q, np.cross(q, np.eye(3)) + w * np.eye(3))
    if shape == "sphere":
        extent = np.full(3, half[0])
    elif shape == "cylinder":
        axis = axes[2]
        extent = half[2] * np.abs(axis) + half[0] * np.sqrt(np.maximum(0.0, 1 - axis**2))
    elif shape == "box":
        extent = np.abs(axes).T @ half
    else:
        raise ValueError(shape)
    footprint = (np.abs(relative[:, 0]) <= extent[0]) & (np.abs(relative[:, 1]) <= extent[1])
    keep = footprint & (relative[:, 2] < -extent[2]) & (relative[:, 2] >= -extent[2] - depth) & (points[:, 2] >= -0.30)
    return float(mass[keep].sum())


@torch.no_grad()
def predict(model, sample):
    """Full recurrent prediction; stop explicitly on nonfinite or unbounded states."""
    is_world_particle = isinstance(model, (WorldParticleNetwork, ParticleNeRDLocalNetwork))
    use_vel_state = is_world_particle and bool(model.config.get("translation_invariant", False))
    history = sample["history"]
    stepper = (
        None
        if is_world_particle
        else ObjectParticleStepper(
            sample["kinds"], sample["sizes"], sample["friction"], sample["material"], sample["dt"]
        )
    )
    if stepper is not None and sample["collider_input"] != OBJECT_PIPELINE["collider_input"]:
        raise ValueError("NeRD evaluation requires pre-rigid collider inputs")
    positions = [history[:, -1].cpu().numpy()]
    failure = None
    for step, (pose, velocity) in enumerate(zip(sample["poses"], sample["velocities"], strict=True)):
        if is_world_particle:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if use_vel_state:
                    current, vel_next = model.step_from_history(
                        history,
                        pose.contiguous(),
                        velocity.contiguous(),
                        sample["kinds"],
                        sample["sizes"],
                        sample["friction"],
                        sample["material"],
                        dt=sample["dt"],
                        return_vel=True,
                    )
                else:
                    current = model.step_from_history(
                        history,
                        pose.contiguous(),
                        velocity.contiguous(),
                        sample["kinds"],
                        sample["sizes"],
                        sample["friction"],
                        sample["material"],
                        dt=sample["dt"],
                    )
        else:
            current = stepper.step(model, history, pose, velocity)
        x = current.cpu().numpy()
        if not np.isfinite(x).all() or np.max(np.abs(x)) > 5.0:
            failure = {
                "time_s": (step + 1) * sample["dt"],
                "reason": "Nonfinite particle state or a particle more than 5 m from the tray origin",
            }
            break
        positions.append(x)
        if use_vel_state:
            prev_synth = current - sample["dt"] * vel_next
            history = torch.stack((history[:, 1], prev_synth, current), dim=1).contiguous()
        else:
            history = torch.cat((history[:, 1:], current[:, None]), dim=1).contiguous()
    return np.stack(positions), failure


def evaluate_one(model, dataset, index, output, checkpoint_sha, device="cuda:0"):
    is_nerd = isinstance(model, ParticleTransolver)
    record, manifest, path = dataset.records[index]
    output.mkdir(parents=True, exist_ok=False)
    with h5py.File(path, "r") as handle:
        steps = len(handle["particle_position"]) - 1
        mass = handle["particle_mass"][:].astype(np.float64)
        body_mass = float(handle["body_mass"][0])
        body_inertia = handle["body_inertia"][0]
        native_force = handle["material_force"][:] if "material_force" in handle else None
        body = handle["body_state"][:]
        collider = handle["collider_state"][:]
    sample = dataset.sequence(np.random.default_rng(42), steps, device, index=index, start=0)
    predicted, failure = predict(model, sample)
    reference = np.concatenate((sample["history"][:, -1].cpu().numpy()[None], sample["targets"].cpu().numpy()))
    dt = sample["dt"]
    if isinstance(model, ParticleNeRDLocalNetwork):
        source_objs = (features_particle_nerd_local, ParticleNeRDLocalNetwork)
    elif isinstance(model, WorldParticleNetwork):
        source_objs = (features_world_particle, WorldParticleNetwork)
    else:
        source_objs = (features, ParticleTransolver, ObjectParticleStepper)
    status = {
        "episode": record,
        "checkpoint_sha256": checkpoint_sha,
        "reference_path": str(path.resolve()),
        "reference_sha256": file_hash(path),
        "network_source_sha256": nerd_source_hashes()
        if is_nerd
        else {Path(inspect.getfile(obj)).name: file_hash(inspect.getfile(obj)) for obj in source_objs},
        "evaluation_source_sha256": {
            name: file_hash(Path(__file__).parent / name)
            for name in ("evaluate.py", "metrics.py", "forces.py", "data.py")
        },
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("torch", "warp-lang", "numpy", "scipy", "scikit-image", "trimesh", "rtree")
        },
        "mode": "Recurrent particle rollout with recorded body inputs",
        "failure": failure,
        "expected_steps": steps,
        "completed_steps": len(predicted) - 1,
        "native_dt_s": dt,
        "particle_metric_interval_s": dt,
        "field_metric_interval_s": 0.10,
        "surface_metric_interval_s": 0.20,
        "grid_spacing_m": 0.02,
        "surface_sensitivity_spacing_m": 0.03,
        "surface_kernel_sigma_m": 0.032,
        "surface_density_threshold_kg_m3": 800.0,
        "surface_samples_per_direction": 4096,
        "surface_seed": 42,
        "contact_distance_m": 0.05,
        "moving_speed_m_s": 0.01,
        "cell_mass_threshold_kg": 0.01 * float(mass.min()),
        "below_body_depth_m": 0.10,
        "below_body_region": (
            "Horizontal rectangle enclosing the body's world x/y projection, from 0 to 0.10 m below "
            "its lowest world z; clipped at the rigid floor z = -0.30 m"
        ),
        "active_selection": OBJECT_PIPELINE["active_k"] if is_nerd else None,
        "displacement_gate": is_nerd,
        "particle_projection": OBJECT_PIPELINE["particle_projection"] if is_nerd else False,
        "pipeline": OBJECT_PIPELINE if is_nerd else None,
        "force_filter": None,
        "force_velocity": "Forward finite difference from particle positions at t and t+dt",
        "body_input": sample["collider_input"],
        "force_body_input": manifest["collider_state_timing"],
        "force_parameters": CONTACT_SETTINGS
        if is_nerd
        else {
            "effective_mass_factor": 10.0,
            "tangent_mass_factor": 10.0,
            "stabilization": 0.01,
            "friction": float(sample["friction"][0]),
        },
        "surface_failures": [],
    }
    rows = []
    for frame in range(1, len(predicted)):
        rv = (reference[frame] - reference[frame - 1]) / dt
        if record["shape"] == "freefall":
            near_rmse = None
            near_cnt = 0
        else:
            mask = (
                np.abs(body_distance(reference[frame], body[frame], record["shape"], manifest["body_size_m"])) <= 0.05
            )
            near_rmse = particle_rmse(reference[frame], predicted[frame], mask)
            near_cnt = int(mask.sum())
        moving = np.linalg.norm(rv, axis=1) > 0.01
        rows.append(
            {
                "time_s": frame * dt,
                "global_rmse_mm": particle_rmse(reference[frame], predicted[frame]),
                "near_contact_rmse_mm": near_rmse,
                "near_contact_count": near_cnt,
                "moving_rmse_mm": particle_rmse(reference[frame], predicted[frame], moving),
                "moving_count": int(moving.sum()),
                "frozen_rmse_mm": particle_rmse(reference[frame], reference[0]),
            }
        )
    lower = np.minimum(reference.min((0, 1)), predicted.min((0, 1))) - 0.18
    upper = np.maximum(reference.max((0, 1)), predicted.max((0, 1))) + 0.18
    status["grid_lower_m"], status["grid_upper_m"] = lower.tolist(), upper.tolist()
    fields = []
    maps = []
    try:
        grid = CommonGrid(lower, upper, 0.02)
        coarse = CommonGrid(lower, upper, 0.03)
        for frame in range(0, len(predicted), round(0.10 / dt)):
            rv = (reference[frame] - reference[max(0, frame - 1)]) / dt
            pv = (predicted[frame] - predicted[max(0, frame - 1)]) / dt
            item, a, b = grid.fields(reference[frame], predicted[frame], mass, rv, pv)
            item["time_s"] = frame * dt
            if record["shape"] == "freefall":
                for name in ("reference", "predicted"):
                    item[name + "_below_body_mass_kg"] = None
                    item[name + "_inside_body_mass_kg"] = None
            else:
                for name, x in (("reference", reference[frame]), ("predicted", predicted[frame])):
                    item[name + "_below_body_mass_kg"] = below_body_mass(
                        x, body[frame], record["shape"], manifest["body_size_m"], mass
                    )
                    inside = body_distance(x, body[frame], record["shape"], manifest["body_size_m"]) < 0
                    item[name + "_inside_body_mass_kg"] = float(mass[inside].sum())
            if frame % round(1.0 / dt) == 0:
                maps.append((frame * dt, (b - a).sum(axis=2), (b - a).sum(axis=1)))
            if frame % round(0.20 / dt) == 0:
                for reconstruction, suffix in ((grid, ""), (coarse, "_30mm_grid")):
                    try:
                        if suffix:
                            ca, _ = coarse.deposit(reference[frame], mass)
                            cb, _ = coarse.deposit(predicted[frame], mass)
                        else:
                            ca, cb = a, b
                        values = surface_distances(
                            reconstruction.surface(ca), reconstruction.surface(cb), seed=42 + frame
                        )
                        item.update({key + suffix: value for key, value in values.items()})
                    except (ValueError, RuntimeError) as error:
                        status["surface_failures"].append(
                            {"time_s": frame * dt, "grid_spacing_m": reconstruction.spacing, "reason": str(error)}
                        )
            fields.append(item)
    except ValueError as error:
        status["field_failure"] = str(error)
    if maps:
        np.savez_compressed(
            output / "mass_difference_maps.npz",
            time_s=[item[0] for item in maps],
            xy_kg=np.stack([item[1] for item in maps]),
            xz_kg=np.stack([item[2] for item in maps]),
            grid_lower_m=grid.lower,
            grid_spacing_m=grid.spacing,
        )
    status["mass_difference_map_definition"] = (
        "Particle NeRD minus MPM physical mass, summed along z for the xy map and y for the xz map; "
        "kg per projected grid column, sampled every 1 s"
    )
    forces = {}
    if native_force is not None and len(predicted) > 1 and record.get("mode") == "dynamic":
        native_force = native_force[: len(predicted) - 1]
        forces["reference"] = native_force
        for name, positions in (("reference_particles", reference), ("predicted_particles", predicted)):
            estimator = (NeRDRecordedForceEstimator if is_nerd else RecordedForceEstimator)(
                mass,
                body_mass,
                body_inertia,
                manifest["body_size_m"],
                record["shape"],
                dt,
                float(sample["friction"][0]),
                device=device,
            )
            selector = (
                ObjectParticleStepper(sample["kinds"], sample["sizes"], sample["friction"], sample["material"], dt)
                if is_nerd
                else None
            )
            values, diagnostics = [], []
            for frame in range(len(native_force)):
                selection = {}
                if selector is not None:
                    hist = (
                        torch.as_tensor(positions[[max(0, frame - 2), max(0, frame - 1), frame]], device=device)
                        .transpose(0, 1)
                        .contiguous()
                    )
                    with torch.no_grad():
                        prepared = selector.prepare(hist, sample["poses"][frame], sample["velocities"][frame])
                    selection["active_idx"] = prepared.selection.topk
                    selector.phase = (selector.phase + 1) % OBJECT_PIPELINE["selection_interval"]
                force, stats = estimator.estimate(
                    positions[frame], (positions[frame + 1] - positions[frame]) / dt, collider[frame], **selection
                )
                values.append(force)
                diagnostics.append(stats)
            forces[name] = np.asarray(values)
            status["force_" + name] = force_errors(native_force, forces[name])
            status["force_" + name]["unconverged_steps"] = int(
                np.sum(np.asarray(diagnostics)[:, 2] > estimator.estimator.tolerance)
            )
    with h5py.File(output / "prediction.h5", "w") as handle:
        handle.create_dataset("particle_position", data=predicted, chunks=(1, len(mass), 3), compression="lzf")
        handle.create_dataset("body_state", data=body[: len(predicted)])
        for name, values in forces.items():
            handle.create_dataset("force_" + name, data=values)
    for name, data in (("particle_metrics", rows), ("field_metrics", fields)):
        keys = list(dict.fromkeys(key for item in data for key in item))
        with (output / (name + ".csv")).open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=keys)
            writer.writeheader()
            writer.writerows(data)
    status["mean_position_rmse_mm"] = float(np.mean([r["global_rmse_mm"] for r in rows])) if rows else None
    status["final_position_rmse_mm"] = rows[-1]["global_rmse_mm"] if rows else None
    (output / "evaluation.json").write_text(json.dumps(status, indent=2, allow_nan=False))
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    device = f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}"
    torch.cuda.set_device(device)
    wp.init()
    wp.set_device(device)
    torch.set_num_threads(4)
    saved = torch.load(args.checkpoint, map_location=device, weights_only=False)
    arch = saved.get("architecture") or saved["config"].get("architecture", "particle_transolver_v1")
    is_world_like = arch in ("world_particle_v1", "particle_nerd_local_v1")
    if not is_world_like:
        validate_object_checkpoint(saved)
    if arch == "particle_nerd_local_v1":
        model_cls = ParticleNeRDLocalNetwork
        m_cfg = dict(saved["model_config"])
        if "spatial_slice_assign" not in m_cfg:
            m_cfg["spatial_slice_assign"] = False
            m_cfg["use_rms_swiglu"] = False
        model = model_cls(**m_cfg).to(device).eval()
    elif arch == "world_particle_v1":
        model_cls = WorldParticleNetwork
        model = model_cls(**saved["model_config"]).to(device).eval()
    else:
        model_cls = ParticleTransolver
        model = model_cls(**saved["model_config"]).to(device).eval()
    dataset = Episodes(args.data, "test", collider_input="material_step" if is_world_like else "pre_rigid_step")
    model.load_state_dict(saved["model"])
    sha = file_hash(args.checkpoint)
    summaries = []
    indices = list(range(len(dataset.records)))[rank::world]
    for index in indices[: args.limit]:
        record = dataset.records[index][0]
        summary = evaluate_one(model, dataset, index, args.output / record["id"], sha, device)
        summaries.append(summary)
        completed_means = [
            s["mean_position_rmse_mm"]
            for s in summaries
            if s["failure"] is None and s["mean_position_rmse_mm"] is not None
        ]
        available_means = [s["mean_position_rmse_mm"] for s in summaries if s["mean_position_rmse_mm"] is not None]
        summary_file = "summary.json" if world == 1 else f"summary_rank{rank}.json"
        (args.output / summary_file).write_text(
            json.dumps(
                {
                    "trajectories": summaries,
                    "rank": rank,
                    "world_size": world,
                    "failed_trajectories": sum(s["failure"] is not None for s in summaries),
                    "aggregation": (
                        "Per-trajectory means have equal weight; partial failures retain their completed duration"
                    ),
                    "complete_trajectory_count": len(completed_means),
                    "complete_trajectory_mean_position_rmse_mm": (
                        float(np.mean(completed_means)) if completed_means else None
                    ),
                    "available_duration_mean_position_rmse_mm": (
                        float(np.mean(available_means)) if available_means else None
                    ),
                },
                indent=2,
                allow_nan=False,
            )
        )
        print(
            json.dumps(
                {
                    "episode": record["id"],
                    "failure": summary["failure"],
                    "final_position_rmse_mm": summary["final_position_rmse_mm"],
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
