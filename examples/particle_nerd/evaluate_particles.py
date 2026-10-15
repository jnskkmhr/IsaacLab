# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Measure learned particle rollouts and analytical-force rigid-body coupling.

The conditional diagnostic supplies reference rigid states while particles run
freely. The coupled diagnostic supplies only initial states: learned particles,
analytical contact forces, and lagged rigid integration advance together.
Neither diagnostic invokes MPM. Reference states/forces are evaluation targets.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from body_contact_reconstruction import reconstruct_body_contact
from geometry import SampledSDF, body_contacts
from learning_data import EpisodeStore
from particle_model import ParticleGraph, graph_inputs
from replay_body_motion import integrate_body
from scipy.spatial.transform import Rotation
from train_particles import one_step_metrics


def rigid_clearance(episode, pose):
    body = episode["moving_body"]
    matrix = Rotation.from_quat(pose[3:]).as_matrix()
    half = episode["body_size"][body] / 2
    shape = episode["body_shapes"][body]
    if shape == "sphere":
        extent = np.full(3, half[0])
    elif shape == "cylinder":
        extent = half[0] * np.linalg.norm(matrix[:, :2], axis=1) + half[2] * abs(matrix[:, 2])
    else:
        extent = abs(matrix) @ half
    fixed = episode["body_pose"][0]
    sizes = episode["body_size"]
    ground_top = fixed[0, 2] + sizes[0, 2] / 2
    lower = np.array([fixed[i, a] + sizes[i, a] / 2 for i, a in ((1, 0), (3, 1))])
    upper = np.array([fixed[i, a] - sizes[i, a] / 2 for i, a in ((2, 0), (4, 1))])
    return min(
        float(pose[2] - extent[2] - ground_top),
        float(np.min(pose[:2] - extent[:2] - lower)),
        float(np.min(upper - pose[:2] - extent[:2])),
    )


def project_particles(position, previous, body_pose, episode, mode="all", radius=0.0):
    """Optional geometric nonpenetration ablation, applied to every particle."""
    position = position.copy()
    for i, pose in enumerate(body_pose):
        if mode == "static" and episode["body_mass"][i] > 0:
            continue
        distance, normal, _ = body_contacts(position, pose, episode["body_size"][i], episode["body_shapes"][i])
        correction = np.maximum(radius - distance, 0)
        position += correction[:, None] * normal
    return position, (position - previous) / episode["dt"]


def rollout(
    model,
    episode,
    device,
    mode="coupled",
    steps=300,
    projection="none",
    sampled_sdf=False,
    particle_baseline=False,
    rigid_solver="free",
):
    e = episode
    body = e["moving_body"]
    dt = e["dt"]
    x = e["particle_position"][0].copy()
    v = e["particle_velocity"][0].copy()
    history = [v.copy() for _ in range(model.config["history_length"])]
    poses = e["body_pose"][0].copy().astype(float)
    velocities = e["body_velocity"][0].copy().astype(float)
    rigid = None
    if mode == "coupled" and rigid_solver == "newton":
        from newton_rigid import NewtonRigidStepper

        rigid = NewtonRigidStepper(e)
    direct_forces = []
    previous_force = np.zeros(6)
    particle_states, particle_velocities = [x.copy()], [v.copy()]
    body_states, body_velocities = [poses.copy()], [velocities.copy()]
    forces, failures, elapsed = [], 0, []
    sdf = (
        SampledSDF.from_primitive(e["body_shapes"][body], e["body_size"][body], spacing=0.005, padding=0.7)
        if sampled_sdf
        else None
    )
    stop_reason = "completed"
    max_steps = min(steps, len(e["particle_position"]) - 1)
    with torch.no_grad():
        for t in range(max_steps):
            start = time.monotonic()
            old_pose, old_velocity = poses[body].copy(), velocities[body].copy()
            if mode == "conditional":
                poses = e["body_pose"][t + 1].copy()
                velocities = e["body_velocity"][t + 1].copy()
            elif rigid is not None:
                wrench = np.zeros((len(poses), 6))
                wrench[body] = previous_force
                poses, velocities = rigid.step(wrench)
            else:
                poses[body], velocities[body] = integrate_body(
                    old_pose,
                    old_velocity,
                    previous_force,
                    e["body_mass"][body],
                    e["body_inertia"][body],
                    e["gravity"],
                    dt,
                )
            if particle_baseline:
                next_x, next_v = x.copy(), np.zeros_like(v)
            else:
                inputs = graph_inputs(
                    x, np.concatenate(history, axis=1), e, poses, velocities, device, model.config["radius"], sdf=sdf
                )
                next_x, next_v = model(*inputs)
                next_x, next_v = next_x.cpu().numpy(), next_v.cpu().numpy()
            if projection != "none":
                next_x, next_v = project_particles(next_x, x, poses, e, mode=projection)
            if not np.isfinite(next_x).all() or np.max(np.linalg.norm(next_v, axis=1)) > 20:
                stop_reason = "particle velocity exceeded 20 m/s or became nonfinite"
                break
            result = reconstruct_body_contact(
                x,
                next_v,
                e["particle_mass"],
                old_pose,
                old_velocity,
                e["body_mass"][body],
                e["body_inertia"][body],
                e["body_size"][body],
                dt=dt,
                gravity=e["gravity"],
                body_shape=e["body_shapes"][body],
                body_sdf=sdf,
                contact_offset=0,
                effective_mass_scale=10,
                friction=float(e["body_friction"][body]),
                extrapolate_velocity=True,
                velocity_gradient_device="cuda:0" if device == "cuda" else None,
                stabilization=0.01,
                max_iterations=2000,
            )
            previous_force = result.body_wrench
            failures += not result.converged
            forces.append(previous_force.copy())
            direct_forces.append(rigid.direct_contact_force[body].copy() if rigid is not None else np.zeros(3))
            x, v = next_x, next_v
            history = history[1:] + [v.copy()]
            particle_states.append(x.copy())
            particle_velocities.append(v.copy())
            body_states.append(poses.copy())
            body_velocities.append(velocities.copy())
            elapsed.append(time.monotonic() - start)
            if mode == "coupled" and rigid is None and rigid_clearance(e, poses[body]) < 0:
                stop_reason = "predicted body intersects rigid tray; rigid-tray contact is outside this diagnostic"
                break
    x_all, v_all, pose_all = np.asarray(particle_states), np.asarray(particle_velocities), np.asarray(body_states)
    count = len(forces)
    reference_x = e["particle_position"][: count + 1]
    reference_pose = e["body_pose"][: count + 1, body]
    position_error = np.linalg.norm(x_all - reference_x, axis=-1)
    force_error = np.asarray(forces).reshape(-1, 6) - e["force_reference"][:count, body]
    angles = (Rotation.from_quat(pose_all[:, body, 3:]) * Rotation.from_quat(reference_pose[:, 3:]).inv()).magnitude()
    body_error = np.linalg.norm(pose_all[:, body, :3] - reference_pose[:, :3], axis=-1)
    peak_step = int(np.argmax(e["force_reference"][:count, body, 2])) if count else None
    metrics = dict(
        episode=e["name"],
        mode=mode,
        projection=projection,
        sampled_sdf=sampled_sdf,
        stationary_particle_baseline=particle_baseline,
        rigid_solver=rigid_solver,
        direct_rigid_contact_steps=int(np.sum(np.linalg.norm(direct_forces, axis=-1) > 0.1)) if direct_forces else 0,
        body_shape=e["body_shapes"][body],
        body_mass_kg=float(e["body_mass"][body]),
        material=e["material"],
        steps=count,
        requested_steps=max_steps,
        complete=count == max_steps,
        stop_reason=stop_reason,
        particle_position_rmse_m=float(np.sqrt(np.mean(position_error**2))),
        final_particle_position_rmse_m=float(np.sqrt(np.mean(position_error[-1] ** 2))),
        body_position_rmse_m=float(np.sqrt(np.mean(body_error**2))),
        body_orientation_rmse_deg=float(np.sqrt(np.mean(angles**2)) * 180 / np.pi),
        force_vector_rmse_N=float(np.sqrt(np.mean(np.sum(force_error[:, :3] ** 2, axis=-1)))) if count else None,
        reference_peak_vertical_force_N=float(e["force_reference"][peak_step, body, 2]) if count else None,
        predicted_peak_vertical_force_N=float(np.asarray(forces)[:, 2].max()) if count else None,
        force_at_reference_peak_error_N=float(force_error[peak_step, 2]) if count else None,
        reference_peak_time_s=(peak_step + 0.5) * dt if count else None,
        torque_vector_rmse_Nm=float(np.sqrt(np.mean(np.sum(force_error[:, 3:] ** 2, axis=-1)))) if count else None,
        force_solver_failures=failures,
        seconds_per_step=float(np.mean(elapsed)) if elapsed else None,
        reference_has_rigid_tray_contact=bool(np.max(abs(e["rigid_contact_impulse"][:max_steps])) > 1e-7),
    )
    arrays = dict(
        particle_position=x_all,
        particle_velocity=v_all,
        body_pose=pose_all,
        body_velocity=np.asarray(body_velocities),
        forces=np.asarray(forces),
        reference_particle_position=reference_x,
        reference_body_pose=e["body_pose"][: count + 1],
        reference_forces=e["force_reference"][:count, body],
        direct_rigid_contact_force=np.asarray(direct_forces),
        dt=dt,
    )
    return metrics, arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--projection", choices=("none", "static", "all"), default="none", nargs="?", const="all")
    parser.add_argument("--sampled-sdf", action="store_true")
    parser.add_argument("--rigid-solver", choices=("free", "newton"), default="free")
    parser.add_argument(
        "--stationary-particles",
        action="store_true",
        help="Diagnostic baseline: hold every material particle at its initial position.",
    )
    parser.add_argument("--modes", nargs="+", choices=("conditional", "coupled"), default=["conditional", "coupled"])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    saved = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = ParticleGraph(**saved["config"]).to(device)
    model.load_state_dict(saved["model"])
    model.eval()
    store = EpisodeStore(args.data, args.split, cache_size=16)
    output = dict(
        checkpoint=str(args.checkpoint),
        checkpoint_step=saved["step"],
        split=args.split,
        expected_episodes=min(args.limit or len(store), len(store)),
        evaluation_complete=False,
        one_step=None if args.stationary_particles else one_step_metrics(model, store, device),
        one_step_scope="Raw network with analytic geometry, before optional geometric projection.",
        particle_predictor="stationary" if args.stationary_particles else "gnn",
        rollouts=[],
        force_parameters=dict(effective_mass_scale=10, stabilization=0.01, contact_offset=0, affine=True),
        limitation=(
            "Newton MJWarp integrates rigid bodies and direct rigid contacts; learned particles remain external."
            if args.rigid_solver == "newton"
            else "Rigid integration is a free-body diagnostic; direct rigid-tray contacts stop coupled rollouts."
        ),
    )
    model.eval()
    for i in range(min(args.limit or len(store), len(store))):
        for mode in args.modes:
            metrics, arrays = rollout(
                model,
                store[i],
                device,
                mode,
                args.steps,
                args.projection,
                args.sampled_sdf,
                args.stationary_particles,
                args.rigid_solver,
            )
            output["rollouts"].append(metrics)
            output["evaluation_complete"] = i + 1 == output["expected_episodes"] and mode == args.modes[-1]
            np.savez_compressed(args.output / f"episode_{i}_{mode}.npz", **arrays)
            (args.output / "metrics.json").write_text(json.dumps(output, indent=2) + "\n")
            print(json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
