# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Dynamic Transolver + analytical contact + MJWarp under a shared external drag target.

Only initialization reads reference particle/body states. Future reference states
are used after simulation for comparison; the controller reads only its own box.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import warp as wp

from particle_nerd.data_collection.trajectories import drag_target_offset
from particle_nerd.transolver.features import material_features
from particle_nerd.runner.simulation import CoupledSimulation
from particle_nerd.transolver.inference import Predictor, manifest_colliders


@wp.kernel
def drag_force(
    pose: wp.array(dtype=wp.transform),
    velocity: wp.array(dtype=wp.spatial_vector),
    targets: wp.array(dtype=wp.vec3),
    envelopes: wp.array(dtype=float),
    index: int,
    mass: float,
    output: wp.array(dtype=wp.spatial_vector),
    recording: wp.array(dtype=wp.spatial_vector),
):
    x = wp.transform_get_translation(pose[0])
    v = wp.spatial_top(velocity[0])
    f = mass * (120.0 * (targets[index] - x) - 15.0 * v)
    cap = 2.5 * mass * 9.81
    f = f * (cap / wp.max(cap, wp.length(f))) * envelopes[index]
    wrench = wp.spatial_vector(f, wp.vec3(0.0))
    output[0] = wrench
    recording[index] = wrench


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--path", choices=("wide-circle", "figure-eight"), required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, default=250)
    p.add_argument("--contact-relaxation", type=float, default=0.1)
    p.add_argument("--contact-solver", choices=("jacobi", "body_velocity", "rft_body_velocity"), default="jacobi")
    p.add_argument("--contact-warm-start", action="store_true")
    p.add_argument("--contact-iterations", type=int, default=2000)
    p.add_argument("--effective-mass-scale", type=float, default=10.0)
    p.add_argument(
        "--fixed-material-velocity",
        action="store_true",
        help="Hold supplied material velocity fixed during contact iterations",
    )
    p.add_argument(
        "--active-k",
        type=lambda v: "auto" if v.lower() in ("auto", "heuristic") else int(v),
        default=None,
        help="Active particle selection ('auto' for proximity+material heuristic, or integer cap)",
    )
    p.add_argument(
        "--force-model",
        choices=("auto", "analytical", "neural_contact"),
        default="auto",
        help="Select physics-structured neural contact head or analytical contact estimator",
    )
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    saved = torch.load(args.checkpoint, map_location="cuda:0", weights_only=False)
    with h5py.File(args.reference) as f:
        m = json.loads(f.attrs["manifest"])
        frames = f["frame_steps"][:]
        start = int(np.searchsorted(frames, m["settled_step"]))
        first = int(frames[start])
        initial_pose = f["body_pose"][first]
        initial_velocity = f["body_velocity"][first]
        mass = f["particle_mass"][:]
        radius = float(f["particle_radius"][0])
        history = np.stack([f["particle_position"][i] for i in range(start - 2, start + 1)], axis=1)
    dt = float(saved["config"]["dt"])
    rigid_dt = float(m["dt"])
    substeps = round(dt / rigid_dt)
    assert np.isclose(dt, 0.02) and np.isclose(rigid_dt, 0.005) and substeps == 4
    count = args.steps * substeps
    times = np.arange(count) * rigid_dt
    targets = np.repeat(initial_pose[None, :3], count, axis=0)
    envelopes = np.zeros(count, dtype=np.float32)
    for i, t in enumerate(times):
        if 0.25 <= t < 4.75:
            targets[i] += drag_target_offset(args.path, t - 0.25, 4.5)
            envelopes[i] = min(1, (t - 0.25) / 0.2, (4.75 - t) / 0.2)
    # Verify the exogenous targets match the existing reference controller.
    with h5py.File(args.reference) as f:
        recorded_targets = f["drag_target_position"][first : first + count]
    target_error = float(np.abs(targets[envelopes > 0] - recorded_targets[envelopes > 0]).max())
    assert target_error < 1e-5, target_error
    predictor = Predictor(
        saved,
        torch.as_tensor(history, device="cuda:0"),
        torch.as_tensor(material_features(m, float(mass[0]), radius), device="cuda:0"),
        *manifest_colliders(m, "cuda:0"),
        active_k=args.active_k,
    )
    sim = CoupledSimulation(
        predictor,
        m,
        mass,
        initial_pose,
        initial_velocity,
        rigid_dt=rigid_dt,
        max_iterations=args.contact_iterations,
        jacobi_relaxation=args.contact_relaxation,
        contact_warm_start=args.contact_warm_start,
        contact_solver=args.contact_solver,
        effective_mass_scale=args.effective_mass_scale,
        fixed_material_velocity=args.fixed_material_velocity,
        force_model=args.force_model,
    )
    target_gpu = wp.array(targets, dtype=wp.vec3, device=sim.device)
    envelope_gpu = wp.array(envelopes, dtype=float, device=sim.device)
    applied = wp.zeros(count, dtype=wp.spatial_vector, device=sim.device)
    wp.synchronize_device(sim.device)
    body_poses = [initial_pose.copy()]
    body_velocities = [initial_velocity.copy()]
    wrenches, stats, step_wall_ms = [], [], []
    failure = None
    with h5py.File(args.output / "rollout.hdf5", "w") as out:
        positions = out.create_dataset(
            "particle_position", (args.steps + 1, len(mass), 3), dtype="f4", compression="lzf"
        )
        positions[0] = history[:, -1]
        for step in range(args.steps):

            def controller(state, substep, wrench):
                wp.launch(
                    drag_force,
                    1,
                    [
                        state.body_q,
                        state.body_qd,
                        target_gpu,
                        envelope_gpu,
                        step * substeps + substep,
                        float(m["body_mass_kg"]),
                        wrench,
                        applied,
                    ],
                    device=sim.device,
                )

            wp.synchronize_stream(sim.stream)
            step_started = time.perf_counter()
            sim.step(substep_controller=controller)
            wp.synchronize_stream(sim.stream)
            step_wall_ms.append((time.perf_counter() - step_started) * 1000)
            x = predictor.x.cpu().numpy()
            pose = sim.rigid.state.body_q.numpy()[0]
            velocity = sim.rigid.state.body_qd.numpy()[0]
            contact = sim.estimator.diagnostics()
            if not (
                np.isfinite(x).all() and np.isfinite(pose).all() and np.isfinite(velocity).all() and contact["finite"]
            ):
                failure = dict(
                    step=step + 1,
                    reason="Nonfinite particle, rigid state, or material wrench",
                    particles_finite=bool(np.isfinite(x).all()),
                    body_pose_finite=bool(np.isfinite(pose).all()),
                    body_velocity_finite=bool(np.isfinite(velocity).all()),
                    wrench_finite=contact["finite"],
                )
                np.savez(
                    args.output / "failure.npz",
                    particles=x,
                    body_pose=pose,
                    body_velocity=velocity,
                    wrench=sim.estimator.wrench.numpy(),
                    predicted_velocity=predictor.v.cpu().numpy(),
                    previous_particles=sim.old_position.numpy(),
                    previous_body_pose=body_poses[-1],
                    previous_body_velocity=body_velocities[-1],
                    contact_stats=sim.estimator.stats.numpy(),
                )
                print(json.dumps(failure), flush=True)
                break
            positions[step + 1] = x
            body_poses.append(pose)
            body_velocities.append(velocity)
            wrenches.append(contact["wrench"])
            stats.append([contact["contacts"], contact["iterations"], contact["residual"]])
            if step % 25 == 0:
                print(json.dumps(dict(step=step + 1, body_xyz=pose[:3].tolist(), contact=contact)), flush=True)
        valid = len(body_poses)
        # Match reference's absolute frame indexing for the shared RTX renderer.
        poses_full = np.repeat(initial_pose[None], first + (valid - 1) * substeps + 1, axis=0)
        poses_full[first::substeps] = body_poses
        out.create_dataset("body_pose", data=poses_full)
        out.create_dataset("sampled_body_pose", data=body_poses)
        out.create_dataset("body_velocity", data=body_velocities)
        out.create_dataset("material_wrench", data=wrenches)
        out.create_dataset("contact_stats", data=stats)
        out.create_dataset("external_wrench", data=applied.numpy()[: (valid - 1) * substeps])
        out.create_dataset("drag_target_position", data=targets[: (valid - 1) * substeps])
        out.create_dataset("frame_steps", data=frames[start : start + valid])
        out.attrs.update(
            start_frame=start,
            valid_frames=valid,
            manifest=json.dumps(m),
            evaluation="Closed-loop Transolver + analytical contact + MJWarp; independently dynamic box",
        )
    steps = valid - 1
    with h5py.File(args.reference) as f:
        ref_pose = f["body_pose"][frames[start : start + valid]]
        # The coupled rigid solver receives applied_feedback_wrench, not the raw current material reaction.
        ref_force = f["applied_feedback_wrench"][first : first + steps * substeps].reshape(steps, substeps, 6).mean(1)
        ref_external = f["external_wrench"][first : first + steps * substeps].reshape(steps, substeps, 6).mean(1)
        final_ref_particles = f["particle_position"][start + steps]
    actual = np.asarray(body_poses)
    force = np.asarray(wrenches)
    metric = dict(
        path=args.path,
        steps=steps,
        duration_s=steps * dt,
        failure=failure,
        target_max_difference_m=target_error,
        body_position_rmse_m=float(np.sqrt(np.mean(np.sum((actual[:, :3] - ref_pose[:, :3]) ** 2, axis=1)))),
        final_body_position_error_m=float(np.linalg.norm(actual[-1, :3] - ref_pose[-1, :3])),
        final_particle_position_rmse_m=float(np.sqrt(np.mean(np.sum((x - final_ref_particles) ** 2, axis=1)))),
        material_force_vector_rmse_N=float(np.sqrt(np.mean(np.sum((force[:, :3] - ref_force[:, :3]) ** 2, axis=1)))),
        estimator_iterations_median=float(np.median(np.asarray(stats)[:, 1])),
        estimator_iterations_max=float(np.max(np.asarray(stats)[:, 1])),
        estimator_converged_fraction=float(np.mean(np.asarray(stats)[:, 2] <= sim.estimator.tolerance)),
        estimated_force_z_min_N=float(force[:, 2].min()),
        estimated_force_z_max_N=float(force[:, 2].max()),
        median_step_wall_ms=float(np.median(step_wall_ms[10:] or step_wall_ms)),
        p95_step_wall_ms=float(np.percentile(step_wall_ms[10:] or step_wall_ms, 95)),
        timing_scope="Synchronized physics; excludes first 10 steps, rendering, diagnostics and HDF5 writes",
        checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        contact_relaxation=sim.estimator.jacobi_relaxation,
        contact_warm_start=sim.estimator.warm_start,
        contact_solver=sim.estimator.contact_solver,
        contact_iteration_limit=sim.estimator.max_iterations,
        effective_mass_scale=None if sim.estimator.fixed_material_velocity else sim.estimator.scale,
        material_response="fixed_velocity" if sim.estimator.fixed_material_velocity else "effective_mass",
        particle_dt_s=dt,
        rigid_dt_s=rigid_dt,
        reference=str(args.reference),
        force_caveat=(
            "Trajectory comparison: rigid-body and particle states diverge; "
            "wrench difference is not same-state force-estimator error."
        ),
    )
    np.savez(
        args.output / "comparison.npz",
        time=np.arange(valid) * dt,
        reference_pose=ref_pose,
        prediction_pose=actual,
        reference_wrench=ref_force,
        prediction_wrench=force,
        reference_external=ref_external,
        prediction_external=applied.numpy()[: steps * substeps].reshape(steps, substeps, 6).mean(1),
        targets=targets[: steps * substeps : substeps],
    )
    (args.output / "metrics.json").write_text(json.dumps(metric, indent=2) + "\n")
    print(json.dumps(metric), flush=True)


if __name__ == "__main__":
    main()
