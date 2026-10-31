# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compare reference endpoint, reference displacement, and predicted displacement inputs.

Recorded rigid states and known external forces are identical in all cases.
This is offline force reconstruction, not a dynamic simulation or a training run.
"""

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import warp as wp

from particle_nerd.force_estimation.estimator import ContactEstimator

LABELS = ("mpm_endpoint", "mpm_difference", "transolver_difference")


def error_metrics(estimate, reference, mask):
    delta = estimate[mask] - reference[mask]
    return dict(
        samples=int(mask.sum()),
        force_vector_rmse_N=float(np.sqrt(np.mean(np.sum(delta[:, :3] ** 2, axis=1)))),
        torque_vector_rmse_Nm=float(np.sqrt(np.mean(np.sum(delta[:, 3:] ** 2, axis=1)))),
        mean_fz_N=float(estimate[mask, 2].mean()),
        reference_mean_fz_N=float(reference[mask, 2].mean()),
        fz_bias_N=float(delta[:, 2].mean()),
        peak_fz_N=float(estimate[mask, 2].max()),
        reference_peak_fz_N=float(reference[mask, 2].max()),
        impulse_z_Ns=float(estimate[mask, 2].sum() * 0.02),
        reference_impulse_z_Ns=float(reference[mask, 2].sum() * 0.02),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", choices=("wide-circle", "figure-eight", "vertical-drop"), required=True)
    parser.add_argument("--effective-mass-scale", type=float, default=10.0)
    parser.add_argument("--fixed-material-velocity", action="store_true")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional particle_contact_transolver_v1 checkpoint for neural contact evaluation",
    )
    args = parser.parse_args()
    with h5py.File(args.reference) as ref, h5py.File(args.prediction) as pred:
        manifest = json.loads(ref.attrs["manifest"])
        start, valid = int(pred.attrs["start_frame"]), int(pred.attrs["valid_frames"])
        assert valid == 251, "Expected a complete 5-second rollout"
        frames = ref["frame_steps"][start : start + valid]
        np.testing.assert_array_equal(pred["frame_steps"][:valid], frames)
        pose_all, velocity_all = ref["body_pose"][:], ref["body_velocity"][:]
        # Verify the saved predictions were conditioned on exactly this recorded box motion.
        np.testing.assert_array_equal(pred["body_pose"][frames], pose_all[frames])
        x = ref["particle_position"][start : start + valid]
        v = ref["particle_velocity"][start : start + valid]
        x_hist_ref = ref["particle_position"][start - 2 : start + valid]
        predicted_x = pred["particle_position"][:valid]
        np.testing.assert_array_equal(predicted_x[0], x[0])
        masses = ref["particle_mass"][:]
        radius = float(ref["particle_radius"][0])
        external_all = ref["external_wrench"][:]
        raw_all = ref["material_on_body_wrench"][:]
        applied_all = ref["applied_feedback_wrench"][:]
    dt = 1.0 / manifest["record_hz"]
    assert np.isclose(dt, 0.02) and np.all(np.diff(frames) == 4)
    size = np.asarray(manifest["body_size_m"])
    mass = float(manifest["body_mass_kg"])
    inertia = (
        np.asarray(manifest["body_inertia"])
        if "body_inertia" in manifest
        else np.diag(mass / 12 * (np.sum(size**2) - size**2))
    )
    wp.init()
    device = "cuda:0"
    neural_predictor = None
    inertia_tensor = None
    if args.checkpoint is not None:
        import torch

        from particle_nerd.transolver.features import material_features
        from particle_nerd.transolver.inference import Predictor, manifest_colliders

        saved = torch.load(args.checkpoint, map_location=device, weights_only=False)
        init_hist = np.stack([x_hist_ref[0], x_hist_ref[1], x_hist_ref[2]], axis=1)
        neural_predictor = Predictor(
            saved,
            torch.as_tensor(init_hist, dtype=torch.float32, device=device),
            torch.as_tensor(material_features(manifest, float(masses[0]), radius), dtype=torch.float32, device=device),
            *manifest_colliders(manifest, device),
            use_cuda_graph=False,
        )
        inertia_tensor = torch.as_tensor(inertia, dtype=torch.float32, device=device)
    estimator = ContactEstimator(
        masses,
        mass,
        inertia,
        size,
        dt,
        friction=manifest.get("body_contact_friction", 0.6),
        shape=manifest.get("geometry", "box"),
        effective_mass_scale=args.effective_mass_scale,
        fixed_material_velocity=args.fixed_material_velocity,
    )
    position = wp.empty(len(masses), dtype=wp.vec3, device=device)
    next_velocity = wp.empty(len(masses), dtype=wp.vec3, device=device)
    pose = wp.empty(1, dtype=wp.transform, device=device)
    body_velocity = wp.empty(1, dtype=wp.spatial_vector, device=device)
    external = wp.empty(1, dtype=wp.spatial_vector, device=device)
    forces = np.zeros((3, valid - 1, 6))
    stats = np.zeros((3, valid - 1, 3))
    reference = np.empty((valid - 1, 6))
    applied_reference = np.empty_like(reference)
    for i in range(valid - 1):
        step, end = int(frames[i]), int(frames[i + 1])
        pose.assign(pose_all[step : step + 1])
        body_velocity.assign(velocity_all[step : step + 1])
        external.assign(external_all[step : step + 1])
        # Raw MPM torque was recorded about each substep's end COM. Express all
        # impulses about the estimator's interval-start COM before averaging.
        raw = raw_all[step:end].astype(np.float64).copy()
        raw[:, 3:] += np.cross(pose_all[step + 1 : end + 1, :3] - pose_all[step, :3], raw[:, :3])
        reference[i] = raw.mean(0)
        applied_reference[i] = applied_all[step:end].mean(0)
        variants = (
            (x[i], v[i + 1]),
            (x[i], (x[i + 1] - x[i]) / dt),
            (predicted_x[i], (predicted_x[i + 1] - predicted_x[i]) / dt),
        )
        if neural_predictor is not None:
            import torch

            hist_ref_i = torch.as_tensor(
                np.stack([x_hist_ref[i], x_hist_ref[i + 1], x_hist_ref[i + 2]], axis=1),
                dtype=torch.float32,
                device=device,
            )
            pred_frames = [
                x_hist_ref[0] if m == -2 else (x_hist_ref[1] if m == -1 else predicted_x[m]) for m in (i - 2, i - 1, i)
            ]
            hist_pred_i = torch.as_tensor(np.stack(pred_frames, axis=1), dtype=torch.float32, device=device)
            pose_t = torch.as_tensor(pose_all[step], dtype=torch.float32, device=device)
            vel_t = torch.as_tensor(velocity_all[step], dtype=torch.float32, device=device)
            ext_t = torch.as_tensor(external_all[step], dtype=torch.float32, device=device)
            neural_variants = (
                (hist_ref_i, torch.as_tensor(v[i + 1], dtype=torch.float32, device=device)),
                (hist_ref_i, torch.as_tensor((x[i + 1] - x[i]) / dt, dtype=torch.float32, device=device)),
                (
                    hist_pred_i,
                    torch.as_tensor((predicted_x[i + 1] - predicted_x[i]) / dt, dtype=torch.float32, device=device),
                ),
            )
            for k, (hist_t, next_v_t) in enumerate(neural_variants):
                out = neural_predictor.estimate_contact(
                    pose_t,
                    vel_t,
                    ext_t,
                    mass,
                    inertia_tensor,
                    history=hist_t,
                    next_velocity=next_v_t,
                )
                w = out["wrench"].detach().cpu().numpy()
                assert np.isfinite(w).all(), (args.case, i, k)
                forces[k, i] = w
                stats[k, i] = [float(out["contact_weight"].item()), 0.0, 0.0]
        else:
            for k, (points, velocities) in enumerate(variants):
                position.assign(points)
                next_velocity.assign(velocities)
                estimator.estimate(position, next_velocity, pose, body_velocity, external)
                diagnostic = estimator.diagnostics()
                assert diagnostic["finite"], (args.case, i, k)
                forces[k, i] = diagnostic["wrench"]
                stats[k, i] = [diagnostic["contacts"], diagnostic["iterations"], diagnostic["residual"]]
        if i % 50 == 0:
            print(
                json.dumps(dict(case=args.case, step=i, reference_fz=reference[i, 2], fz=forces[:, i, 2].tolist())),
                flush=True,
            )
    t = np.arange(valid - 1) * dt
    masks = {"all": np.ones(len(t), dtype=bool)}
    if args.case == "vertical-drop":
        masks.update(pre_contact=t < 0.28, impact=(t >= 0.2) & (t < 0.7), later_support=t >= 1.0)
    else:
        masks.update(pre_drag_support=t < 0.24, dragging=(t >= 0.26) & (t < 4.74))
    rows = {}
    for k, label in enumerate(LABELS):
        rows[label] = dict(
            phases={name: error_metrics(forces[k], reference, mask) for name, mask in masks.items()},
            converged_fraction=float(np.mean(stats[k, :, 2] <= estimator.tolerance)),
            median_iterations=float(np.median(stats[k, :, 1])),
            max_iterations=int(stats[k, :, 1].max()),
            force_rmse_vs_applied_feedback_N=float(
                np.sqrt(np.mean(np.sum((forces[k, :, :3] - applied_reference[:, :3]) ** 2, axis=1)))
            ),
            first_fz_above_1N_s=float(t[np.flatnonzero(forces[k, :, 2] > 1)[0]])
            if np.any(forces[k, :, 2] > 1)
            else None,
        )
    summary = dict(
        case=args.case,
        dt_s=dt,
        duration_s=5.0,
        frames=250,
        box_mass_kg=mass,
        variants=rows,
        masks={key: [int(i) for i in np.flatnonzero(mask)] for key, mask in masks.items()},
        first_reference_fz_above_1N_s=float(t[np.flatnonzero(reference[:, 2] > 1)[0]]),
        reference=str(args.reference),
        prediction=str(args.prediction),
        prediction_sha256=hashlib.sha256(args.prediction.read_bytes()).hexdigest(),
        settings=dict(
            effective_mass_scale=None if estimator.fixed_material_velocity else estimator.scale,
            material_response="fixed_velocity" if estimator.fixed_material_velocity else "effective_mass",
            stabilization=estimator.stabilization,
            gradient_radius_m=estimator.radius,
            friction=estimator.friction,
            max_iterations=estimator.max_iterations,
            tolerance=estimator.tolerance,
        ),
        reference_definition="Raw MPM material reaction averaged over 20 ms; torque about interval-start box COM",
        caveat="MPM endpoint inputs still test the approximate 20 ms estimator, not exact inverse dynamics.",
    )
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output / "forces.npz",
        time=t,
        reference=reference,
        applied_reference=applied_reference,
        estimate=forces,
        stats=stats,
        body_pose=pose_all[frames],
        body_velocity=velocity_all[frames],
        external_start=external_all[frames[:-1]],
    )
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
