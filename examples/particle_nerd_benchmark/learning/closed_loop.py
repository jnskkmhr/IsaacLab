# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run fully dynamic object/NeRD coupling from a matched MPM initial state.

Only the initial reference particle/body state and static scene parameters enter
simulation. No recorded future pose, velocity, force, or particle state is fed
back. MJWarp advances the object under gravity, rigid tray contacts and the last
estimated material wrench. NeRD uses the begin-of-step body pose and velocity. Force reconstruction uses
the end velocity with the last wrench rewound, matching G1 lagged coupling.
"""

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import time
from pathlib import Path

import h5py
import newton
import numpy as np
import torch
import warp as wp
from newton.solvers import SolverMuJoCo

from .data import Episodes
from .forces import NeRDRecordedForceEstimator, RecordedForceEstimator
from .network import (
    CONTACT_SETTINGS,
    OBJECT_PIPELINE,
    ObjectParticleStepper,
    ParticleNeRDLocalNetwork,
    ParticleTransolver,
    WorldParticleNetwork,
    features_particle_nerd_local,
    features_world_particle,
    nerd_source_hashes,
    validate_object_checkpoint,
)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def build_rigid(reference, device):
    """Use recorded physical inertia, collision shapes and the reference MJWarp settings."""
    builder = newton.ModelBuilder(gravity=wp.vec3(0.0, 0.0, -9.81))
    initial = reference["body_state"][0]
    body = builder.add_body(
        xform=wp.transform(initial[:3], initial[3:7]),
        mass=float(reference["body_mass"][0]),
        inertia=wp.mat33(*reference["body_inertia"][0].ravel()),
        lock_inertia=True,
        label="InteractingObject",
    )
    if not np.allclose(reference["body_com"][0], 0.0):
        raise ValueError("Contact reconstruction requires a body origin at the center of mass")
    for i, label in enumerate(reference["shape_labels"].asstr()[:]):
        moving = reference["shape_body"][i] >= 0
        if not moving and "/Rigid" not in label:
            continue
        cfg = newton.ModelBuilder.ShapeConfig(
            density=0.0,
            mu=float(reference["shape_material_mu"][i]),
            margin=float(reference["shape_margin"][i]),
            gap=0.0,
            ke=160000.0,
            kd=1100.0,
        )
        transform = reference["shape_transform"][i]
        shape_type = newton.GeoType(int(reference["shape_type"][i]))
        size = reference["shape_scale"][i]
        kwargs = dict(
            body=body if moving else -1, xform=wp.transform(transform[:3], transform[3:]), cfg=cfg, label=label
        )
        if shape_type == newton.GeoType.BOX:
            builder.add_shape_box(hx=float(size[0]), hy=float(size[1]), hz=float(size[2]), **kwargs)
        elif shape_type == newton.GeoType.SPHERE:
            builder.add_shape_sphere(radius=float(size[0]), **kwargs)
        elif shape_type == newton.GeoType.CYLINDER:
            builder.add_shape_cylinder(radius=float(size[0]), half_height=float(size[1]), **kwargs)
        else:
            raise ValueError(f"Unsupported geometry: {shape_type}")
    model = builder.finalize(device=device)
    states = (model.state(), model.state())
    for state in states:
        state.body_qd.assign(initial[None, 7:])
        newton.eval_ik(model, state, state.joint_q, state.joint_qd)
    solver = SolverMuJoCo(
        model,
        use_mujoco_contacts=False,
        integrator="implicitfast",
        cone="pyramidal",
        iterations=100,
        nconmax=300,
        njmax=1000,
    )
    pipeline = newton.CollisionPipeline(model)
    return model, states, solver, pipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument(
        "--disable-feedback", action="store_true", help="Diagnostic only: do not apply estimated wrench"
    )
    args = parser.parse_args()
    device = "cuda:0"
    torch.cuda.set_device(device)
    torch.set_num_threads(4)
    wp.init()
    wp.set_device(device)
    saved = torch.load(args.checkpoint, map_location=device, weights_only=False)
    arch = saved.get("architecture") or saved["config"].get("architecture", "particle_transolver_v1")
    is_world_like = arch in ("world_particle_v1", "particle_nerd_local_v1")
    if not is_world_like:
        validate_object_checkpoint(saved)
    if arch == "particle_nerd_local_v1":
        m_cfg = dict(saved["model_config"])
        if "spatial_slice_assign" not in m_cfg:
            m_cfg["spatial_slice_assign"] = False
            m_cfg["use_rms_swiglu"] = False
        network = ParticleNeRDLocalNetwork(**m_cfg).to(device).eval()
    elif arch == "world_particle_v1":
        network = WorldParticleNetwork(**saved["model_config"]).to(device).eval()
    else:
        network = ParticleTransolver(**saved["model_config"]).to(device).eval()
    network.load_state_dict(saved["model"])
    use_vel_state = is_world_like and bool(network.config.get("translation_invariant", False))
    dataset = Episodes(
        args.reference.parent, "test", collider_input="material_step" if is_world_like else "pre_rigid_step"
    )
    if len(dataset.records) != 1 or dataset.records[0][2] != args.reference:
        raise ValueError("Reference folder must contain exactly the requested reference")
    record, manifest, _ = dataset.records[0]
    if record["mode"] != "dynamic" or record["motion"] != "drop":
        raise ValueError("This comparison uses dynamic drop-and-support references")
    sample = dataset.sequence(np.random.default_rng(42), 1, device, index=0, start=0)
    dt = sample["dt"]
    steps = round(args.duration / dt)
    args.output.mkdir(parents=True, exist_ok=False)
    with h5py.File(args.reference, "r") as reference:
        model, states, rigid, pipeline = build_rigid(reference, device)
        mass = reference["particle_mass"][:]
        body_mass = float(reference["body_mass"][0])
        inertia = reference["body_inertia"][0].astype(np.float64)
        initial_body = reference["body_state"][0]
        initial_particles = reference["particle_position"][0]
    if steps > round(record["duration_s"] / dt):
        raise ValueError("Duration exceeds matched reference")
    estimator_cls = RecordedForceEstimator if is_world_like else NeRDRecordedForceEstimator
    estimator = estimator_cls(
        mass,
        body_mass,
        inertia,
        manifest["body_size_m"],
        record["shape"],
        dt,
        float(sample["friction"][0]),
        device=device,
    )
    contacts = pipeline.contacts()
    control = model.control()
    body_states = [initial_body]
    collider_states, wrenches, applied_wrenches, external_wrenches, stats = [], [], [], [], []
    history = sample["history"]
    stepper = (
        None
        if is_world_like
        else ObjectParticleStepper(sample["kinds"], sample["sizes"], sample["friction"], sample["material"], dt)
    )
    poses = sample["poses"][0].clone()
    velocities = sample["velocities"][0].clone()
    lagged = np.zeros(6, dtype=np.float32)
    start = time.perf_counter()
    failure = None
    output = args.output / "prediction.h5"
    with h5py.File(output, "w") as recording, torch.inference_mode():
        positions = recording.create_dataset(
            "particle_position",
            shape=(steps + 1, len(mass), 3),
            dtype="f4",
            chunks=(8, len(mass), 3),
            compression="gzip",
            compression_opts=1,
            shuffle=True,
        )
        positions[0] = initial_particles
        for step in range(steps):
            begin_pose = states[0].body_q.numpy()[0]
            begin_velocity = states[0].body_qd.numpy()[0]
            external = np.zeros(6, dtype=np.float32)
            if step < round(1.0 / dt):
                external[2] = body_mass * 9.81
            applied = np.zeros(6, dtype=np.float32) if args.disable_feedback else lagged.copy()
            for _ in range(2):
                states[0].body_f.assign((external + applied)[None])
                pipeline.collide(states[0], contacts)
                rigid.step(states[0], states[1], control, contacts, dt / 2)
                states = (states[1], states[0])
            body_state = np.concatenate((states[0].body_q.numpy()[0], states[0].body_qd.numpy()[0]))
            free_velocity = body_state[7:].copy()
            free_velocity[:3] -= dt * applied[:3] / body_mass
            rotation = np.asarray(wp.quat_to_matrix(wp.quat(*begin_pose[3:]))).reshape(3, 3)
            free_velocity[3:] -= dt * (rotation @ np.linalg.solve(inertia, rotation.T @ applied[3:]))
            collider = np.concatenate((begin_pose, free_velocity))
            if is_world_like:
                poses[0] = torch.as_tensor(begin_pose, device=device)
                velocities[0] = torch.as_tensor(begin_velocity, device=device)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    if use_vel_state:
                        current, vel_next = network.step_from_history(
                            history,
                            poses,
                            velocities,
                            sample["kinds"],
                            sample["sizes"],
                            sample["friction"],
                            sample["material"],
                            dt=dt,
                            return_vel=True,
                        )
                    else:
                        current = network.step_from_history(
                            history,
                            poses,
                            velocities,
                            sample["kinds"],
                            sample["sizes"],
                            sample["friction"],
                            sample["material"],
                            dt=dt,
                        )
                x = current.cpu().numpy()
                old = history[:, -1].cpu().numpy()
                estimator.estimate(old, (x - old) / dt, collider)
            else:
                poses[0] = torch.as_tensor(begin_pose, device=device)
                velocities[0] = torch.as_tensor(begin_velocity, device=device)
                current = stepper.step(network, history, poses, velocities)
                x = current.cpu().numpy()
                old = history[:, -1].cpu().numpy()
                estimator.estimate(old, (x - old) / dt, collider, active_idx=stepper.geometry["cached_topk"])
            lagged = estimator.estimator.wrench.numpy()[0].copy()
            diagnostic = estimator.estimator.stats.numpy()
            if not all(np.isfinite(v).all() for v in (x, body_state, lagged)) or np.max(np.abs(x)) > 5.0:
                failure = {"time_s": (step + 1) * dt, "reason": "Nonfinite state/wrench or particle outside 5 m limit"}
                break
            positions[step + 1] = x
            body_states.append(body_state)
            collider_states.append(collider)
            wrenches.append(lagged)
            applied_wrenches.append(applied)
            external_wrenches.append(external)
            stats.append(diagnostic)
            if use_vel_state:
                prev_synth = current - dt * vel_next
                history = torch.stack((history[:, 1], prev_synth, current), dim=1).contiguous()
            else:
                history = torch.cat((history[:, 1:], current[:, None]), dim=1).contiguous()
            if (step + 1) % 200 == 0:
                print(
                    json.dumps(
                        {
                            "time_s": (step + 1) * dt,
                            "body_z_m": float(body_state[2]),
                            "material_fz_N": float(lagged[2]),
                            "contacts": float(diagnostic[0]),
                            "elapsed_s": round(time.perf_counter() - start, 1),
                        }
                    ),
                    flush=True,
                )
        completed = len(body_states) - 1
        if completed != steps:
            # Keep only completed states, without silently padding a failed run.
            valid = positions[: completed + 1]
            del recording["particle_position"]
            recording.create_dataset("particle_position", data=valid, compression="gzip", compression_opts=1)
        for name, values in (
            ("body_state", body_states),
            ("collider_state", collider_states),
            ("material_wrench", wrenches),
            ("applied_material_wrench", applied_wrenches),
            ("external_wrench", external_wrenches),
            ("contact_stats", stats),
        ):
            recording.create_dataset(name, data=np.asarray(values, dtype=np.float32))
        recording.attrs["complete"] = failure is None and completed == steps
    metadata = dict(
        episode=record,
        checkpoint_sha256=digest(args.checkpoint),
        reference_sha256=digest(args.reference),
        reference_path=str(args.reference),
        prediction_sha256=digest(output),
        native_dt_s=dt,
        expected_steps=steps,
        completed_steps=completed,
        failure=failure,
        elapsed_s=time.perf_counter() - start,
        mode="Fully dynamic object and recurrent neural particles with lagged estimated force/torque feedback",
        feedback_enabled=not args.disable_feedback,
        active_selection=OBJECT_PIPELINE["active_k"],
        displacement_gate=True,
        particle_projection=OBJECT_PIPELINE["particle_projection"],
        pipeline=OBJECT_PIPELINE,
        force_filter=None,
        native_mpm_steps=0,
        body_input="Own begin-of-step pose and velocity",
        force_body_input="Own MJWarp end velocity minus the previously applied material impulse",
        force_application=(
            "Estimated world force and center-of-mass torque applied over both MJWarp substeps "
            "of the next 5 ms exchange"
        ),
        release="Upward external force balances gravity for t < 1 s; removed at 1 s. No later pose or velocity writes.",
        force_parameters={
            **CONTACT_SETTINGS,
            "friction": float(sample["friction"][0]),
            "contact_solver": "rft_body_velocity",
            "tolerance": 1e-9,
        },
        rigid=dict(
            integrator="implicitfast",
            cone="pyramidal",
            substeps=2,
            substep_dt_s=dt / 2,
            body_mass_kg=body_mass,
            body_inertia_kg_m2=inertia.tolist(),
            geometry="Recorded moving collision shape and five rigid tray shapes",
            contacts="Newton collision pipeline",
        ),
        network_source_sha256=(
            {
                str(Path(inspect.getfile(obj)).name): digest(inspect.getfile(obj))
                for obj in (
                    (features_particle_nerd_local, ParticleNeRDLocalNetwork)
                    if arch == "particle_nerd_local_v1"
                    else (features_world_particle, WorldParticleNetwork)
                )
            }
            if is_world_like
            else nerd_source_hashes()
        ),
        implementation_source_sha256={
            str(Path(inspect.getfile(obj))): digest(inspect.getfile(obj))
            for obj in (main, estimator_cls, type(estimator.estimator))
        },
        versions={name: importlib.metadata.version(name) for name in ("torch", "warp-lang", "newton", "numpy")},
        gpu=torch.cuda.get_device_name(),
    )
    (args.output / "evaluation.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"completed_steps": completed, "failure": failure, "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
