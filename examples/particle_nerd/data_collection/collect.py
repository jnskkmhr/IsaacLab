# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""One open-ground MPM trajectory, also used by the dataset collection workers."""

import argparse
import hashlib
import importlib.metadata
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import warp as wp

from particle_nerd.data_collection.trajectories import PATHS, drag_target_offset


def configure_drag_path(cfg, args, sustained):
    """Configure controlled path tests without changing existing campaign defaults."""
    if args.drag_path == "training":
        return
    if args.case != "continuous-drag" or args.duration <= 1.25:
        raise ValueError("Named drag paths require continuous-drag and duration >1.25 s")
    # Only the initial box state and external dragging command change.
    cfg.events.randomize_box.params["pose_range"] = {key: (0.0, 0.0) for key in ("x", "y", "z", "roll", "pitch", "yaw")}
    cfg.events.randomize_box.params["velocity_range"] = {
        key: (0.0, 0.0) for key in ("x", "y", "z", "roll", "pitch", "yaw")
    }
    cfg.scene.body5.init_state.pos = (0.0, 0.0, 0.60)
    sustained.update(
        active_interval_s=[0.25, args.duration - 0.25],
        anchor_local_m=[0.0, 0.0, 0.0],
        named_path=args.drag_path,
        lead_in_s=0.75,
        initialization="Settle at bed center; bounded force moves box toward path start",
        path_direction=args.path_direction,
        horizontal_span_m=[0.76, 0.76 if args.drag_path == "wide-circle" else 0.52],
        target_time_parameter="quintic smoothstep over active interval",
        vertical_target_change_m=-0.02,
    )


def configure_airborne_release(cfg, args):
    """Hold an initially stationary box clear of the material during preparation."""
    if args.case == "airborne":
        cfg.scene.body5.init_state.pos = (0.0, 0.0, 0.75)
        for key in ("pose_range", "velocity_range"):
            cfg.events.randomize_box.params[key] = {
                axis: (0.0, 0.0) for axis in ("x", "y", "z", "roll", "pitch", "yaw")
            }


def support_box_before_release(command, args, step, release_step):
    """Compensate box gravity during material preparation, with no support after release."""
    if args.case == "airborne" and step < release_step:
        command[0, 2] = args.body_mass * 9.81


def set_airborne_launch_velocity(env, args, step, release_step):
    """Assign the box's initial launch velocity once, at the evaluation boundary."""
    if args.case == "airborne" and step == release_step:
        env.scene["body5"].write_root_velocity_to_sim_index(
            root_velocity=torch.tensor([list(args.launch_velocity) + [0.0, 0.0, 0.0]], device=env.device)
        )
        from isaaclab_newton.physics import NewtonManager

        NewtonManager.forward()


def configure_body_mass(args, p):
    """Use measured shovel mass/inertia; retain the box mass option."""
    asset_metadata = None
    if args.geometry == "shovel":
        from particle_nerd.shovel.asset import ASSET_DIR

        asset_metadata = json.loads((ASSET_DIR / "metadata.json").read_text())
        if args.body_mass is not None and not np.isclose(args.body_mass, asset_metadata["body_mass_kg"]):
            p.error("Shovel uses the reference asset's measured mass and inertia")
    if args.body_mass is None:
        args.body_mass = asset_metadata["body_mass_kg"] if asset_metadata else 7.5
    if not np.isfinite(args.body_mass) or args.body_mass <= 0:
        p.error("Body mass must be finite and positive")
    return asset_metadata


def validate_rigid_asset(model, args, asset_metadata):
    """Reject silently changed imported geometry or inertia before recording."""
    from newton import GeoType, ShapeFlags

    colliding = (model.shape_flags.numpy() & int(ShapeFlags.COLLIDE_SHAPES | ShapeFlags.COLLIDE_PARTICLES)) != 0
    shape_types = model.shape_type.numpy()[colliding]
    num_envs = getattr(args, "num_envs", 1)
    assert model.body_count == num_envs
    # Viewers may retain extra visual-only meshes; validate physical geometry.
    assert int(colliding.sum()) == (1 + 66 * num_envs if asset_metadata else 1 + num_envs), (
        int(colliding.sum()),
        model.shape_label,
    )
    np.testing.assert_allclose(model.body_mass.numpy(), np.full(num_envs, args.body_mass), rtol=1e-6)
    assert sum(shape_types == int(GeoType.PLANE)) == 1
    if asset_metadata:
        np.testing.assert_allclose(model.body_com.numpy(), np.zeros((num_envs, 3)), atol=1e-7)
        np.testing.assert_allclose(
            model.body_inertia.numpy(), np.tile(asset_metadata["body_inertia"], (num_envs, 1, 1)), atol=1e-6
        )
    else:
        assert sum(shape_types == int(GeoType.BOX)) == num_envs


def create_collection_env_cfg(geometry, body_mass, seed, *, bed_size=(1.0, 1.0, 0.5)):
    """Shared open-ground MPM/MJWarp scene for dataset collection and interactive play."""
    from isaaclab.assets import AssetBaseCfg
    from isaaclab.managers import ActionTermCfg, RecorderManagerBaseCfg
    from isaaclab.sim import GroundPlaneCfg
    from isaaclab.sim.spawners.materials import UsdPhysicsRigidBodyMaterialCfg

    from particle_nerd.data_collection.env_cfg import ParticleEnvCfg
    from particle_nerd.data_collection.scene_cfg import configure_collection_scene
    from particle_nerd.data_collection.terms.actions import BodyWrenchAction

    cfg = ParticleEnvCfg()
    cfg.seed = seed
    cfg.scene.num_envs = 1
    cfg.recorders = RecorderManagerBaseCfg()
    cfg.actions.wrench = ActionTermCfg(class_type=BodyWrenchAction, asset_name="body5")
    metadata = configure_collection_scene(cfg, "small", 0.04, bed_size, 1.25)
    for i in range(5):
        setattr(cfg.scene, f"body{i}", None)
    cfg.scene.ground = AssetBaseCfg(
        prim_path="/World/Ground",
        spawn=GroundPlaneCfg(
            size=(10.0, 10.0),
            physics_material=UsdPhysicsRigidBodyMaterialCfg(static_friction=0.6, dynamic_friction=0.6),
        ),
    )
    length, width, depth = bed_size
    cfg.scene.sand.spawn.lower = (-length / 2, -width / 2, 0.0)
    cfg.scene.sand.spawn.upper = (length / 2, width / 2, depth)
    cfg.events.reset_material.params.update(jitter_fraction=0.2, particle_spacing=metadata["particle_spacing_m"])
    cfg.events.randomize_box.params["pose_range"].update(
        x=(-0.08, 0.08), y=(-0.08, 0.08), z=(0.0, 0.06), roll=(-0.25, 0.25), pitch=(-0.25, 0.25), yaw=(-3.14, 3.14)
    )
    cfg.events.randomize_box.params["velocity_range"] = dict(
        zip(
            ("x", "y", "z", "roll", "pitch", "yaw"),
            ((-0.25, 0.25), (-0.25, 0.25), (-0.15, 0.0), (-1.0, 1.0), (-1.0, 1.0), (-2.0, 2.0)),
            strict=True,
        )
    )
    cfg.scene.body5.init_state.pos = (0.0, 0.0, depth + 0.10)
    cfg.scene.body5.spawn.mass_props.mass = body_mass
    if geometry == "shovel":
        from isaaclab.sim import UsdFileCfg

        from particle_nerd.shovel.asset import ASSET_DIR

        old = cfg.scene.body5.spawn
        cfg.scene.body5.spawn = UsdFileCfg(
            usd_path=str(ASSET_DIR / "shovel.usda"),
            rigid_props=old.rigid_props,
            collision_props=old.collision_props,
            physics_material=UsdPhysicsRigidBodyMaterialCfg(static_friction=0.5, dynamic_friction=0.5),
        )
    coupling = cfg.sim.physics.solver_cfg
    coupling.entries[0].bodies = [r"/World/envs/env_.*/Body5"]
    coupling.proxies[0].bodies = [r"/World/envs/env_.*/Body5"]
    coupling.entries[0].include_static_shapes = True
    coupling.entries[1].include_static_shapes = False
    coupling.entries[1].solver_cfg.max_iterations = 25
    metadata.update(
        scene="open-ground",
        container_walls=False,
        tray_motion=None,
        ground_z_m=0.0,
        initial_material_surface_z_m=depth,
        voxel_size_m=0.04,
    )
    return cfg, metadata


def main():
    from isaaclab.app import add_launcher_args, launch_simulation

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--case", choices=("rest", "push-spin", "drag-lift", "long-push", "continuous-drag", "airborne"), required=True
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--duration", type=float, default=5.0)
    p.add_argument("--body-mass", type=float, default=None)
    p.add_argument("--geometry", choices=("box", "shovel"), default="box")
    p.add_argument("--name", help="Output name; defaults to the motion case")
    p.add_argument("--seed", type=int, help="Independent episode seed")
    p.add_argument("--randomize-motion", action="store_true", help="Randomize sustained push/drag parameters")
    p.add_argument("--drag-path", choices=("training", *PATHS), default="training")
    p.add_argument("--path-direction", type=int, choices=(-1, 1), default=1)
    p.add_argument("--launch-velocity", nargs=3, type=float, default=(0.0, 0.0, 0.0), metavar=("VX", "VY", "VZ"))
    add_launcher_args(p)
    args = p.parse_args()
    if args.duration <= 0:
        p.error("Duration must be positive")
    asset_metadata = configure_body_mass(args, p)
    args.output.mkdir(parents=True, exist_ok=True)
    stem = args.output / (args.name or args.case)
    if stem.with_suffix(".hdf5").exists() or stem.with_suffix(".partial.hdf5").exists():
        raise FileExistsError(stem)

    from isaaclab_newton.physics import NewtonManager

    from isaaclab.envs import ManagerBasedEnv

    case_id = {"rest": 0, "push-spin": 1, "drag-lift": 2, "long-push": 0, "continuous-drag": 0, "airborne": 0}[
        args.case
    ]
    seed = 41001 + case_id if args.seed is None else args.seed
    rng = np.random.default_rng(seed)
    cfg, metadata = create_collection_env_cfg(args.geometry, args.body_mass, seed)
    material = {
        k: getattr(cfg.scene.sand.spawn.material, k)
        for k in (
            "density",
            "young_modulus",
            "poisson_ratio",
            "friction",
            "yield_pressure",
            "tensile_yield_ratio",
            "yield_stress",
            "hardening",
            "dilatancy",
            "viscosity",
        )
    }
    # Random timings/directions are fixed by the case seed and recorded verbatim.
    pulses = []
    for start in (rng.uniform(0.4, 0.8), rng.uniform(1.8, 2.3), rng.uniform(3.3, 3.8)):
        angle = rng.uniform(-np.pi, np.pi)
        magnitude = rng.uniform(12.0, 22.0)
        pulses.append(
            dict(
                start=float(start),
                duration=float(rng.uniform(0.25, 0.45)),
                wrench=[
                    magnitude * np.cos(angle),
                    magnitude * np.sin(angle),
                    float(rng.uniform(-5.0, 5.0)),
                    *rng.uniform(-0.12, 0.12, 3),
                ],
            )
        )
    drag_times = np.array([0.0, 0.5, 1.3, 2.2, 3.2, 4.2, 5.0])
    drag_offsets = rng.uniform(-0.22, 0.22, (7, 3))
    drag_offsets[:, 2] = [0.0, 0.0, 0.02, 0.12, 0.28, 0.08, 0.0]
    drag_offsets[:2] = 0.0
    sustained = dict(
        active_interval_s=[0.25, 4.9],
        ramp_s=0.2,
        push_force_weight_fraction=0.9,
        downward_force_weight_fraction=0.2,
        push_damping_per_mass_per_s=6.0,
        push_turn_period_s=4.65,
        drag_radius_x_m=0.35,
        drag_radius_y_m=0.25,
        drag_turn_period_s=3.0,
        drag_vertical_offset_m=-0.02,
        drag_vertical_amplitude_m=0.04,
        drag_stiffness_per_mass_per_s2=120.0,
        drag_damping_per_mass_per_s=15.0,
        drag_force_limit_weight_fraction=2.5,
        anchor_local_m=[0.04, 0.02, 0.03],
    )
    sustained.update(direction=1.0, horizontal_angle_rad=0.0)
    if args.randomize_motion:
        sustained.update(
            active_interval_s=[float(rng.uniform(0.1, 0.8)), float(rng.uniform(4.5, 4.95))],
            push_force_weight_fraction=float(rng.uniform(0.6, 1.8)),
            downward_force_weight_fraction=float(rng.uniform(0.0, 0.35)),
            push_damping_per_mass_per_s=float(rng.uniform(4.0, 9.0)),
            push_turn_period_s=float(rng.uniform(2.5, 7.0)),
            drag_radius_x_m=float(rng.uniform(0.18, 0.40)),
            drag_radius_y_m=float(rng.uniform(0.12, 0.28)),
            drag_turn_period_s=float(rng.uniform(2.5, 5.0)),
            drag_vertical_offset_m=float(rng.uniform(-0.05, 0.01)),
            drag_vertical_amplitude_m=float(rng.uniform(0.0, 0.06)),
            drag_stiffness_per_mass_per_s2=float(rng.uniform(90.0, 150.0)),
            drag_damping_per_mass_per_s=float(rng.uniform(12.0, 20.0)),
            drag_force_limit_weight_fraction=float(rng.uniform(2.0, 3.0)),
            direction=float(rng.choice([-1, 1])),
            horizontal_angle_rad=float(rng.uniform(-np.pi, np.pi)),
            anchor_local_m=rng.uniform([-0.05, -0.05, 0.015], [0.05, 0.05, 0.035]).tolist(),
        )
    configure_drag_path(cfg, args, sustained)
    configure_airborne_release(cfg, args)
    anchor_local = torch.tensor(sustained["anchor_local_m"], device="cuda:0")
    dt = cfg.sim.dt
    post_steps = round(args.duration / dt)
    max_settle_steps = round(5.0 / dt)
    # Legacy field also marks the rollout start for airborne evaluation.
    settled_step = round(2.0 / dt) if args.case == "airborne" else None
    stable_steps = 0
    rows = {
        k: []
        for k in (
            "body_pose",
            "body_velocity",
            "external_wrench",
            "applied_feedback_wrench",
            "material_on_body_wrench",
            "ground_on_material_force",
            "time_s",
        )
    }
    if args.drag_path != "training":
        rows["drag_target_position"] = []
    partial = stem.with_suffix(".partial.hdf5")
    started = time.perf_counter()
    with launch_simulation(cfg, args):
        env = ManagerBasedEnv(cfg)
        try:
            model = NewtonManager.get_model()
            entry = NewtonManager._solver._entries["sand"]
            solver = entry.solver
            # Existing host-loop fallback; solver kernels and physical states remain on GPU.
            solver._use_cuda_graph = False
            validate_rigid_asset(model, args, asset_metadata)
            assert np.any(solver.collider_body_index.numpy() == -1)
            assert model.particle_count == 16384, model.particle_count
            feedback = next(
                m.coupling_forces
                for m in NewtonManager._solver._proxy_mappings
                if m.src_name == "rigid" and m.dst_name == "sand"
            )
            env.reset(seed=seed)
            n = model.particle_count
            command = torch.zeros((1, 6), device=env.device)
            from isaaclab.utils.math import quat_apply

            def state_arrays():
                s = NewtonManager.get_state_0()
                return s.particle_q.numpy(), s.particle_qd.numpy(), s.body_q.numpy()[0], s.body_qd.numpy()[0]

            with h5py.File(partial, "w") as f:
                xout = f.create_dataset(
                    "particle_position",
                    (0, n, 3),
                    maxshape=(None, n, 3),
                    chunks=(1, n, 3),
                    dtype="f4",
                    compression="lzf",
                )
                vout = f.create_dataset(
                    "particle_velocity",
                    (0, n, 3),
                    maxshape=(None, n, 3),
                    chunks=(1, n, 3),
                    dtype="f4",
                    compression="lzf",
                )
                f.create_dataset("particle_mass", data=model.particle_mass.numpy())
                f.create_dataset("particle_radius", data=model.particle_radius.numpy())
                frame_steps = []
                min_particle_z = float("inf")
                max_ground_violation_fraction = 0.0

                def record(step):
                    nonlocal min_particle_z, max_ground_violation_fraction
                    x, v, q, qd = state_arrays()
                    if not all(np.isfinite(a).all() for a in (x, v, q, qd)):
                        raise FloatingPointError(f"Nonfinite state at step {step}")
                    min_particle_z = min(min_particle_z, float(x[:, 2].min()))
                    max_ground_violation_fraction = max(max_ground_violation_fraction, float((x[:, 2] < -0.04).mean()))
                    rows["body_pose"].append(q.copy())
                    rows["body_velocity"].append(qd.copy())
                    rows["time_s"].append(step * dt)
                    if step % 4 == 0:
                        frame_steps.append(step)
                        for dataset, a in ((xout, x), (vout, v)):
                            dataset.resize(len(frame_steps), axis=0)
                            dataset[-1] = a
                    return q, qd

                q, qd = record(0)
                initial_q, initial_qd = q.copy(), qd.copy()
                drag_origin = None
                step = 0
                while settled_step is None or step < settled_step + post_steps:
                    if settled_step is None and step >= max_settle_steps:
                        raise RuntimeError("Box failed to settle within 5 seconds; no perturbations applied")
                    command.zero_()
                    support_box_before_release(command, args, step, settled_step)
                    named_target = q[:3].copy()
                    if settled_step is not None:
                        t = (step - settled_step) * dt
                        if (
                            args.case in ("long-push", "continuous-drag")
                            and sustained["active_interval_s"][0] <= t < sustained["active_interval_s"][1]
                        ):
                            s = NewtonManager.get_state_0()
                            pose = wp.to_torch(s.body_q)[0]
                            velocity = wp.to_torch(s.body_qd)[0]
                            elapsed = t - sustained["active_interval_s"][0]
                            envelope = min(
                                1.0,
                                elapsed / sustained["ramp_s"],
                                (sustained["active_interval_s"][1] - t) / sustained["ramp_s"],
                            )
                            weight = args.body_mass * 9.81
                            if args.case == "long-push":
                                angle = (
                                    sustained["horizontal_angle_rad"]
                                    + sustained["direction"] * 2 * np.pi * elapsed / sustained["push_turn_period_s"]
                                )
                                direction = torch.tensor(
                                    [np.cos(angle), np.sin(angle)], dtype=torch.float32, device=env.device
                                )
                                command[0, :2] = envelope * (
                                    sustained["push_force_weight_fraction"] * weight * direction
                                    - sustained["push_damping_per_mass_per_s"] * args.body_mass * velocity[:2]
                                )
                                command[0, 2] = -envelope * sustained["downward_force_weight_fraction"] * weight
                            else:
                                angle = sustained["direction"] * 2 * np.pi * elapsed / sustained["drag_turn_period_s"]
                                offset = np.array(
                                    [
                                        sustained["drag_radius_x_m"] * np.sin(angle),
                                        sustained["drag_radius_y_m"] * (1 - np.cos(angle)),
                                        sustained["drag_vertical_offset_m"]
                                        + sustained["drag_vertical_amplitude_m"] * np.sin(2 * angle),
                                    ]
                                )
                                if args.drag_path != "training":
                                    interval = sustained["active_interval_s"]
                                    offset = drag_target_offset(
                                        args.drag_path, elapsed, interval[1] - interval[0], args.path_direction
                                    )
                                az = sustained["horizontal_angle_rad"]
                                offset[:2] = (
                                    np.array([[np.cos(az), -np.sin(az)], [np.sin(az), np.cos(az)]]) @ offset[:2]
                                )
                                target = drag_origin + torch.as_tensor(offset, dtype=torch.float32, device=env.device)
                                if args.drag_path != "training":
                                    named_target = target.cpu().numpy().copy()
                                arm = quat_apply(pose[3:][None], anchor_local[None])[0]
                                point_velocity = velocity[:3] + torch.cross(velocity[3:], arm, dim=0)
                                force = args.body_mass * (
                                    sustained["drag_stiffness_per_mass_per_s2"] * (target - pose[:3] - arm)
                                    - sustained["drag_damping_per_mass_per_s"] * point_velocity
                                )
                                limit = sustained["drag_force_limit_weight_fraction"] * weight
                                force *= limit / force.norm().clamp_min(limit)
                                command[0, :3] = envelope * force
                                command[0, 3:] = torch.cross(arm, command[0, :3], dim=0)
                        elif args.case == "push-spin":
                            for pulse in pulses:
                                phase = (t - pulse["start"]) / pulse["duration"]
                                if 0 < phase < 1:
                                    command[0] += (
                                        torch.tensor(pulse["wrench"], device=env.device) * np.sin(np.pi * phase) ** 2
                                    )
                        elif args.case == "drag-lift" and 0.5 <= t < 4.8:
                            s = NewtonManager.get_state_0()
                            pose = wp.to_torch(s.body_q)[0]
                            velocity = wp.to_torch(s.body_qd)[0]
                            quat = pose[3:][None]
                            arm = quat_apply(quat, anchor_local[None])[0]
                            offset = np.array([np.interp(t, drag_times, drag_offsets[:, k]) for k in range(3)])
                            target = drag_origin + torch.as_tensor(offset, dtype=torch.float32, device=env.device)
                            point_velocity = velocity[:3] + torch.cross(velocity[3:], arm, dim=0)
                            force = 150.0 * (target - pose[:3] - arm) - 6.0 * point_velocity
                            force *= 45.0 / force.norm().clamp_min(45.0)
                            command[0, :3] = force
                            command[0, 3:] = torch.cross(arm, force, dim=0)
                    if args.drag_path != "training":
                        rows["drag_target_position"].append(named_target)
                    rows["external_wrench"].append(command[0].cpu().numpy().copy())
                    rows["applied_feedback_wrench"].append(feedback.numpy()[0].copy())
                    env.step(command)
                    impulses, points, colliders = solver.collect_collider_impulses(entry.state_0)
                    ids = wp.to_torch(colliders).long()
                    valid = ids >= 0
                    impulse = wp.to_torch(impulses)[valid]
                    point = wp.to_torch(points)[valid]
                    owner = wp.to_torch(solver.collider_body_index)[ids[valid]].long()
                    moving = owner >= 0
                    center = wp.to_torch(entry.state_0.body_q)[0, :3]
                    force = impulse[moving].sum(0) / dt
                    torque = torch.cross(point[moving] - center, impulse[moving], dim=-1).sum(0) / dt
                    rows["material_on_body_wrench"].append(torch.cat((force, torque)).cpu().numpy())
                    rows["ground_on_material_force"].append((-impulse[~moving].sum(0) / dt).cpu().numpy())
                    step += 1
                    set_airborne_launch_velocity(env, args, step, settled_step)
                    q, qd = record(step)
                    if settled_step is None:
                        supported = (
                            np.linalg.norm(rows["applied_feedback_wrench"][-1][:3]) > 0.1 * args.body_mass * 9.81
                        )
                        stable = (
                            step * dt >= 0.5
                            and np.linalg.norm(qd[:3]) < 0.02
                            and np.linalg.norm(qd[3:]) < 0.2
                            and supported
                        )
                        stable_steps = stable_steps + 1 if stable else 0
                        if stable_steps >= round(0.3 / dt):
                            # Align the five-second training window with saved 50 Hz states.
                            if step % 4 == 0:
                                settled_step = step
                                pose = wp.to_torch(NewtonManager.get_state_0().body_q)[0]
                                arm = quat_apply(pose[3:][None], anchor_local[None])[0]
                                drag_origin = (pose[:3] + arm).clone()
                                print(f"SETTLED {args.case} at {step * dt:.3f}s", flush=True)
                    if step % 100 == 0:
                        print(f"STEP {args.case} {step} settled={settled_step}", flush=True)
                for k, value in rows.items():
                    f.create_dataset(k, data=np.asarray(value))
                f.create_dataset("frame_steps", data=frame_steps)
                report = dict(
                    case=args.case,
                    seed=seed,
                    scene=metadata,
                    material=material,
                    particle_count=n,
                    body_mass_kg=args.body_mass,
                    geometry=args.geometry,
                    body_size_m=asset_metadata["body_size_m"] if asset_metadata else [0.12, 0.12, 0.08],
                    body_inertia=model.body_inertia.numpy()[0].tolist(),
                    body_contact_friction=0.5 if asset_metadata else 0.6,
                    geometry_metadata=asset_metadata,
                    initial_body_pose=initial_q.tolist(),
                    initial_body_velocity=initial_qd.tolist(),
                    dt=dt,
                    record_hz=50,
                    solver_max_iterations=25,
                    solver_tolerance=1e-4,
                    solver_iteration_policy="Newton stock convergence/iteration loop",
                    solver=solver.solver,
                    settled_step=settled_step,
                    settled_time_s=settled_step * dt,
                    settled_criterion="Airborne release after 2 s material preparation; box is not settled"
                    if args.case == "airborne"
                    else (
                        "Body speed <0.02 m/s and angular speed <0.2 rad/s, supported by sand (>10% "
                        "of body weight), sustained 0.3 s after initial 0.5 s"
                    ),
                    post_settle_duration_s=post_steps * dt,
                    evaluation_start="airborne release" if args.case == "airborne" else "settled box",
                    release_step=settled_step if args.case == "airborne" else None,
                    requested_launch_velocity=list(args.launch_velocity) if args.case == "airborne" else None,
                    release_body_pose=rows["body_pose"][settled_step].tolist(),
                    release_body_velocity=rows["body_velocity"][settled_step].tolist(),
                    total_steps=step,
                    jitter_fraction=0.2,
                    min_particle_center_z_m=min_particle_z,
                    max_fraction_particle_centers_below_ground_by_more_than_grid=max_ground_violation_fraction,
                    pulses=pulses if args.case == "push-spin" else [],
                    drag=dict(
                        times=drag_times.tolist(),
                        offsets_m=drag_offsets.tolist(),
                        anchor_local_m=anchor_local.cpu().tolist(),
                        stiffness_N_per_m=150,
                        damping_Ns_per_m=6,
                        force_limit_N=45,
                        active_interval_s=[0.5, 4.8],
                    )
                    if args.case == "drag-lift"
                    else None,
                    sustained_interaction=sustained if args.case in ("long-push", "continuous-drag") else None,
                    elapsed_wall_seconds=time.perf_counter() - started,
                    gpu=torch.cuda.get_device_name(),
                    packages={
                        k: importlib.metadata.version(k) for k in ("newton", "warp-lang", "torch", "mujoco-warp")
                    },
                    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    status="complete",
                    physics=(
                        f"MPM particles + MJWarp dynamic {args.geometry}; "
                        "no learned particle or estimated force feedback"
                    ),
                )
                f.attrs["manifest"] = json.dumps(report)
            partial.replace(stem.with_suffix(".hdf5"))
            stem.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
            print(f"COMPLETE {stem}", flush=True)
        finally:
            env.close()


if __name__ == "__main__":
    main()
