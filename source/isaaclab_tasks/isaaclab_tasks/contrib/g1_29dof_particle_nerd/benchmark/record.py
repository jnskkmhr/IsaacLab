# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Record matched fixed-bed G1 rollouts, sampling foot reactions every physics step."""

import argparse
import copy
import hashlib
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import warp as wp
from isaaclab_newton.physics import NewtonManager
from rsl_rl.models import MLPModel

from isaaclab.app import launch_simulation
from isaaclab.utils import to_dict

from isaaclab_tasks.contrib.g1_29dof_mpm.env_cfg.physics_cfg import configure_sparse_mpm_capacities
from isaaclab_tasks.utils import resolve_task_config


def make_config(backend, speed, duration, seed):
    """Use the NeRD flat scene for both solvers, with the robot initially above sand."""
    task = "IsaacContrib-Velocity-Sand-Flat-G1-29dof-ParticleNeRD-Play"
    reference, agent = resolve_task_config(task, "rsl_rl_cfg_entry_point", overrides=[])
    if backend == "mpm":
        task = task.replace("ParticleNeRD", "MPM")
        cfg, _ = resolve_task_config(task, "rsl_rl_cfg_entry_point", overrides=[])
        for name in ("scene", "observations", "actions", "commands", "events", "rewards", "terminations", "curriculum"):
            setattr(cfg, name, copy.deepcopy(getattr(reference, name)))
        cfg.sim.physics.solver_cfg.entries[1].solver_cfg.voxel_size = reference.scene.sand.spawn.voxel_size
        cfg.particle_max_velocity = reference.particle_max_velocity
        cfg.reset_particle_jitter = reference.reset_particle_jitter
    else:
        cfg = reference
    cfg.seed = seed
    cfg.scene.num_envs = 1
    # The default task starts on a rigid approach; a zero command there would never test sand.
    cfg.scene.robot.init_state.pos = (-2.0, 0.0, 0.76)
    cfg.sim.dt = 0.005
    cfg.decimation = 4
    cfg.sim.visualizer_cfgs = []
    cfg.video_recorders = []
    cfg.episode_length_s = duration + 2.0
    cmd = cfg.commands.base_velocity
    cmd.ranges.lin_vel_x = (speed, speed)
    cmd.ranges.lin_vel_y = (0.0, 0.0)
    cmd.ranges.ang_vel_z = (0.0, 0.0)
    cmd.rel_standing_envs = float(speed == 0.0)
    cmd.resampling_time_range = (1000.0, 1000.0)
    if backend == "mpm":
        configure_sparse_mpm_capacities(cfg)
    return task, cfg, agent


@torch.inference_mode()
def record(args):
    """Run one closed-loop policy episode and save forces before any environment reset."""
    task, cfg, agent = make_config(args.backend, args.speed, args.duration, args.seed)
    # Fixed proxy relaxation is an EMA of the complete sand-to-body wrench.
    # Keep this experiment opt-in; task and production solver defaults stay unchanged.
    alpha = 1.0 if args.force_filter_tau_ms == 0 else -np.expm1(-cfg.sim.dt / (args.force_filter_tau_ms * 0.001))
    if args.force_filter_tau_ms:
        assert cfg.sim.physics.solver_cfg.iterations == 1
        for proxy in cfg.sim.physics.solver_cfg.proxies:
            proxy.proxy_relaxation = float(alpha)
            proxy.proxy_relaxation_mode = "fixed"
    training_config = args.checkpoint.parent / "training_config.json"
    saved = json.loads(training_config.read_text()) if training_config.exists() else None
    if saved is not None:
        cfg.observations.policy.history_length = saved["env_cfg"]["observations"]["policy"]["history_length"]
        trained_action = saved["env_cfg"]["actions"]["joint_pos"]
        for key in ("joint_names", "scale", "use_default_offset", "preserve_order"):
            assert getattr(cfg.actions.joint_pos, key) == trained_action[key], f"Action contract differs: {key}"
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "backend": args.backend,
        "force_filter": {
            "type": "none" if args.force_filter_tau_ms == 0 else "exponential_moving_average",
            "tau_ms": args.force_filter_tau_ms,
            "alpha": float(alpha),
            "scope": "sand-to-body force and torque; feedback used by next lagged rigid step",
            "initial_history": "zero feedback after reset",
        },
        "task": task,
        "seed": args.seed,
        "command_m_s": args.speed,
        "physics_dt_s": cfg.sim.dt,
        "control_dt_s": cfg.sim.dt * cfg.decimation,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "initial_root_position_m": cfg.scene.robot.init_state.pos,
        "material": to_dict(cfg.scene.sand.spawn.material),
        "particle_grid": {
            name: getattr(cfg.scene.sand.spawn, name)
            for name in ("lower", "upper", "voxel_size", "particles_per_cell", "particle_placement", "jitter")
        },
        "physics": to_dict(cfg.sim.physics),
        "native_warmup_steps": 0,
        "force_frame": "world; force exerted on each foot; z points upwards",
        "comparison": (
            "Independent closed-loop rollouts; same policy, initial particles and scene, not identical foot motion"
        ),
    }
    rows = []
    controls = []
    with launch_simulation(cfg):
        env = gym.make(task, cfg=cfg).unwrapped
        original_update = env.scene.update
        try:
            obs, _ = env.reset()
            actor_cfg = copy.deepcopy(saved["train_cfg"]["actor"]) if saved else agent.actor.to_dict()
            assert actor_cfg.pop("class_name") == "MLPModel"
            obs_groups = saved["train_cfg"]["obs_groups"] if saved else agent.obs_groups
            actor = MLPModel(obs, obs_groups, "actor", env.action_manager.total_action_dim, **actor_cfg).to(env.device)
            checkpoint = torch.load(args.checkpoint, map_location=env.device, weights_only=True)
            actor.load_state_dict(checkpoint["actor_state_dict"], strict=True)
            actor.eval()
            model = NewtonManager.get_model()
            robot = env.scene["robot"]
            foot_ids, foot_names = robot.find_bodies(cfg.foot_body_expr, preserve_order=True)
            metadata["foot_names"] = foot_names
            metadata["particle_count"] = model.particle_count
            metadata["initial_particles_sha256"] = hashlib.sha256(
                NewtonManager.get_state_0().particle_q.numpy().tobytes()
            ).hexdigest()
            robot_ids = [i for i, name in enumerate(model.body_label) if "/Robot/" in name]
            metadata["robot_mass_kg"] = float(model.body_mass.numpy()[robot_ids].sum())
            metadata["particle_mass_kg"] = float(model.particle_mass.numpy()[0])
            metadata["active_k"] = getattr(NewtonManager._solver.solver("sand"), "active_counts", None)
            sand_solver = NewtonManager._solver.solver("sand")
            wrench_order = None
            if args.backend == "nerd":
                collider_names = [
                    sand_solver.model.body_label[i].split("/")[-1] for i in sand_solver.collider_ids_wp.numpy()
                ]
                wrench_order = [collider_names.index(name) for name in foot_names]
            metadata["outcome"] = "duration"
            metadata["checkpoint_iteration"] = int(checkpoint["iter"])

            def sample_physics_step(dt):
                original_update(dt)
                if not np.isclose(dt, cfg.sim.dt):
                    return
                # Read the same feedback buffer used by the task, at 200 Hz rather than 50 Hz.
                feedback = wp.to_torch(env._coupling_forces)[env._foot_body_ids[0]].clone()
                sand = feedback[:, :3]
                raw = (
                    wp.to_torch(sand_solver.estimator.wrench)[wrench_order].clone()
                    if wrench_order is not None
                    else feedback.clone()
                )
                env._refresh_contact_forces()
                rigid = wp.to_torch(env._rigid_foot_force).reshape(1, len(foot_names), 3)[0].clone()
                if not torch.isfinite(sand).all() or not torch.isfinite(rigid).all():
                    raise FloatingPointError("Nonfinite raw foot reaction force")
                row = {
                    "time": (len(rows) + 1) * cfg.sim.dt,
                    "sand_force": sand.cpu().numpy(),
                    "raw_sand_wrench": raw.cpu().numpy(),
                    "feedback_wrench": feedback.cpu().numpy(),
                    "rigid_force": rigid.cpu().numpy(),
                    "force": (sand + rigid).cpu().numpy(),
                    "foot_position": robot.data.body_pos_w.torch[0, foot_ids].cpu().numpy().copy(),
                    "foot_velocity": robot.data.body_lin_vel_w.torch[0, foot_ids].cpu().numpy().copy(),
                    "root_position": robot.data.root_pos_w.torch[0].cpu().numpy().copy(),
                    "root_velocity": robot.data.root_lin_vel_w.torch[0].cpu().numpy().copy(),
                    "root_quaternion": robot.data.root_quat_w.torch[0].cpu().numpy().copy(),
                }
                rows.append(row)

            env.scene.update = sample_physics_step
            for i in range(round(args.duration / env.step_dt)):
                action = actor(obs)
                if agent.clip_actions is not None:
                    action = action.clamp(-agent.clip_actions, agent.clip_actions)
                controls.append(action[0].cpu().numpy().copy())
                obs, _, terminated, truncated, _ = env.step(action)
                if i % 50 == 0:
                    print(f"PROGRESS {args.backend} command={args.speed} t={(i + 1) * env.step_dt:.2f}s", flush=True)
                if terminated.any() or truncated.any():
                    metadata["outcome"] = "terminated"
                    metadata["terminal_terms"] = [
                        name
                        for name in env.termination_manager.active_terms
                        if bool(env.termination_manager.get_term(name)[0])
                    ]
                    break
                if rows[-1]["root_position"][0] > cfg.scene.sand.spawn.upper[0] - 0.4:
                    metadata["outcome"] = "reached_sand_bed_edge"
                    break
        finally:
            env.scene.update = original_update
            if rows:
                np.savez_compressed(
                    args.output / "trace.npz",
                    action=np.asarray(controls),
                    **{key: np.asarray([row[key] for row in rows]) for key in rows[0]},
                )
                metadata["recorded_seconds"] = rows[-1]["time"]
            (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str) + "\n")
            env.close()
    print(f"SAVED {args.output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", required=True, choices=("mpm", "nerd"))
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--speed", type=float, default=0.0)
    parser.add_argument("--duration", type=float, default=6.0)
    parser.add_argument(
        "--force_filter_tau_ms",
        type=float,
        default=0.0,
        help="Opt-in EMA time constant for NeRD sand feedback; zero disables filtering.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.duration <= 0:
        parser.error("duration must be positive")
    if not np.isfinite(args.force_filter_tau_ms) or args.force_filter_tau_ms < 0:
        parser.error("force_filter_tau_ms must be finite and nonnegative")
    if args.force_filter_tau_ms and args.backend != "nerd":
        parser.error("This filtering experiment applies to NeRD only")
    record(args)


if __name__ == "__main__":
    main()
