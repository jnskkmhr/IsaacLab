# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run dynamic G1 with native MPM or particle NeRD and analytical foot forces."""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import torch
import warp as wp

from particle_nerd.g1 import DEFAULT_POLICY
from particle_nerd.g1.collect import configure


def main():
    import sys

    frozen_source = Path.home() / "isaac/particle_nerd/experiments/recurrent-05s-20261021/g1-source-stage/source/source"
    for pkg in ("isaaclab_rl", "isaaclab_assets", "isaaclab_newton", "isaaclab_tasks", "isaaclab"):
        pkg_dir = frozen_source / pkg
        if pkg_dir.is_dir() and str(pkg_dir) not in sys.path:
            sys.path.insert(0, str(pkg_dir))
    for mod in [k for k in list(sys.modules) if k == "isaaclab" or k.startswith("isaaclab.") or k.startswith("isaaclab_")]:
        del sys.modules[mod]

    from isaaclab.app import add_launcher_args, launch_simulation

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, help="G1 particle checkpoint; omit for MPM")
    p.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--num-envs", "--num_envs", dest="num_envs", type=int, default=1, help="Number of parallel G1 environments")
    p.add_argument("--env-spacing", "--env_spacing", dest="env_spacing", type=float, default=5.0, help="Distance between G1 environment origins [m]")
    p.add_argument("--schedule", type=Path)
    p.add_argument("--trial", type=int, default=1)
    p.add_argument(
        "--command",
        type=float,
        nargs=3,
        metavar=("VX", "VY", "WZ"),
        help="Override base velocity command [lin_vel_x, lin_vel_y, ang_vel_z] in m/s and rad/s",
    )
    p.add_argument(
        "--command-end",
        type=float,
        nargs=3,
        metavar=("VX", "VY", "WZ"),
        help="Optional second velocity command after --switch-time [lin_vel_x, lin_vel_y, ang_vel_z]",
    )
    p.add_argument("--switch-time", type=float, default=None, help="Time in seconds to switch from --command to --command-end")
    p.add_argument(
        "--initial-pose",
        type=float,
        nargs=3,
        metavar=("X", "Y", "YAW"),
        help="Override initial robot pose [x_m, y_m, yaw_rad] relative to environment origin",
    )
    p.add_argument(
        "--diverse-commands",
        action="store_true",
        help="Assign distinct velocity commands (forward, backward, diagonal, curved) across parallel environments",
    )
    p.add_argument("--duration", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=120000)
    p.add_argument("--effective-mass-scale", type=float, default=10.0)
    p.add_argument("--iterations", type=int, default=50)
    p.add_argument("--contact-relaxation", type=float, default=0.1, help="Jacobi impulse-update relaxation in (0, 1]")
    p.add_argument(
        "--contact-solver",
        choices=("jacobi", "body_velocity", "rft_body_velocity"),
        default="jacobi",
        help="Analytical foot contact solver mode",
    )
    p.add_argument(
        "--active-k",
        type=lambda v: "none"
        if v.lower() in ("none", "all", "full", "off", "0")
        else ("auto" if v.lower() in ("auto", "heuristic") else int(v)),
        default=None,
        help="Active particle selection ('none'/'all' to disable, 'auto' for heuristic, or integer cap)",
    )
    p.add_argument("--video", action="store_true", help="Record an MP4 video to <output>/video.mp4 during play")
    p.add_argument("--video_length", "--video-length", dest="video_length", type=int, default=None, help="Video length in 50 Hz policy steps")
    p.add_argument(
        "--settle-steps",
        type=int,
        default=None,
        help="Native MPM physics steps to pre-settle the sand bed before rollout (default: 150 when --checkpoint is set)",
    )
    p.add_argument("--headless", action="store_true")
    add_launcher_args(p)
    args = p.parse_args()
    torch.set_num_threads(2)
    import gymnasium as gym
    from isaaclab_newton.physics import NewtonManager
    from rsl_rl.models import MLPModel

    from isaaclab.utils.math import quat_from_euler_xyz

    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg

    from particle_nerd.g1.adapter import LearnedMaterial

    task = "IsaacContrib-Velocity-Sand-G1-29dof-MPM-Play"
    cfg = configure(parse_env_cfg(task, num_envs=args.num_envs), args.num_envs)
    cfg.scene.env_spacing = args.env_spacing
    cfg.seed = args.seed
    cfg.sim.physics.use_cuda_graph = args.checkpoint is None
    cfg.sim.physics.load_visual_shapes = True
    cfg.episode_length_s = args.duration + 1
    for viz in cfg.sim.visualizer_cfgs:
        viz.headless = args.headless
        viz.origin_type = "world"
        viz.origin_track_path = None
    if args.video:
        viz_name = (args.visualizer or ["newton_rtx"])[0]
        cfg.sim.visualizer_cfgs = [v for v in cfg.sim.visualizer_cfgs if v.visualizer_type == viz_name]
        for viz in cfg.sim.visualizer_cfgs:
            viz.headless = args.headless
    trial = dict(
        trial_id=args.trial,
        initial_x_m=0,
        initial_y_m=0,
        initial_yaw_rad=0,
        command_start=[0.15, 0, 0],
        command_end=[0.1, 0.05, 0.25],
        switch_time_s=2.5,
        mode="policy",
        action_noise_amplitude=0,
    )
    if args.schedule:
        trial = next(t for t in json.loads(args.schedule.read_text())["trials"] if t["trial_id"] == args.trial)
    if args.command is not None:
        trial["command_start"] = list(args.command)
        trial["command_end"] = list(args.command)
    if args.command_end is not None:
        trial["command_end"] = list(args.command_end)
    if args.switch_time is not None:
        trial["switch_time_s"] = float(args.switch_time)
    if args.initial_pose is not None:
        trial["initial_x_m"], trial["initial_y_m"], trial["initial_yaw_rad"] = map(float, args.initial_pose)
    args.output.mkdir(parents=True, exist_ok=True)
    saved = json.loads(args.policy.with_name("wandb_config.json").read_text())
    launcher = dict(
        vars(args),
        visualizer=(args.visualizer or ["newton_rtx"]) if (args.video or not args.headless) else [],
    )
    with launch_simulation(cfg, launcher):
        env = RslRlVecEnvWrapper(gym.make(task, cfg=cfg), clip_actions=saved["train_cfg"].get("clip_actions"))
        raw, adapter = env.unwrapped, None
        video_proc = None
        original_hook = raw.recorder_manager.record_post_physics_decimation_step
        try:
            NewtonManager.set_decimation(1)
            raw._physics_handles_decimation = False
            torch.manual_seed(args.seed)
            env.reset()
            robot = raw.scene["robot"]
            init_joint_pos = robot.data.joint_pos.torch.clone()
            init_joint_vel = torch.zeros_like(robot.data.joint_vel.torch)
            settle_steps = (150 if args.checkpoint else 0) if args.settle_steps is None else int(args.settle_steps)
            if settle_steps > 0:
                robot_park = raw.scene["robot"]
                park_pose = robot_park.data.root_state_w.torch.clone()
                park_pose[:, :3] = raw.scene.env_origins + torch.tensor([0.0, 0.0, 2.5], device=raw.device)
                park_pose[:, 7:] = 0
                robot_park.write_root_state_to_sim(park_pose)
                raw.scene.write_data_to_sim()
                raw.sim.forward()
                zero_act = torch.zeros(args.num_envs, raw.action_manager.total_action_dim, device=raw.device)
                settle_policy_steps = max(1, (settle_steps + 3) // 4)
                with torch.inference_mode():
                    for _ in range(settle_policy_steps):
                        env.step(zero_act)
                wp.to_torch(NewtonManager.get_state_0().particle_qd).zero_()
                robot.write_joint_state_to_sim(init_joint_pos, init_joint_vel)
                raw.action_manager.reset()
                raw.observation_manager.reset()
            settled_sand_z = float(
                wp.to_torch(NewtonManager.get_state_0().particle_q)[:88360, 2].quantile(0.98).item()
            )
            spawn_z = 0.76 if settle_steps == 0 else (0.76 + min(0.0, settled_sand_z + 0.015))
            pose = robot.data.root_state_w.torch.clone()
            pose[:, :3] = raw.scene.env_origins + torch.tensor(
                [trial["initial_x_m"], trial["initial_y_m"], spawn_z], device=raw.device
            )
            yaw = torch.full((args.num_envs,), float(trial["initial_yaw_rad"]), device=raw.device)
            pose[:, 3:7] = quat_from_euler_xyz(yaw * 0, yaw * 0, yaw)
            pose[:, 7:] = 0
            robot.write_root_state_to_sim(pose)
            raw.scene.write_data_to_sim()
            raw.sim.forward()
            raw.scene.update(dt=0)
            obs = raw.observation_manager.compute()
            actor_cfg = saved["train_cfg"]["actor"].copy()
            assert actor_cfg.pop("class_name") == "MLPModel"
            actor = MLPModel(
                obs, saved["train_cfg"]["obs_groups"], "actor", raw.action_manager.total_action_dim, **actor_cfg
            ).to(raw.device)
            actor.load_state_dict(
                torch.load(args.policy, map_location=raw.device, weights_only=True)["actor_state_dict"]
            )
            actor.eval()
            model = NewtonManager.get_model()
            if args.checkpoint:
                # Loading the neural model must not alter the policy/noise RNG sequence.
                with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                    adapter = LearnedMaterial(
                        raw,
                        args.checkpoint,
                        iterations=args.iterations,
                        effective_mass_scale=args.effective_mass_scale,
                        contact_relaxation=args.contact_relaxation,
                        contact_solver=args.contact_solver,
                        active_k=args.active_k,
                    )
            feet = raw._foot_body_ids[0].cpu().numpy()
            trace, snapshots = [], []
            step = 0

            def record():
                state = NewtonManager.get_state_0()
                bq, q = state.body_q.numpy(), state.particle_q.numpy()
                wrench = raw._coupling_forces.numpy()[feet]
                fields = {"particle_position": q, "body_pose": bq, "foot_wrench": wrench}
                invalid = [name for name, value in fields.items() if not np.isfinite(value).all()]
                if invalid:
                    raise FloatingPointError(f"Nonfinite {invalid} at step {step}")
                trace.append(
                    dict(
                        time_s=step * 0.005,
                        foot_wrench=wrench,
                        root_pose=bq[0].copy(),
                        root_velocity=state.body_qd.numpy()[0].copy(),
                    )
                )
                if adapter and adapter.estimator is not None:
                    trace[-1]["contact_stats"] = np.stack([f["stats"].numpy() for f in adapter.estimator.feet])
                if step % 8 == 0:
                    snapshots.append(dict(time_s=step * 0.005, body_pose=bq.copy(), particle_position=q.copy()))

            def hook():
                nonlocal step
                original_hook()
                step += 1
                record()

            raw.recorder_manager.record_post_physics_decimation_step = hook
            record()
            term = raw.command_manager.get_term("base_velocity")
            noise = torch.zeros(args.num_envs, raw.action_manager.total_action_dim, device=raw.device)
            target = torch.zeros_like(noise)
            outcome, failure = "duration", None
            termination_terms = []
            diverse_presets = [
                ([1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]),
                ([-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]),
                ([0.7, 0.5, 0.0], [-0.7, -0.5, 0.0]),
                ([0.6, 0.0, 1.0], [0.6, 0.0, -1.0]),
            ]
            with torch.inference_mode():
                for k in range(round(args.duration / 0.02)):
                    if args.diverse_commands and args.num_envs > 1:
                        for e in range(args.num_envs):
                            c_start, c_end = diverse_presets[e % len(diverse_presets)]
                            c_cur = c_end if k * 0.02 >= trial["switch_time_s"] else c_start
                            term.vel_command_b[e] = torch.tensor(c_cur, device=raw.device)
                    else:
                        command = trial["command_end"] if k * 0.02 >= trial["switch_time_s"] else trial["command_start"]
                        term.vel_command_b[:] = torch.tensor(command, device=raw.device)
                    term.is_standing_env[:] = False
                    obs["command"][:] = term.command
                    action = actor(obs)
                    if k % 10 == 0:
                        target.uniform_(-1, 1)
                    noise.lerp_(target, 0.2)
                    if trial["mode"] == "policy-noise":
                        action += trial["action_noise_amplitude"] * noise
                    elif trial["mode"] == "random":
                        action = trial["action_noise_amplitude"] * noise
                    elif trial["mode"] == "zero":
                        action.zero_()
                    try:
                        obs, _, done, info = env.step(action)
                    except (FloatingPointError, RuntimeError) as error:
                        outcome, failure = "simulation_error", str(error)
                        break
                    if args.video and raw.sim.visualizers and (args.video_length is None or k < args.video_length):
                        viz_obj = raw.sim.visualizers[0]
                        NewtonManager._mark_transforms_changed()
                        NewtonManager._mark_particles_dirty()
                        raw.scene.update(0.02)
                        viz_obj._camera_tracking_step = -1
                        viz_obj.update_camera_tracking()
                        viz_obj._render_frame()
                        frame_rgb = np.ascontiguousarray(viz_obj._viewer.get_frame().numpy()[..., :3], dtype=np.uint8)
                        if video_proc is None:
                            vh, vw = frame_rgb.shape[:2]
                            video_proc = subprocess.Popen(
                                [
                                    "ffmpeg",
                                    "-y",
                                    "-loglevel",
                                    "error",
                                    "-f",
                                    "rawvideo",
                                    "-pix_fmt",
                                    "rgb24",
                                    "-s",
                                    f"{vw}x{vh}",
                                    "-r",
                                    "50",
                                    "-i",
                                    "-",
                                    "-an",
                                    "-c:v",
                                    "libx264",
                                    "-preset",
                                    "fast",
                                    "-crf",
                                    "20",
                                    "-pix_fmt",
                                    "yuv420p",
                                    "-movflags",
                                    "+faststart",
                                    str(args.output / "video.mp4"),
                                ],
                                stdin=subprocess.PIPE,
                            )
                        video_proc.stdin.write(frame_rgb.tobytes())
                    if bool(done.all()):
                        outcome = "native_termination"
                        termination_terms = [
                            name.removeprefix("Episode_Termination/")
                            for name, value in info.get("log", {}).items()
                            if name.startswith("Episode_Termination/") and float(value) > 0
                        ]
                        break
                    if k % 25 == 0:
                        print(
                            "EVALUATION",
                            json.dumps(dict(time_s=step * 0.005, learned_steps=adapter.steps if adapter else 0)),
                            flush=True,
                        )
            metadata = dict(
                trial=trial,
                seed=args.seed,
                outcome=outcome,
                duration_s=float(trace[-1]["time_s"]),
                requested_duration_s=args.duration,
                failure=failure,
                termination_terms=termination_terms,
                backend="particle_nerd" if adapter else "mpm",
                checkpoint=str(args.checkpoint),
                particle_count=model.particle_count,
                body_labels=list(model.body_label),
                estimator=adapter.metadata if adapter else None,
                learned_steps=adapter.steps if adapter else 0,
                history_initialization="constant velocity extrapolation from initial state",
                physics_hz=200,
                policy_hz=50,
                snapshots_hz=25,
                use_cuda_graph=cfg.sim.physics.use_cuda_graph,
            )
            for name, records in [("trace", trace), ("snapshots", snapshots)]:
                np.savez(
                    args.output / f"{name}.npz", **{key: np.stack([r[key] for r in records]) for key in records[0]}
                )
            (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
            print("EVALUATION_COMPLETE", json.dumps(dict(outcome=outcome, duration_s=trace[-1]["time_s"])), flush=True)
            if failure:
                raise RuntimeError(f"Saved valid rollout prefix before simulation failure: {failure}")
        finally:
            if video_proc is not None:
                video_proc.stdin.close()
                video_proc.wait()
            raw.recorder_manager.record_post_physics_decimation_step = original_hook
            if adapter:
                adapter.close()
            env.close()


if __name__ == "__main__":
    main()
