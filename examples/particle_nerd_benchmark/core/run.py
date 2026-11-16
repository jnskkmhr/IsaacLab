# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run one manager-based native MPM benchmark and record the same rollout as video/data."""

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from isaaclab.app import add_launcher_args, launch_simulation

from .motions import MOTIONS, SETTLE_SECONDS, TERRAINS, object_command
from .terrain import PARTICLE_JITTER, PARTICLE_SEED, SPACING, VOXEL_SIZE


def create_config(args):
    """Compose a standalone task and its recording camera."""
    from isaaclab_visualizers.newton import NewtonGLVisualizerCfg

    from isaaclab.visualizers import WindowCfg

    from isaaclab_tasks.utils import load_cfg_from_registry

    from .case_config import make_config

    if args.task:
        cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
    else:
        cfg = make_config(args.motion, args.terrain, geometry=args.geometry)
    cfg.sim.device = args.device
    bed_size = {"franka": (1.2, 1.0, 0.25), "g1": (3.6, 1.6, 0.30), "quadruped": (2.0, 1.2, 0.30)}.get(
        args.motion, (2.0, 1.2, 0.30)
    )
    cfg.sim.visualizer_cfgs = (
        []
        if args.no_video and not args.interactive
        else [
            NewtonGLVisualizerCfg(
                headless=not args.interactive or args.offscreen,
                show_particles=True,
                particle_color=(0.72, 0.59, 0.36),
                eye=(1.65, -2.0, 1.55),
                lookat=(0.0, 0.0, -0.05),
                window=WindowCfg(size=(1280, 720)),
                streaming_view=False,
            )
        ]
    )
    if cfg.sim.visualizer_cfgs:
        cfg.sim.visualizer_cfgs[0].focal_length = 24.0
        if args.motion == "g1":
            cfg.sim.visualizer_cfgs[0].eye = (3.2, -4.4, 3.0)
            cfg.sim.visualizer_cfgs[0].lookat = (0.0, 0.0, 0.35)
    if args.no_video and not args.interactive:
        args.visualizer = []
    return cfg, bed_size


def _parse_args(argv: list[str] | None, interactive: bool) -> argparse.Namespace:
    """Parse standalone collection or registered playback options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion", choices=(*MOTIONS, "franka", "quadruped", "g1"), default="horizontal")
    parser.add_argument("--terrain", choices=TERRAINS, default="flat")
    parser.add_argument("--geometry", choices=("box", "sphere", "cylinder"))
    parser.add_argument("--duration", type=float, default=6.0)
    parser.add_argument("--task", help="Registered benchmark task ID")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--video", action="store_true", help="Save the displayed simulation as MP4")
    parser.add_argument("--real-time", action="store_true", help="Pace playback to simulation time when possible")
    parser.add_argument("--output", type=Path, required=not interactive, help="Save aligned HDF5 and provenance here")
    parser.add_argument(
        "--checkpoint", type=Path, help="G1 policy checkpoint; training_config.json must be alongside it"
    )
    parser.add_argument("--no-video", action="store_true")
    add_launcher_args(parser)
    parser.set_defaults(visualizer=[] if interactive else ["newton_gl"])
    args = parser.parse_args(argv)
    args.interactive = interactive
    args.offscreen = not bool(args.visualizer)
    if args.video and args.offscreen:
        args.visualizer = ["newton_gl"]
    if args.num_envs != 1:
        parser.error("These recorded reference examples require --num_envs 1")
    if args.task:
        import gymnasium as gym

        from .. import config  # noqa: F401

        spec = gym.spec(args.task)
        if spec.kwargs.get("play_entry_point") != "particle_nerd_benchmark.core.play":
            parser.error("--task must name a registered particle_nerd_benchmark case")
        args.motion = spec.kwargs["benchmark_motion"]
        args.terrain = spec.kwargs["benchmark_terrain"]
        args.geometry = spec.kwargs.get("benchmark_geometry")
    elif interactive:
        parser.error("--task is required for isaaclab play")
    args.record_data = args.output is not None
    if interactive:
        args.no_video = not args.video
        if args.video and args.output is None:
            args.output = Path("logs/particle_nerd_benchmark") / (args.task + time.strftime("_%Y%m%d_%H%M%S"))
    args.save_run = args.output is not None
    if (
        not np.isfinite(args.duration)
        or args.duration <= 0
        or not np.isclose(args.duration * 50, round(args.duration * 50))
    ):
        parser.error("Duration must be positive and an integer number of 50 fps frames")
    if args.save_run and args.output.exists():
        parser.error("Output already exists; use a new run directory to preserve provenance")
    if args.motion == "g1" and (
        args.checkpoint is None
        or not args.checkpoint.is_file()
        or not (args.checkpoint.parent / "training_config.json").is_file()
    ):
        parser.error("G1 requires --checkpoint and its adjacent training_config.json")
    return args


def _command(env, args, step, obs, actor, controller):
    """Evaluate the original controller for one physical benchmark step."""
    t = step * env.physics_dt
    if args.motion == "g1":
        action = actor(obs)
    elif args.motion == "quadruped":
        action = controller.command(t)
    elif args.motion == "franka":
        progress = np.clip(t / 6.0, 0.0, 1.0)
        x = -0.34 + 0.40 * progress
        y = -0.22 * np.cos(2.0 * np.pi * progress)
        z = -0.06
        # Command the fingertip frame in robot base coordinates (base at [-0.72, 0.0, 0.05]).
        action = torch.tensor([[x + 0.72, y, z - 0.05, 1.0, 0.0, 0.0, 0.0]], device=env.device)
    elif args.motion == "projectile":
        action = torch.zeros((1, 6), device=env.device)
        if t < SETTLE_SECONDS:
            action[:, 2] = env.cfg.scene.object.spawn.mass_props.mass * 9.81
        if step == round(SETTLE_SECONDS / env.physics_dt):
            env.scene["object"].write_root_velocity_to_sim_index(
                root_velocity=torch.tensor([[1.0, 0.0, 0.4, 0.0, 1.0, 0.0]], device=env.device)
            )
    else:
        action = torch.as_tensor(object_command(t, args.motion, args.terrain, env.physics_dt), device=env.device)[None]
    return action


def run(argv: list[str] | None = None, *, interactive: bool = False) -> None:
    """Play or collect one registered physical benchmark with its original controller."""
    args = _parse_args(argv, interactive)
    record_data = args.record_data
    save_run = args.save_run
    from isaaclab.envs import ManagerBasedRLEnv

    from .provenance import sha256
    from .recording import Recorder

    cfg, bed_size = create_config(args)
    if save_run:
        args.output.mkdir(parents=True)
    package = Path(__file__).parents[1]
    if save_run:
        for source in package.rglob("*.py"):
            target = args.output / "source" / source.relative_to(package)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    steps = round(args.duration / cfg.sim.dt)
    metadata = {
        "motion": args.motion,
        "terrain": args.terrain,
        "geometry": args.geometry or ("sphere" if args.motion == "rolling" else "box"),
        "solver": "Implicit MPM",
        "physics_dt_s": cfg.sim.dt,
        "video_fps": 50,
        "duration_s": steps * cfg.sim.dt,
        "settling_time_s": SETTLE_SECONDS if args.motion in MOTIONS else 0.0,
        "seed": cfg.seed,
        "voxel_size_m": VOXEL_SIZE,
        "particle_spacing_m": SPACING,
        "particle_initialization": {
            "distribution": "Uniform coordinate jitter, truncated to the initial bed",
            "max_offset_per_axis_m": PARTICLE_JITTER,
            "seed": PARTICLE_SEED,
        },
        "tray_wall_top_z_m": cfg.scene.wall_x0.init_state.pos[2] + cfg.scene.wall_x0.spawn.size[2] / 2,
        "bed_size_m": list(bed_size),
        "material": cfg.scene.sand.spawn.material.to_dict(),
        "interacting_asset": (
            cfg.scene.robot if args.motion in ("g1", "franka", "quadruped") else cfg.scene.object
        ).to_dict(),
        "controller": {
            "projectile": "Free projectile after release",
            "franka": "Franka damped least-squares IK",
            "quadruped": "Go2 free-base crawl with leg IK and PD joint drives",
            "g1": "G1 closed-loop MPM-trained RL policy",
        }.get(args.motion, "Prescribed kinematic collider"),
        "physics": cfg.sim.physics.to_dict(),
        "control_dt_s": cfg.sim.dt * cfg.decimation,
        "source_files": {
            str(f.relative_to(Path(__file__).parents[1])): sha256(f) for f in Path(__file__).parents[1].rglob("*.py")
        },
        "command": " ".join(sys.argv),
        "status": "running",
    }
    if save_run:
        (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str))
    encoder = None
    recorder = None
    original_update = None
    with launch_simulation(cfg, args):
        if args.task:
            import gymnasium as gym

            env = gym.make(args.task, cfg=cfg).unwrapped
        else:
            env = ManagerBasedRLEnv(cfg)
        try:
            with torch.inference_mode():
                obs, _ = env.reset()
                actor = controller = None
                if args.motion == "quadruped":
                    from .quadruped_controller import CrawlController

                    controller = CrawlController(env)
                if args.motion == "g1":
                    from rsl_rl.models import MLPModel

                    saved = json.loads((args.checkpoint.parent / "training_config.json").read_text())
                    actor_cfg = saved["train_cfg"]["actor"].copy()
                    if actor_cfg.pop("class_name") != "MLPModel":
                        raise ValueError("Unsupported G1 actor class")
                    actor = MLPModel(
                        obs, saved["train_cfg"]["obs_groups"], "actor", env.action_manager.total_action_dim, **actor_cfg
                    ).to(env.device)
                    checkpoint = torch.load(args.checkpoint, map_location=env.device, weights_only=True)
                    actor.load_state_dict(checkpoint["actor_state_dict"], strict=True)
                    actor.eval()
                    metadata.update(
                        policy_checkpoint=str(args.checkpoint.resolve()),
                        policy_sha256=sha256(args.checkpoint),
                        policy_iteration=int(checkpoint["iter"]),
                        controller="G1 closed-loop MPM-trained RL policy",
                        command_velocity_m_s=cfg.forward_speed,
                    )
                if record_data:
                    recorder = Recorder(
                        env, args.output / "trajectory.h5", steps, check_tray_contacts=args.motion == "franka"
                    )
                    recorder.record(0)
                physics_index = 0
                original_update = env.scene.update

                def record_physics(dt):
                    nonlocal physics_index
                    original_update(dt)
                    physics_index += 1
                    if recorder:
                        recorder.record(physics_index)

                env.scene.update = record_physics
                video_steps = []
                collection_start = time.perf_counter()
                for control_step in range(steps // cfg.decimation):
                    control_start = time.perf_counter()
                    if not env.sim.is_running():
                        break
                    step = control_step * cfg.decimation
                    action = _command(env, args, step, obs, actor, controller)
                    obs, _, _, _, _ = env.step(action)
                    if not args.no_video and physics_index % 4 == 0:
                        frame = env.sim._visualizers[0].render_rgb_array()
                        if frame is None:
                            raise RuntimeError("Visualizer did not produce a video frame")
                        if encoder is None:
                            h, w = frame.shape[:2]
                            encoder = subprocess.Popen(
                                [
                                    "ffmpeg",
                                    "-v",
                                    "error",
                                    "-y",
                                    "-f",
                                    "rawvideo",
                                    "-pix_fmt",
                                    "rgb24",
                                    "-s",
                                    f"{w}x{h}",
                                    "-r",
                                    "50",
                                    "-i",
                                    "-",
                                    "-an",
                                    "-c:v",
                                    "libx264",
                                    "-crf",
                                    "20",
                                    "-pix_fmt",
                                    "yuv420p",
                                    "-movflags",
                                    "+faststart",
                                    str(args.output / "simulation.mp4"),
                                ],
                                stdin=subprocess.PIPE,
                            )
                        encoder.stdin.write(frame[:, :, :3].astype(np.uint8).tobytes())
                        video_steps.append(physics_index)
                    if args.real_time:
                        time.sleep(max(0.0, env.step_dt - (time.perf_counter() - control_start)))
                    if physics_index % 200 == 0:
                        print(f"Simulated {physics_index * cfg.sim.dt:.1f} s / {steps * cfg.sim.dt:.1f} s", flush=True)
                from isaaclab_newton.physics import NewtonManager

                model = NewtonManager.get_model()
                metadata.update(
                    particle_count=model.particle_count,
                    body_names=model.body_label,
                    frame_state_indices=video_steps,
                    collection_wall_seconds=time.perf_counter() - collection_start,
                    timing_scope="Collection includes state downloads and video encoding; not solver performance",
                    status="complete" if physics_index == steps else "interrupted",
                    recorded_physics_steps=physics_index,
                )
                env.scene.update = original_update
                if recorder:
                    recorder.close()
                if encoder:
                    encoder.stdin.close()
                    if encoder.wait() != 0:
                        raise RuntimeError("Video encoding failed")
                    encoder = None
                metadata["artifacts"] = {
                    name: sha256(args.output / name)
                    for name in (
                        *(("trajectory.h5",) if record_data else ()),
                        *(() if args.no_video else ("simulation.mp4",)),
                    )
                }
                if save_run:
                    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str))
        finally:
            if original_update is not None:
                env.scene.update = original_update
            if recorder is not None:
                recorder.close()
            if encoder:
                encoder.stdin.close()
                encoder.wait()
            env.close()
    outcome = "Completed" if metadata["status"] == "complete" else "Stopped early"
    print(f"{outcome} {args.task or args.motion}" + (f" → {args.output}" if save_run else ""), flush=True)


if __name__ == "__main__":
    run()
