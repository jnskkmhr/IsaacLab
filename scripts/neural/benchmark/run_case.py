# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run one closed-loop benchmark case in Isaac Lab, with Particle NeRD or with MPM, and save the motion.

A case is an object (``box``, ``sphere``, ``cylinder``) and a motion on the example sand bed. The rigid object is
coupled to the sand both ways, so the sand's reaction acts back on it. The same case is run once per solver and
the two results are compared with ``kinetic_energy.py`` and ``surface_distance.py``.

Motions (the object is first held for ``--settle`` steps while the bed settles):

- ``push``: lowered into the bed, then dragged 0.9 m along +x by a PD-controlled force and torque.
- ``circle``: lowered onto a circle of radius 0.30 m and driven around it.
- ``stamp``: driven vertically at 2 Hz with a 12 cm stroke at a fixed position.
- ``drop``: held 0.40 m above the bed, then released with a sideways push of about 1 m/s and left free.

The result is ``<output>/<shape>-<motion>/<solver>.npz`` with the particle positions ``q`` [T, N, 3], the object
pose ``pose`` [T, 7] and the physics step of every saved frame ``step`` [T].

Run from the repository root, once per solver:

    uv run python scripts/neural/benchmark/run_case.py --shape box --motion push --solver nerd
    uv run python scripts/neural/benchmark/run_case.py --shape box --motion push --solver mpm

``--checkpoint`` and ``--sleep-speed`` change the Particle NeRD solver of the task; they have no effect with MPM.
MPM is not reproducible bit for bit once the object touches the sand, so two MPM runs differ slightly.
"""

import argparse
import math
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import warp as wp

from isaaclab.app import add_launcher_args, launch_simulation

from isaaclab_rl.entrypoints.common import (
    add_video_args,
    apply_video_recording,
    enable_cameras_for_video,
    pre_launch_video_config,
)

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli

SHAPES = ("box", "sphere", "cylinder")
MOTIONS = ("push", "circle", "stamp", "drop")
SOLVERS = ("nerd", "mpm")
HOVER_HEIGHT = 0.16
"""Height of the object centre above the bed before it is lowered [m]."""
CIRCLE_RADIUS = 0.30
CIRCLE_RATE = 2.0 * math.pi / 2.0
"""Angular rate on the circle [rad/s]: one lap in 2 s."""


def task_name(shape: str, solver: str) -> str:
    """Gym task of the example scene for this object and solver."""
    return f"IsaacContrib-ParticleNeRD-{shape.capitalize()}{'-MPM' if solver == 'mpm' else ''}-v0"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shape", choices=SHAPES, required=True)
    parser.add_argument("--motion", choices=MOTIONS, required=True)
    parser.add_argument("--solver", choices=SOLVERS, required=True)
    parser.add_argument("--output", type=Path, default=Path("logs/neural/benchmark"), help="Folder of the results.")
    parser.add_argument("--settle", type=int, default=600, help="Steps before the motion starts (not saved).")
    parser.add_argument("--steps", type=int, default=560, help="Steps of the motion; 560 steps are 2.8 s.")
    parser.add_argument("--every", type=int, default=4, help="Save every this many steps.")
    parser.add_argument("--checkpoint", default=None, help="Particle NeRD checkpoint instead of the task's default.")
    parser.add_argument("--sleep-speed", type=float, default=None, help="Particle sleeping threshold [m/s].")
    add_video_args(parser, action="the benchmark case")
    add_launcher_args(parser)
    args, hydra_args = setup_preset_cli(parser, argv)
    if args.solver == "mpm" and (args.checkpoint or args.sleep_speed is not None):
        parser.error("--checkpoint and --sleep-speed apply to the Particle NeRD solver, not to MPM")
    enable_cameras_for_video(args)
    sys.argv = [sys.argv[0]] + hydra_args
    torch.manual_seed(42)

    task = task_name(args.shape, args.solver)
    env_cfg, _ = resolve_task_config(task, "")
    env_cfg.scene.num_envs = 1
    env_cfg.episode_length_s = 60.0
    if args.checkpoint or args.sleep_speed is not None:
        neural = [e.solver_cfg for e in env_cfg.sim.physics.solver_cfg.entries if hasattr(e.solver_cfg, "checkpoint")]
        if len(neural) != 1:
            parser.error(f"expected one Particle NeRD solver in {task}, found {len(neural)}")
        if args.checkpoint:
            neural[0].checkpoint = args.checkpoint
        if args.sleep_speed is not None:
            neural[0].sleep_speed = args.sleep_speed
    case_dir = args.output / f"{args.shape}-{args.motion}"
    case_dir.mkdir(parents=True, exist_ok=True)

    from isaaclab_newton.physics import NewtonManager

    pre_launch_video_config(env_cfg, args)
    with launch_simulation(env_cfg, args):
        if args.video:
            apply_video_recording(env_cfg, str(case_dir), args, subdir=args.solver)
            for recorder in env_cfg.video_recorders:
                recorder.step_offset = args.settle
        env = gym.make(task, cfg=env_cfg)
        obs, _ = env.reset()
        base = env.unwrapped
        device = base.device
        act_dim = env.action_space.shape[-1]
        obj = base.scene[args.shape]
        origin = base.scene.env_origins
        scale = float(getattr(base.cfg.actions.wrench, "scale", 250.0))

        def particles():
            return wp.to_torch(NewtonManager.get_state_0().particle_q).view(1, -1, 3)[0].cpu().numpy().copy()

        def pose():
            offset = np.concatenate([origin[0].cpu().numpy(), [0, 0, 0, 0]])
            return obj.data.root_pose_w.torch[0].cpu().numpy() - offset

        def hold(position, velocity=None):
            """Wrench action of a PD controller that drives the object centre to ``position`` and keeps it upright."""
            pose_w = obj.data.root_pose_w.torch
            vel_w = obj.data.root_vel_w.torch
            pos = pose_w[:, :3] - origin
            quat = pose_w[:, 3:7]
            velocity = torch.zeros_like(pos) if velocity is None else velocity
            force = 3500.0 * (position - pos) + 150.0 * (velocity - vel_w[:, :3])
            force[:, 2] += 2.0 * 9.81
            sign = torch.where(quat[:, 3:4] >= 0, 1.0, -1.0)
            torque = -40.0 * 2.0 * quat[:, :3] * sign - 4.0 * vel_w[:, 3:6]
            action = torch.zeros((1, act_dim), device=device)
            action[:, :3] = force / scale
            if act_dim >= 6:
                action[:, 3:6] = torque / (0.04 * scale)
            return action

        if args.motion == "push":
            z_low = -0.01 if args.shape == "sphere" else -0.03
            start_xy = (obj.data.root_pose_w.torch[:, :2] - origin[:, :2]).clone()
            dt = base.step_dt

            def act(step, obs):
                if step == 0:  # the push starts from where the object is held
                    start_xy.copy_(obj.data.root_pose_w.torch[:, :2] - origin[:, :2])
                t = step * dt
                u = min(1.0, t / 0.40)
                blend = u * u * (3.0 - 2.0 * u)
                travel = max(0.0, min(1.0, (t - 0.40) / 2.00))
                x = start_xy[:, 0] * (1.0 - blend) + (-0.45) * blend + 0.90 * travel
                z = (1.0 - blend) * HOVER_HEIGHT + blend * z_low
                target = torch.stack([x, start_xy[:, 1], torch.full((1,), z, device=device)], dim=-1)
                target_vel = torch.zeros((1, 3), device=device)
                target_vel[:, 0] = (0.90 / 2.00) if (0.40 <= t <= 2.40) else 0.0
                target_vel[:, 2] = (z_low - HOVER_HEIGHT) * (6.0 * u * (1.0 - u) / 0.40) if t < 0.40 else 0.0
                return hold(target, target_vel)

            for _ in range(args.settle):
                with torch.inference_mode():
                    obs, *_ = env.step(act(0, obs))

        elif args.motion in ("circle", "stamp"):
            z_low = -0.01 if args.shape == "sphere" else -0.03
            start = (obj.data.root_pose_w.torch[:, :3] - origin)[0].clone()
            hover = torch.tensor([start[0].item(), start[1].item(), HOVER_HEIGHT], device=device)[None]
            for _ in range(args.settle):
                with torch.inference_mode():
                    obs, *_ = env.step(hold(hover))
            dt = base.step_dt

            def act(step, obs):
                t = step * dt
                if args.motion == "stamp":
                    frequency = 2.0
                    z_high = z_low + 0.12
                    phase = 2 * math.pi * frequency * t
                    z = z_high + (z_low - z_high) * 0.5 * (1 - math.cos(phase))
                    vz = (z_low - z_high) * 0.5 * math.sin(phase) * 2 * math.pi * frequency
                    target = torch.tensor([[hover[0, 0].item(), hover[0, 1].item(), z]], device=device)
                    target_vel = torch.tensor([[0.0, 0.0, vz]], device=device)
                else:
                    u = min(1.0, t / 0.40)
                    blend = u * u * (3 - 2 * u)
                    angle = CIRCLE_RATE * max(0.0, t - 0.40)
                    cx, cy = CIRCLE_RADIUS * math.cos(angle), CIRCLE_RADIUS * math.sin(angle)
                    x = hover[0, 0].item() * (1 - blend) + cx * blend
                    y = hover[0, 1].item() * (1 - blend) + cy * blend
                    z = HOVER_HEIGHT * (1 - blend) + z_low * blend
                    moving = t > 0.40
                    target_vel = torch.tensor(
                        [
                            [
                                (-CIRCLE_RADIUS * CIRCLE_RATE * math.sin(angle)) if moving else 0.0,
                                (CIRCLE_RADIUS * CIRCLE_RATE * math.cos(angle)) if moving else 0.0,
                                0.0,
                            ]
                        ],
                        device=device,
                    )
                    target = torch.tensor([[x, y, z]], device=device)
                return hold(target, target_vel)

        else:
            release = torch.tensor([-0.3, 0.0, 0.40], device=device)[None]
            for _ in range(args.settle):
                with torch.inference_mode():
                    obs, *_ = env.step(hold(release))

            def act(step, obs):
                action = torch.zeros((1, act_dim), device=device)
                if step < 10:  # 40 N for 0.05 s on the 2 kg object: about 1 m/s along +x
                    action[:, 0] = 40.0 / scale
                return action

        frames, poses, steps = [particles()], [pose()], [0]
        for step in range(args.steps):
            with torch.inference_mode():
                obs, *_ = env.step(act(step, obs))
            if (step + 1) % args.every == 0:
                frames.append(particles())
                poses.append(pose())
                steps.append(step + 1)
        result = case_dir / f"{args.solver}.npz"
        np.savez_compressed(result, q=np.stack(frames).astype(np.float32), pose=np.stack(poses), step=np.array(steps))
        print(f"Saved {result}: {len(frames)} frames, object z at the end {poses[-1][2]:.4f} m", flush=True)
        env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
