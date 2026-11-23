# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Checkpoint-free interactive and recorded playback for ParticleNeRDLocal tasks."""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import math
import os
import sys
from pathlib import Path

import gymnasium as gym
import torch

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.utils import validate

from isaaclab_rl.entrypoints.common import (
    add_video_args,
    apply_env_overrides,
    apply_video_recording,
    close_env,
    enable_cameras_for_video,
    normalize_task_name,
    pre_launch_video_config,
    video_playback_steps,
)

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli

logger = logging.getLogger(__name__)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive playback for ParticleNeRDLocal tasks.")
    parser.add_argument(
        "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
    )
    parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
    parser.add_argument("--task", type=str, default=None, help="Name of the task.")
    parser.add_argument(
        "--max_steps", type=int, default=None, help="Number of environment steps to run. Runs unbounded when omitted."
    )
    add_video_args(parser, action="the ParticleNeRDLocal playback run")
    add_launcher_args(parser)
    parser.set_defaults(device=None)
    args_cli, hydra_args = setup_preset_cli(parser, argv)
    enable_cameras_for_video(args_cli)
    sys.argv = [sys.argv[0]] + hydra_args
    return args_cli


def _build_task_controller(env: gym.Env, task_name: str):
    base_env = env.unwrapped
    device = base_env.device
    num_envs = base_env.num_envs
    act_dim = env.action_space.shape[-1]
    dt = base_env.step_dt

    if "G1" in task_name:
        ckpt_dir = (
            Path(__file__).resolve().parents[2]
            / "g1_29dof_particle_nerd"
            / "benchmark"
            / "data"
            / "checkpoints"
            / "jqur5jhm"
        )
        ckpt_path = ckpt_dir / "model_1999.pt"
        cfg_path = ckpt_dir / "training_config.json"
        if ckpt_path.is_file() and cfg_path.is_file():
            from rsl_rl.models import MLPModel

            saved = json.loads(cfg_path.read_text())
            actor_cfg = saved["train_cfg"]["actor"].copy()
            actor_cfg.pop("class_name", None)
            obs_dict = base_env.observation_manager.compute()
            actor = MLPModel(obs_dict, saved["train_cfg"]["obs_groups"], "actor", act_dim, **actor_cfg).to(device)
            checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
            actor.load_state_dict(checkpoint["actor_state_dict"], strict=True)
            actor.eval()

            def _g1_policy(step: int, obs):
                return actor(obs)

            return _g1_policy

    if "Franka" in task_name:
        phase_offsets = torch.linspace(0.0, 0.45, num_envs, device=device)

        def _franka_policy(step: int, _obs):
            t = step * dt + phase_offsets
            actions = torch.zeros((num_envs, act_dim), device=device)
            actions[:, 0] = 1.35 * torch.sin(1.8 * t)
            actions[:, 1] = 0.55 + 0.20 * torch.cos(3.6 * t)
            actions[:, 2] = 0.55 * torch.cos(1.8 * t)
            actions[:, 3] = -0.45 + 0.25 * torch.sin(3.6 * t)
            actions[:, 5] = 0.40 * torch.cos(1.8 * t)
            return actions

        return _franka_policy

    if "Go2" in task_name:
        robot = base_env.scene["robot"]
        fl_thigh = robot.find_joints("FL_thigh_joint")[0][0]
        fr_thigh = robot.find_joints("FR_thigh_joint")[0][0]
        rl_thigh = robot.find_joints("RL_thigh_joint")[0][0]
        rr_thigh = robot.find_joints("RR_thigh_joint")[0][0]
        fl_calf = robot.find_joints("FL_calf_joint")[0][0]
        fr_calf = robot.find_joints("FR_calf_joint")[0][0]
        rl_calf = robot.find_joints("RL_calf_joint")[0][0]
        rr_calf = robot.find_joints("RR_calf_joint")[0][0]
        phase_offsets = torch.linspace(0.0, 0.5 * math.pi, num_envs, device=device)

        def _go2_policy(step: int, _obs):
            t = step * dt
            actions = torch.zeros((num_envs, act_dim), device=device)
            if t < 0.25:
                return actions
            phi = 2.0 * math.pi * 2.4 * (t - 0.25) + phase_offsets
            s_a = torch.sin(phi)
            s_b = torch.sin(phi + math.pi)
            actions[:, fl_thigh] = -0.95 * s_a
            actions[:, rr_thigh] = -0.95 * s_a
            actions[:, fr_thigh] = -0.95 * s_b
            actions[:, rl_thigh] = -0.95 * s_b
            actions[:, fl_calf] = -1.10 * torch.clamp(s_a, min=0.0)
            actions[:, rr_calf] = -1.10 * torch.clamp(s_a, min=0.0)
            actions[:, fr_calf] = -1.10 * torch.clamp(s_b, min=0.0)
            actions[:, rl_calf] = -1.10 * torch.clamp(s_b, min=0.0)
            return actions

        return _go2_policy

    # Box, Sphere, Cylinder: lower into the sand bed and plow horizontally across the tray via WrenchAction.
    obj_key = "sphere" if "Sphere" in task_name else ("cylinder" if "Cylinder" in task_name else "box")
    obj = base_env.scene[obj_key]
    origins = base_env.scene.env_origins
    p0_xy = (obj.data.root_pos_w.torch[:, :2] - origins[:, :2]).clone()
    z_target = -0.010 if obj_key == "sphere" else -0.030
    scale = float(getattr(base_env.cfg.actions.wrench, "scale", 250.0))

    def _object_wrench_policy(step: int, _obs):
        if step == 0:
            p0_xy.copy_(obj.data.root_pos_w.torch[:, :2] - origins[:, :2])
        t = step * dt
        pose_w = obj.data.root_pose_w.torch
        vel_w = obj.data.root_vel_w.torch
        pos = pose_w[:, :3] - origins
        quat = pose_w[:, 3:7]  # xyzw
        lin_v = vel_w[:, :3]
        ang_v = vel_w[:, 3:6]

        u = min(1.0, t / 0.40)
        blend = u * u * (3.0 - 2.0 * u)
        travel = max(0.0, min(1.0, (t - 0.40) / 2.00))
        x_start = p0_xy[:, 0] * (1.0 - blend) + (-0.45) * blend
        x_des = x_start + 0.90 * travel
        vx_des = (0.90 / 2.00) if (0.40 <= t <= 2.40) else 0.0
        z_des = (1.0 - blend) * 0.16 + blend * z_target
        vz_des = (z_target - 0.16) * (6.0 * u * (1.0 - u) / 0.40) if t < 0.40 else 0.0

        p_des = torch.stack([x_des, p0_xy[:, 1], torch.full((num_envs,), z_des, device=device)], dim=-1)
        v_des = torch.zeros((num_envs, 3), device=device)
        v_des[:, 0] = vx_des
        v_des[:, 2] = vz_des
        f_cmd = 3500.0 * (p_des - pos) + 150.0 * (v_des - lin_v)
        f_cmd[:, 2] += 2.0 * 9.81

        qw_sign = torch.where(quat[:, 3:4] >= 0.0, 1.0, -1.0)
        rot_err = 2.0 * quat[:, :3] * qw_sign
        tau_cmd = -40.0 * rot_err - 4.0 * ang_v

        actions = torch.zeros((num_envs, act_dim), device=device)
        actions[:, :3] = f_cmd / scale
        if act_dim >= 6:
            actions[:, 3:6] = tau_cmd / (0.04 * scale)
        return actions

    return _object_wrench_policy


def run(argv: list[str] | None = None) -> int:
    """Run interactive or video-recorded playback for a ParticleNeRDLocal task."""
    args_cli = _parse_args(argv)
    torch.manual_seed(42)

    env_cfg, _ = resolve_task_config(args_cli.task, "")
    apply_env_overrides(args_cli, env_cfg)
    pre_launch_video_config(env_cfg, args_cli)
    try:
        validate(env_cfg)
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"Invalid environment configuration: {exc}") from None

    with launch_simulation(env_cfg, args_cli), contextlib.ExitStack() as cleanup:
        log_dir = os.path.abspath(os.path.join("logs", "particle_nerd_local", normalize_task_name(args_cli.task)))
        apply_video_recording(env_cfg, log_dir, args_cli, subdir="play")
        env = gym.make(args_cli.task, cfg=env_cfg)
        cleanup.callback(lambda: close_env(env))
        obs, _ = env.reset()
        controller = _build_task_controller(env, args_cli.task)

        budgets = [n for n in (args_cli.max_steps, video_playback_steps(args_cli, env_cfg)) if n is not None]
        max_steps = min(budgets, default=None)

        sim = env.unwrapped.sim
        step = 0
        while sim.is_running():
            if max_steps is not None and step >= max_steps:
                break
            with torch.inference_mode():
                actions = controller(step, obs)
                obs, _, _, _, _ = env.step(actions)
            step += 1
    return 0
