# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Validate flat mixed terrain and independently timed constant torso-force pulses."""

import gymnasium as gym
import numpy as np
import pytest
import torch
import warp as wp

from isaaclab.app import launch_simulation

from isaaclab_tasks.utils import resolve_task_config


def test_flat_force_pulses():
    if not torch.cuda.is_available():
        pytest.skip("Particle NeRD integration requires CUDA")
    task = "IsaacContrib-Velocity-G1-29dof-WBC-ParticleNeRD-MovingPatch-MixedFlatTerrain"
    cfg, _ = resolve_task_config(task, "rsl_rl_cfg_entry_point", overrides=["env.scene.num_envs=4"])
    cfg.sim.visualizer_cfgs = []
    cfg.video_recorders = []
    cfg.sim.physics.solver_cfg.entries[1].solver_cfg.compile_model = False
    cfg.sim.physics.solver_cfg.entries[1].solver_cfg.active_k = 512
    with launch_simulation(cfg):
        env = gym.make(task, cfg=cfg).unwrapped
        try:
            obs, _ = env.reset(seed=42)
            robot = env.scene["robot"]
            torso = robot.find_bodies("torso_link")[0][0]
            composer = robot.permanent_wrench_composer
            torch.testing.assert_close(
                env.command_manager.get_term("base_height").terrain_height,
                torch.zeros(4, device=env.device),
                atol=1e-5,
                rtol=0,
            )
            assert env.scene.terrain.mpm_env_mask.tolist() == [True, True, False, False]
            series = []
            for _ in range(1700):
                env.event_manager.apply(mode="interval", dt=env.step_dt)
                force = wp.to_torch(composer.global_force_at_com_w).clone()
                assert (force[:, :, 1:] == 0).all()
                assert (force[:, torso, 0] <= 0).all() and (force[:, torso, 0] >= -50).all()
                other = force.clone()
                other[:, torso] = 0
                assert (other == 0).all()
                series.append(force[:, torso, 0].cpu().numpy())
            series = np.asarray(series)
            starts_per_env = []
            for i in range(4):
                active = series[:, i] < 0
                starts = np.flatnonzero(np.diff(np.r_[False, active].astype(int)) == 1)
                ends = np.flatnonzero(np.diff(np.r_[active, False].astype(int)) == -1) + 1
                assert len(starts) >= 2
                assert 10 - env.step_dt <= (starts[0] + 1) * env.step_dt <= 15 + env.step_dt
                assert np.all(np.diff(starts) * env.step_dt >= 10 - env.step_dt)
                assert np.all(np.diff(starts) * env.step_dt <= 15 + env.step_dt)
                for start, end in zip(starts, ends):
                    if end < len(series):
                        assert end - start == 150, (start, end)
                        assert np.all(series[start:end, i] == series[start, i])
                starts_per_env.append(starts.tolist())
            assert len({x[0] for x in starts_per_env}) > 1
            print("PULSE_TIMING", starts_per_env, "all complete pulses exactly 3.00s; Fx in [-50, 0] N")
            # Reset one active robot, preserving external forces in the other worlds.
            for _ in range(800):
                env.event_manager.apply(mode="interval", dt=env.step_dt)
                before = wp.to_torch(composer.global_force_at_com_w).clone()
                if before[0, torso, 0] < 0:
                    env.event_manager.reset(torch.tensor([0], device=env.device))
                    after = wp.to_torch(composer.global_force_at_com_w).clone()
                    assert (after[0] == 0).all()
                    torch.testing.assert_close(after[1:], before[1:])
                    break
            else:
                raise AssertionError("No active pulse found for reset check")
            for _ in range(10):
                obs, rewards, *_ = env.step(torch.zeros((4, 15), device=env.device))
                assert torch.isfinite(rewards).all()
                assert all(torch.isfinite(v).all() for v in obs.values())
            print("RESET_AND_STEP_CHECK passed")
        finally:
            env.close()
