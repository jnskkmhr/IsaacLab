# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Direction semantics and joint foot/object particle coupling at the task boundary."""

import gymnasium as gym
import pytest
import torch
import warp as wp
from isaaclab_newton.physics import NewtonManager

from isaaclab.app import launch_simulation

from isaaclab_tasks.utils import resolve_task_config


def test_direction_and_object_coupling():
    if not torch.cuda.is_available():
        pytest.skip("Particle NeRD requires CUDA")
    task = "IsaacContrib-Push-G1-29dof-ParticleNeRD-MovingPatch-ObjectPush-Play"
    cfg, _ = resolve_task_config(task, "rsl_rl_cfg_entry_point", overrides=[])
    cfg.scene.num_envs = 2
    cfg.sim.visualizer_cfgs = []
    cfg.video_recorders = []
    solver_cfg = cfg.sim.physics.solver_cfg.entries[1].solver_cfg
    solver_cfg.compile_model = False
    solver_cfg.active_k = 1024
    with launch_simulation(cfg):
        env = gym.make(task, cfg=cfg).unwrapped
        try:
            obs, _ = env.reset(seed=42)
            # Independently measured source checkpoint contracts: 799->15 and 76->14.
            assert obs["locomotion_teacher_obs"].shape == (2, 790)
            assert obs["locomotion_teacher_command"].shape == (2, 9)
            assert obs["manipulation_teacher_obs"].shape == (2, 76)
            command = env.command_manager.get_term("object_direction")
            command.direction[:] = torch.tensor([[1.0, 0.0], [0.0, 0.0]], device=env.device)
            command._update_command()
            torch.testing.assert_close(
                command.velocity, torch.tensor([[0.8, 0.0, 0.0], [0.0, 0.0, 0.0]], device=env.device)
            )
            neutral = torch.tensor(command.cfg.neutral_hand_poses, device=env.device)
            torch.testing.assert_close(command.hand_pose[1], neutral)
            assert command.pitch[1, 0] == 0
            assert command.height[1, 0] == 0.75
            command.direction[:] = torch.tensor([[0.0, 1.0], [0.0, -1.0]], device=env.device)
            command._update_command()
            torch.testing.assert_close(command.velocity[:, 2], torch.tensor([0.5, -0.5], device=env.device))
            command.direction.zero_()
            command._update_command()
            solver = NewtonManager._solver._entries["sand"].solver
            # Removing the Object proxy, or grouping three colliders as two, breaks this simulation.
            assert solver.collider_ids.shape == (2, 3)
            contact_seen = torch.zeros(2, dtype=torch.bool, device=env.device)
            for _ in range(25):
                obs, reward, done, _, _ = env.step(torch.zeros((2, 29), device=env.device))
                assert torch.isfinite(reward).all()
                assert all(torch.isfinite(value).all() for value in obs.values())
                assert not done.any()
                wrench = wp.to_torch(solver.estimator.wrench).reshape(2, 3, 6)
                assert torch.isfinite(wrench).all()
                contact_seen |= wrench[:, 2, :3].norm(dim=-1) > 1
            assert contact_seen.all(), "Both objects must receive a nonzero reaction from the particle solver"
            other_history = solver.history[solver.capacity :].clone()
            other_object = env.scene["object"].data.root_state_w.torch[1].clone()
            env._reset_idx(torch.tensor([0], device=env.device))
            torch.testing.assert_close(env.scene["object"].data.root_state_w.torch[1], other_object)
            torch.testing.assert_close(solver.history[solver.capacity :], other_history)
            assert env.object_contact_force[0].abs().max() == 0
            assert torch.isfinite(wp.to_torch(NewtonManager.get_state_0().particle_q)).all()
        finally:
            env.close()
