# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Lower-body actions and height commands coexist with independently randomized arms."""

import math

import gymnasium as gym
import pytest
import torch
from tensordict import TensorDict

from isaaclab.app import launch_simulation
from isaaclab.utils.math import quat_from_euler_xyz

from isaaclab_contrib.mdp import compute_mirrored_states

from isaaclab_tasks.contrib.g1_29dof_wbc_particle_nerd_moving_patch.env_cfg.terrain_cfg import FLAT_TERRAINS_CFG
from isaaclab_tasks.utils import resolve_task_config


@pytest.mark.parametrize(
    "task",
    [
        "IsaacContrib-Velocity-Sand-G1-29dof-WBC-ParticleNeRD-MovingPatch",
        "IsaacContrib-Velocity-G1-29dof-WBC-ParticleNeRD-MovingPatch-MixedTerrain",
    ],
)
def test_lower_body_actions_arm_randomization_and_regularization(task):
    if not torch.cuda.is_available():
        pytest.skip("Particle NeRD integration requires CUDA")
    cfg, _ = resolve_task_config(task, "rsl_rl_cfg_entry_point", overrides=["env.scene.num_envs=2"])
    cfg.sim.visualizer_cfgs = []
    cfg.video_recorders = []
    cfg.scene.terrain.terrain_generator = FLAT_TERRAINS_CFG.replace(num_rows=2, num_cols=2)
    cfg.actions.random_arms.sample_range = (0.02, 0.04)
    cfg.events.push_robot = None
    cfg.events.reset_base.params = {"pose_range": {}, "velocity_range": {}}
    cfg.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
    cfg.sim.physics.solver_cfg.entries[1].solver_cfg.compile_model = False
    cfg.sim.physics.solver_cfg.entries[1].solver_cfg.active_k = 512
    with launch_simulation(cfg):
        env = gym.make(task, cfg=cfg).unwrapped
        try:
            obs, _ = env.reset()
            assert env.action_manager.total_action_dim == 15
            robot = env.scene["robot"]
            arms = env.action_manager.get_term("random_arms")
            arm_ids, _ = robot.find_joints(cfg.actions.random_arms.joint_names, preserve_order=True)
            lower_ids, _ = robot.find_joints(cfg.actions.joint_pos.joint_names, preserve_order=True)
            start = robot.data.joint_pos.torch[:, arm_ids].clone()
            targets = arms.processed_actions.clone()
            action = torch.zeros((2, 15), device=env.device)
            for _ in range(20):
                obs, reward, terminated, truncated, _ = env.step(action)
                assert torch.isfinite(reward).all()
                assert all(torch.isfinite(value).all() for value in obs.values())
                assert not (terminated | truncated).any()
            assert (arms.processed_actions - targets).abs().max() > 1e-3
            assert (robot.data.joint_pos.torch[:, arm_ids] - start).abs().max() > 1e-3
            assert not torch.allclose(arms.processed_actions[0], arms.processed_actions[1])
            height_command = env.command_manager.get_command("base_height")
            assert height_command.shape == (2, 1)
            # Both flat sand and flat rigid terrain have their initial surface at world Z=0.
            # Referencing the support mesh would incorrectly return -0.3 in sand worlds.
            torch.testing.assert_close(
                env.command_manager.get_term("base_height").terrain_height,
                torch.zeros(2, device=env.device),
                atol=1e-5,
                rtol=0.0,
            )
            assert ((height_command >= 0.30) & (height_command <= 0.80)).all()
            torch.testing.assert_close(obs["command"][:, -1:], height_command)
            pitch_term = env.command_manager.get_term("torso_pitch")
            pitch_command = pitch_term.command
            assert ((pitch_command >= 0.0) & (pitch_command <= math.pi / 6)).all()
            torch.testing.assert_close(obs["command"][:, -2:-1], pitch_command)
            torch.testing.assert_close(obs["command"][:, :-2], env.command_manager.get_command("base_velocity"))
            assert obs["policy"].shape == (2, 790)
            assert obs["critic"].shape == (2, 800)

            # The same mirrors must handle full joint observations and only 15 actions.
            action = torch.randn_like(action)
            augmented_obs, augmented_action = compute_mirrored_states(env, TensorDict(obs, [2]), action)
            torch.testing.assert_close(augmented_obs["command"][2:, -2:-1], pitch_command)
            torch.testing.assert_close(augmented_obs["command"][2:, -1:], height_command)
            twice_obs, twice_action = compute_mirrored_states(env, augmented_obs[2:], augmented_action[2:])
            torch.testing.assert_close(twice_action[2:], action)
            for name in obs:
                torch.testing.assert_close(twice_obs[name][2:], obs[name])

            # Large changes confined to randomized arms must not incur policy joint penalties.
            terms = [
                env.reward_manager.get_term_cfg(name)
                for name in ("energy", "dof_vel_l2", "dof_acc_l2", "dof_pos_limits", "joint_deviation")
            ]
            before = [term.func(env, **term.params).clone() for term in terms]
            buffers = [
                robot.data.joint_pos.torch,
                robot.data.joint_vel.torch,
                robot.data.joint_acc.torch,
                robot.data.applied_torque.torch,
            ]
            saved = [value.clone() for value in buffers]
            try:
                for value in buffers:
                    value[:, arm_ids] = 100.0
                for term, expected in zip(terms, before, strict=True):
                    torch.testing.assert_close(term.func(env, **term.params), expected)
                # Positive control: selected leg motion must still incur regularization.
                buffers[1][:, lower_ids] = 100.0
                velocity_term = env.reward_manager.get_term_cfg("dof_vel_l2")
                assert (velocity_term.func(env, **velocity_term.params) > before[1]).all()
            finally:
                for value, original in zip(buffers, saved, strict=True):
                    value.copy_(original)

            # The command changes the reward target, and the curriculum tightens its tolerance.
            saved_height_command = height_command.clone()
            saved_step = env.common_step_counter
            empty_ids = torch.empty(0, dtype=torch.long, device=env.device)
            try:
                for step, expected_std in ((0, 0.20), (7500 * 24, 0.125), (15000 * 24, 0.05)):
                    env.common_step_counter = step
                    env.curriculum_manager.compute(empty_ids)
                    term = env.reward_manager.get_term_cfg("base_height")
                    assert term.params["std"] == pytest.approx(expected_std)
                    measured_height = (
                        robot.data.root_pos_w.torch[:, 2] - env.command_manager.get_term("base_height").terrain_height
                    )
                    height_command[:, 0] = measured_height
                    torch.testing.assert_close(term.func(env, **term.params), torch.ones(2, device=env.device))
                    height_command[:, 0] += expected_std
                    torch.testing.assert_close(
                        term.func(env, **term.params), torch.full((2,), 1 / torch.e, device=env.device)
                    )
            finally:
                height_command.copy_(saved_height_command)
                env.common_step_counter = saved_step
                env.curriculum_manager.compute(empty_ids)

            # Torso tracking must be independent of root pose and world yaw, and reward
            # forward lean rather than fighting it with the old upright-torso term.
            orientation_term = env.reward_manager.get_term_cfg("trunk_orientation")
            orientation_buffer = robot.data.body_quat_w.torch
            saved_orientation = orientation_buffer.clone()
            saved_pitch = pitch_command.clone()
            try:
                pitch_command[:, 0] = math.pi / 6
                zero = torch.zeros(2, device=env.device)
                yaw = torch.tensor([0.0, 1.7], device=env.device)
                orientation_buffer[:, pitch_term.body_idx] = quat_from_euler_xyz(zero, pitch_command[:, 0], yaw)
                torch.testing.assert_close(
                    orientation_term.func(env, **orientation_term.params), torch.ones(2, device=env.device)
                )
                # One tolerance of roll error or pitch error should yield exp(-1).
                std = orientation_term.params["std"]
                orientation_buffer[:, pitch_term.body_idx] = quat_from_euler_xyz(
                    torch.tensor([std, 0.0], device=env.device),
                    pitch_command[:, 0] - torch.tensor([0.0, std], device=env.device),
                    yaw,
                )
                torch.testing.assert_close(
                    orientation_term.func(env, **orientation_term.params),
                    torch.full((2,), 1 / torch.e, device=env.device),
                )
            finally:
                orientation_buffer.copy_(saved_orientation)
                pitch_command.copy_(saved_pitch)

            untouched_pitch = pitch_command[1].clone()
            untouched_height = height_command[1].clone()
            untouched = arms.processed_actions[1].clone()
            env._reset_idx(torch.tensor([0], device=env.device))
            torch.testing.assert_close(pitch_command[1], untouched_pitch)
            assert 0.0 <= pitch_command[0, 0] <= math.pi / 6
            torch.testing.assert_close(height_command[1], untouched_height)
            assert 0.30 <= height_command[0, 0] <= 0.80
            torch.testing.assert_close(arms.processed_actions[1], untouched)
            torch.testing.assert_close(arms.processed_actions[0], robot.data.default_joint_pos.torch[0, arm_ids])
            _, reward, _, _, _ = env.step(torch.zeros_like(action))
            assert torch.isfinite(reward).all()
        finally:
            env.close()
