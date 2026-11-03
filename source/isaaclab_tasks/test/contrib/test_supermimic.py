# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Physical reward parity for the differentiable rigid and coupled velocity tasks."""

import gymnasium as gym
import pytest
import torch

from isaaclab.app import launch_simulation

from isaaclab_tasks.core.velocity.supermimic import VelocityTask
from isaaclab_tasks.utils.hydra import resolve_task_config


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires Newton on CUDA")
@pytest.mark.parametrize("terrain", ["Rigid", "MixedTerrain"])
def test_differentiable_rewards_match_physical_terms(terrain, record_property):
    """Every exact term must agree; separately report the PR's PD/qacc approximations."""
    task_name = f"IsaacContrib-Velocity-G1-29dof-SuperMimic-{terrain}"
    cfg, _ = resolve_task_config(task_name, None, overrides=[])
    cfg.scene.num_envs = 4
    cfg.seed = 42
    cfg.video_recorders = []
    approximations = {"dof_torques_l2": 0.0, "dof_acc_l2": 0.0}
    with launch_simulation(cfg, {"visualizer": ["none"]}):
        env = gym.make(task_name, cfg=cfg)
        try:
            obs, _ = env.reset()
            task = VelocityTask(env, env.unwrapped.device)
            phase = torch.linspace(0, 2 * torch.pi, 4 * task.num_actions, device=task.device).reshape(4, -1)
            for step in range(24):
                state = task.representation.unflatten(obs["state"].clone())
                goal = obs["goal"].clone()
                history = torch.stack(
                    (env.unwrapped.action_manager.prev_action, env.unwrapped.action_manager.action), -2
                )
                actions = 0.1 * torch.sin(phase + step * 0.3)
                obs, _, terminated, _, extras = env.step(actions)
                final = extras["final_obs"]
                successor = task.representation.unflatten(final["state"])
                predicted = task.reward_terms(state, actions, successor, goal, history, final["labels"])
                manager = env.unwrapped.reward_manager
                for i, name in enumerate(manager.active_terms):
                    reference = manager._step_reward[:, i] * task.dt
                    if name in approximations:
                        error = (predicted[name] - reference).abs().max().item()
                        assert torch.isfinite(predicted[name]).all()
                        approximations[name] = max(approximations[name], error)
                    else:
                        torch.testing.assert_close(predicted[name], reference, atol=1e-6, rtol=1e-5, msg=name)
                torch.testing.assert_close(task.termination_prob(successor, goal, final["labels"]), terminated.float())
            for name, error in approximations.items():
                record_property(f"{name}_max_absolute_reward_error", error)
            action = actions.detach().clone().requires_grad_()
            task.reward(state, action, successor, goal, history, final["labels"]).sum().backward()
            assert torch.isfinite(action.grad).all() and action.grad.abs().sum() > 0
        finally:
            env.close()


def test_no_jump_penalizes_flight_and_preserves_contact_gradients():
    """Flight is penalized on both contact sources; imagined ascent encourages contact."""
    from types import SimpleNamespace

    from isaaclab.managers import SceneEntityCfg

    from isaaclab_tasks.contrib.g1_29dof_supermimic.env_cfg import G1SuperMimicEnvCfg
    from isaaclab_tasks.contrib.g1_29dof_supermimic.mdp.rewards import no_jump
    from isaaclab_tasks.core.velocity.supermimic import VelocityStateRepresentation

    # Flight, left stance, right stance, double stance, then a soft imagined event.
    contacts = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.2, 0.4]])
    clocks = torch.zeros(4, 3)
    clocks[:, [2, 0]] = contacts[:4] * 0.02
    clocks[:, 1] = 0.02  # Contact on an unrelated body must not mask airborne feet.
    sensor = SimpleNamespace(data=SimpleNamespace(current_contact_time=SimpleNamespace(torch=clocks)))
    env = SimpleNamespace(
        scene=SimpleNamespace(sensors={"feet": sensor}),
        foot_contact_time=clocks[:, [2, 0]],
    )
    expected = torch.tensor([1.0, 0.0, 0.0, 0.0])
    torch.testing.assert_close(no_jump(env, SceneEntityCfg("feet", body_ids=[2, 0])), expected)
    torch.testing.assert_close(no_jump(env), expected)

    task = object.__new__(VelocityTask)
    task.num_feet = task.num_bodies = 2
    task.dt = 0.02
    task.gravity = torch.tensor([0.0, 0.0, -1.0])
    task.rewards = {"no_jump": G1SuperMimicEnvCfg().rewards.no_jump}
    representation = VelocityStateRepresentation(2, 2)
    flat = torch.zeros(5, sum(representation.widths))
    flat[:, 9] = 1.0  # Identity XYZW root quaternion.
    state = representation.unflatten(flat)
    labels = torch.cat((contacts, torch.zeros(5, 4)), -1).requires_grad_()
    reward = task.reward_terms(state, torch.zeros(5, 2), state, torch.zeros(5, 3), torch.zeros(5, 2, 2), labels)[
        "no_jump"
    ]
    torch.testing.assert_close(reward, torch.tensor([-0.04, 0.0, 0.0, 0.0, -0.0192]))
    reward[-1].backward()
    assert torch.isfinite(labels.grad).all()
    assert (labels.grad[-1, :2] > 0).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires Newton on CUDA")
def test_agile_supermimic_matches_native_rewards_and_observations():
    """Prevent silently changing Agile terms, observation scales or reset memory in imagination."""
    from isaaclab_tasks.contrib.velocity.config.g1_29dof_rigid_agile.mdp.supermimic import AgileTask

    task_name = "IsaacContrib-Velocity-Rough-G1-29dof-Rigid-Agile-SuperMimic"
    cfg, _ = resolve_task_config(task_name, None, overrides=[])
    cfg.scene.num_envs = 4
    cfg.scene.terrain.terrain_generator.num_rows = 2
    cfg.scene.terrain.terrain_generator.num_cols = 2
    cfg.scene.terrain.max_init_terrain_level = 1
    cfg.observations.policy.enable_corruption = False
    cfg.video_recorders = []
    with launch_simulation(cfg, {"visualizer": ["none"]}):
        env = gym.make(task_name, cfg=cfg)
        try:
            obs, _ = env.reset()
            task = AgileTask(env, env.unwrapped.device)
            for step in range(32):
                if step == 16:
                    obs, _ = env.reset()
                state = task.representation.unflatten(obs["state"].clone())
                goal = obs["goal"].clone()
                previous = torch.stack((obs["prev_prev_action"], obs["prev_action"]), -2).clone()
                policy_parts = obs["policy"].split(tuple(5 * n for n in task.frame_widths), -1)
                history = torch.cat(tuple(p.reshape(4, 5, -1) for p in policy_parts), -1)
                torch.testing.assert_close(task.actor_obs(state, goal, obs["context"], previous), history[:, -1])
                torch.testing.assert_close(
                    task.critic_obs_with_history(state, goal, obs["context"], previous, history),
                    torch.cat((obs["policy"], obs["critic"], obs["height_scan"]), -1),
                )
                actions = torch.randn(4, 15, device=task.device) * 0.2
                obs, _, terminated, _, extras = env.step(actions)
                final = extras["final_obs"]
                successor = task.representation.unflatten(final["state"])
                predicted = task.reward_terms(state, actions, successor, goal, previous, final["labels"])
                manager = env.unwrapped.reward_manager
                for i, name in enumerate(manager.active_terms):
                    torch.testing.assert_close(
                        predicted[name], manager._step_reward[:, i] * task.dt, atol=1e-5, rtol=1e-4, msg=name
                    )
                torch.testing.assert_close(task.termination_prob(successor, goal, final["labels"]), terminated.float())
            increment = task.representation.increments(state, successor).detach().requires_grad_()
            action = actions.detach().requires_grad_()
            imagined = task.post_transition(
                state, task.representation.integrate(state, increment), action, previous, final["labels"]
            )
            task.reward(state, action, imagined, goal, previous, final["labels"]).sum().backward()
            assert torch.isfinite(action.grad).all() and action.grad.abs().sum() > 0
            assert torch.isfinite(increment.grad).all() and increment.grad.abs().sum() > 0
        finally:
            env.close()
