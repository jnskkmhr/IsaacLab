# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Check G1 soft-contact task registration and bounded GPU rollouts.

Run with: ./isaaclab.sh -p -m pytest source/isaaclab_tasks/test/test_g1_29dof_soft.py -v -s
"""

import sys

# Prefer the installed Pinocchio libraries over the simulator's bundled libraries.
if sys.platform != "win32":
    import pinocchio  # noqa: F401

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import gymnasium as gym
import pytest
import torch

import omni.usd

from isaaclab.sim import SimulationContext

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg

TASKS = sorted(task for task in gym.registry if task.startswith("Isaac-Velocity-Flat-G1-29dof-Soft-"))


def test_registration():
    """All migrated vanilla, teacher, and student variants must remain discoverable."""
    assert len(TASKS) == 8


@pytest.mark.parametrize("task", TASKS)
def test_configuration(task):
    """Resolve every registered environment and agent entry point."""
    cfg = parse_env_cfg(task, device="cuda:0", num_envs=4)
    cfg.validate()
    for key in gym.spec(task).kwargs:
        if key != "env_cfg_entry_point":
            load_cfg_from_registry(task, key)


@pytest.mark.parametrize("task", TASKS)
def test_rollout(task):
    """Stepping and resetting must keep observations, rewards, and foot forces finite."""
    cfg = parse_env_cfg(task, device="cuda:0", num_envs=4)
    omni.usd.get_context().new_stage()
    env = None
    try:
        env = gym.make(task, cfg=cfg)
        env.unwrapped.sim._app_control_on_stop_handle = None
        env.reset()
        callback = env.unwrapped.action_manager.get_term("physics_callback")
        assert callback.backend == "3D-warp"
        assert env.action_space.shape == (4, 29)
        max_force = 0.0
        with torch.inference_mode():
            for step in range(100):
                actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
                obs, reward, terminated, truncated, _ = env.step(actions)
                for name, value in obs.items():
                    assert torch.isfinite(value).all(), f"{task}: {name} at step {step}"
                assert torch.isfinite(reward).all(), f"{task}: reward at step {step}"
                assert torch.isfinite(callback.contact_wrench).all(), f"{task}: foot wrench at step {step}"
                assert torch.isfinite(env.unwrapped.scene["robot"].data.root_state_w).all()
                max_force = max(max_force, callback.contact_wrench[..., :3].norm(dim=-1).max().item())
                if step == 49:
                    env.reset()
        assert max_force > 0.0, "The soft-contact model never exerted force on the robot's feet."
    finally:
        if env is not None:
            env.close()
        elif SimulationContext.instance() is not None:
            sim = SimulationContext.instance()
            sim.clear_all_callbacks()
            sim.clear_instance()
