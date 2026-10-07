# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""G1 terrain resets honor current ranges and preserve unselected robots."""

import gymnasium as gym
import pytest
import torch

from isaaclab.app import launch_simulation
from isaaclab.test.utils import DeviceScope, test_devices

from isaaclab_tasks.contrib.g1_29dof_mpm_moving_patch.mpm_env_cfg import G1MovingPatchEnvCfg_PLAY
from isaaclab_tasks.utils.hydra import resolve_presets


@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_g1_terrain_reset_uses_current_ranges(device):
    """Full and partial resets sample current bounds before a finite simulation step."""
    cfg = resolve_presets(G1MovingPatchEnvCfg_PLAY(), selected={"mjwarp_mpm_proxy"})
    cfg.scene.num_envs = 2
    cfg.sim.device = device
    cfg.sim.visualizer_cfgs = []
    cfg.sim.default_visualizer_cfg = None
    cfg.sim.physics.use_cuda_graph = False
    cfg.video_recorders = []
    cfg.events.reset_base.params["pose_range"] = {"x": (0.1, 0.1)}
    cfg.events.reset_base.params["velocity_range"] = {}

    with launch_simulation(cfg.sim, {"headless": True, "device": device}):
        env = gym.make("IsaacContrib-Velocity-Sand-G1-29dof-MPM-MovingPatch-Play", cfg=cfg)
        try:
            env.reset()
            raw = env.unwrapped
            robot = raw.scene["robot"]
            expected = robot.data.default_root_pose.torch[:, :3].clone() + raw.scene.env_origins
            expected[:, 0] += 0.1
            torch.testing.assert_close(robot.data.root_pos_w.torch, expected)

            term = raw.event_manager.get_term_cfg("reset_base")
            term.params["pose_range"] = {"x": (0.3, 0.3)}
            term.params["velocity_range"] = {"x": (0.2, 0.2)}
            before = robot.data.root_state_w.torch.clone()
            raw.reset(env_ids=torch.tensor([1], device=device))
            torch.testing.assert_close(robot.data.root_state_w.torch[0], before[0])
            expected[1, 0] += 0.2
            torch.testing.assert_close(robot.data.root_pos_w.torch, expected)
            torch.testing.assert_close(robot.data.root_lin_vel_w.torch[1], torch.tensor([0.2, 0.0, 0.0], device=device))

            before = robot.data.root_state_w.torch.clone()
            raw.reset(env_ids=slice(0, 1))
            torch.testing.assert_close(robot.data.root_state_w.torch[1], before[1])
            expected[0, 0] += 0.2
            torch.testing.assert_close(robot.data.root_pos_w.torch, expected)
            actions = torch.zeros((raw.num_envs, raw.action_manager.total_action_dim), device=device)
            for _ in range(3):
                observations, rewards, _, _, _ = env.step(actions)
                assert torch.isfinite(rewards).all()
                assert all(torch.isfinite(value).all() for value in observations.values())
                assert torch.isfinite(robot.data.root_state_w.torch).all()
        finally:
            env.close()
