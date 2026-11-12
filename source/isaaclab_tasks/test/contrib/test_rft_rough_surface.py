# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Keep RFT penetration and height rewards referenced to the material surface."""

import gymnasium as gym
import pytest
import torch
import warp as wp

from isaaclab.app import launch_simulation
from isaaclab.terrains import MeshPlaneTerrainCfg

from isaaclab_tasks.utils.hydra import resolve_task_config


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires Newton on CUDA")
def test_rft_queries_surface_above_buried_support():
    """A raised surface stays at 0.30 m while its rigid support is lowered to 0.18 m."""
    name = "IsaacContrib-Velocity-Rough-G1-29dof-SoftContact-Newton"
    cfg, _ = resolve_task_config(name, None, overrides=[])
    cfg.scene.num_envs = 2
    cfg.scene.terrain.terrain_generator = cfg.scene.terrain.terrain_generator.replace(
        num_rows=2,
        num_cols=2,
        sub_terrains={"flat": MeshPlaneTerrainCfg(proportion=1.0, ground_height_range=(0.3, 0.3))},
    )
    cfg.video_recorders = []
    with launch_simulation(cfg, {"visualizer": ["none"]}):
        env = gym.make(name, cfg=cfg).unwrapped
        try:
            terrain = env.scene.terrain
            terrain.terrain_levels[:] = 1
            terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
            env.reset()
            robot = env.scene["robot"]
            torch.testing.assert_close(robot.data.root_pos_w.torch[:, 2], torch.full((2,), 1.06, device=env.device))
            assert terrain.support_depths[1, 1] == pytest.approx(0.12)
            foot_ids, _ = robot.find_bodies(["left_ankle_roll_link", "right_ankle_roll_link"], preserve_order=True)
            for _ in range(4):
                obs, reward, *_ = env.step(torch.zeros((2, 29), device=env.device))
                assert torch.isfinite(reward).all()
                assert all(torch.isfinite(value).all() for value in obs.values())
                solver = env.action_manager.get_term("physics_callback").contact_solver
                heights = wp.to_torch(solver.terrain_height)
                torch.testing.assert_close(heights, torch.full_like(heights, 0.3), atol=1e-5, rtol=0)
                for foot_id, scanner in zip(foot_ids, ("left_foot_scanner", "right_foot_scanner")):
                    hits = env.scene[scanner].data.ray_hits_w.torch[:, 0]
                    torch.testing.assert_close(
                        hits[:, :2], robot.data.body_pos_w.torch[:, foot_id, :2], atol=1e-5, rtol=0
                    )
                    torch.testing.assert_close(hits[:, 2], torch.full_like(hits[:, 2], 0.3), atol=1e-5, rtol=0)
        finally:
            env.close()
