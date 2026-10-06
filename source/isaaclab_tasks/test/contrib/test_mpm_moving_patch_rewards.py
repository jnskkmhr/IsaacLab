# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Terrain-relative foot pitch at first contact in the moving-patch task."""

import math
from types import SimpleNamespace

import torch

from isaaclab.utils.math import quat_from_euler_xyz

from isaaclab_tasks.contrib.g1_29dof_mpm_moving_patch.env_cfg.reward_cfg import G1RewardsCfg


def test_touchdown_pitch_follows_scanner_slope_and_contact_order():
    """Fit current scanner hits in world coordinates and penalize only new contacts."""
    # Flat, toe-down, heel-down, swing, uphill, cross-slope, downhill, diagonal,
    # all missed, too few hits, and collinear hits.
    slopes = torch.tensor([[0.0, 0.0]] * 4 + [[0.25, 0.0], [0.25, -0.3], [0.25, 0.0], [0.25, 0.1]] + [[0.0, 0.0]] * 3)
    yaw = torch.tensor([0.0] * 5 + [math.pi / 2, math.pi, math.pi / 4] + [0.0] * 3)
    elevation = torch.tensor(
        [
            0.0,
            -0.35,
            0.35,
            -0.35,
            math.atan(0.25),
            -math.atan(0.3),
            -math.atan(0.25),
            math.atan(0.35 / math.sqrt(2)),
            -0.35,
            -0.35,
            -0.35,
        ]
    )
    num_envs = len(slopes)
    x, y = torch.meshgrid(torch.tensor([-0.1, 0.0, 0.1]), torch.tensor([-0.1, 0.0, 0.1]), indexing="ij")
    # A yawed scanner and shuffled rays must still yield world-frame slopes.
    xy = torch.stack((x.flatten(), y.flatten()), dim=-1)
    rotation = torch.tensor([[math.cos(0.4), -math.sin(0.4)], [math.sin(0.4), math.cos(0.4)]])
    xy = (xy @ rotation.T)[torch.tensor([4, 2, 8, 0, 1, 3, 5, 7, 6])]
    hits = torch.zeros(num_envs, 9, 3)
    hits[..., :2] = xy
    hits[..., 2] = (xy.unsqueeze(0) * slopes.unsqueeze(1)).sum(dim=-1)
    hits += torch.tensor([10.0, -5.0, 3.0])
    hits[4, 0] = float("nan")  # Partial misses still leave a usable plane.
    hits[5, 8] = float("inf")
    hits[8] = float("inf")
    hits[9, 2:] = float("nan")
    hits[10, :, 1] = 0.0  # All remaining points lie on a line.
    pitch = torch.stack([-elevation, torch.full_like(elevation, 0.8)], dim=-1)
    quaternions = quat_from_euler_xyz(torch.zeros_like(pitch), pitch, yaw[:, None].expand_as(pitch))
    robot = SimpleNamespace(
        num_bodies=2,
        find_bodies=lambda *args, **kwargs: ([0, 1], ["left_ankle_roll_link", "right_ankle_roll_link"]),
        data=SimpleNamespace(body_quat_w=SimpleNamespace(torch=quaternions)),
    )
    contact = torch.tensor([[1.0, 0.0]] * num_envs)
    contact[3, 0] = 0.0
    env = SimpleNamespace(
        num_envs=num_envs,
        device="cpu",
        scene={
            "robot": robot,
            "height_scanner": SimpleNamespace(data=SimpleNamespace(ray_hits_w=SimpleNamespace(torch=hits))),
        },
        cfg=SimpleNamespace(foot_body_expr=".*_ankle_roll_link"),
        foot_first_contact=contact,
    )
    cfg = G1RewardsCfg().foot_touch_down_angle_penalty
    cfg.params["asset_cfg"].body_ids = [1, 0]  # Reward selection can differ from contact storage order.
    term = cfg.func(cfg, env)
    expected = torch.zeros(num_envs)
    expected[1:3] = (0.35 - cfg.params["angle_tolerance"]) ** 2
    torch.testing.assert_close(term(env, **cfg.params), expected, atol=1.0e-6, rtol=1.0e-5)
    # Use the scanner's latest values, not a cached slope from the first call.
    hits[0, :, 2] = 0.3 * hits[0, :, 0]
    expected[0] = (math.atan(0.3) - cfg.params["angle_tolerance"]) ** 2
    torch.testing.assert_close(term(env, **cfg.params), expected, atol=1.0e-6, rtol=1.0e-5)
    contact[:] = 0.0
    quaternions[0, 0] = float("nan")
    contact[0, 0] = 1.0
    torch.testing.assert_close(term(env, **cfg.params), torch.zeros(num_envs))
