# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Analytic and lifecycle checks for gravity observations."""

import math
from types import SimpleNamespace

import pytest
import torch
import warp as wp

import isaaclab.envs.mdp as mdp
from isaaclab.envs.mdp.observations import body_projected_gravity_b
from isaaclab.managers import (
    CurriculumTermCfg,
    ObservationGroupCfg,
    ObservationManager,
    ObservationTermCfg,
    SceneEntityCfg,
)
from isaaclab.test.utils import test_devices
from isaaclab.utils.warp import ProxyArray


@pytest.mark.unit
@pytest.mark.parametrize("device", test_devices())
def test_body_projected_gravity_b_stacks_every_selected_body(device):
    """Each selected body receives its own environment's gravity, including integer selections."""
    num_envs = 2
    angles = (0.0, 0.5 * math.pi, math.pi)
    body_quat = torch.tensor([[math.sin(0.5 * a), 0.0, 0.0, math.cos(0.5 * a)] for a in angles], device=device).repeat(
        num_envs, 1, 1
    )
    asset = SimpleNamespace(
        data=SimpleNamespace(
            body_quat_w=ProxyArray(wp.from_torch(body_quat, dtype=wp.quat)),
            GRAVITY_VEC_W=ProxyArray(wp.array([[0.0, 0.0, -9.81], [0.0, 9.81, 0.0]], dtype=wp.vec3, device=device)),
        )
    )
    env = SimpleNamespace(scene={"robot": asset}, num_envs=num_envs)
    # R_x(a)^T applied to -Z in the first environment and +Y in the second.
    expected_z = [[0.0, -math.sin(a), -math.cos(a)] for a in angles]
    expected_y = [[0.0, math.cos(a), -math.sin(a)] for a in angles]
    expected = torch.tensor([expected_z, expected_y], device=device).reshape(num_envs, -1)
    torch.testing.assert_close(body_projected_gravity_b(env, SceneEntityCfg("robot")), expected)
    for body_ids in ([1], 1):
        asset_cfg = SceneEntityCfg("robot", body_ids=body_ids)
        torch.testing.assert_close(body_projected_gravity_b(env, asset_cfg), expected[:, 3:6])


@pytest.mark.unit
@pytest.mark.parametrize("axis, expected", [("roll", [0.0, -1.0, 0.0]), ("pitch", [1.0, 0.0, 0.0])])
def test_bias_projected_gravity_rotation_and_noise(axis, expected):
    """A positive quarter-turn has the documented sign; angular noise preserves unit gravity."""
    gravity = torch.tensor([[0.0, 0.0, -1.0]]).repeat(128, 1)
    env = SimpleNamespace(
        num_envs=128,
        device="cpu",
        scene={"robot": SimpleNamespace(data=SimpleNamespace(projected_gravity_b=ProxyArray(wp.from_torch(gravity))))},
    )
    cfg = ObservationTermCfg(func=mdp.bias_projected_gravity, params={f"{axis}_bias_range": (math.pi / 2, math.pi / 2)})
    term = mdp.bias_projected_gravity(cfg, env)
    actual = term(env, **cfg.params)
    torch.testing.assert_close(actual, torch.tensor([expected]).expand_as(actual), atol=1e-6, rtol=0.0)
    noisy = term(env, roll_noise_std=0.1, pitch_noise_std=0.1)
    torch.testing.assert_close(torch.linalg.vector_norm(noisy, dim=-1), torch.ones(128))
    assert not torch.allclose(noisy, actual)
    torch.testing.assert_close(gravity, torch.tensor([[0.0, 0.0, -1.0]]).repeat(128, 1))


@pytest.mark.unit
def test_bias_curriculum_updates_only_reset_episodes():
    """The live manager config reaches reset sampling, without altering ongoing episodes."""
    gravity = torch.tensor([[0.0, 0.0, -1.0]]).repeat(32, 1)
    env = SimpleNamespace(
        num_envs=32,
        device="cpu",
        common_step_counter=0,
        sim=SimpleNamespace(is_playing=lambda: True),
        scene={"robot": SimpleNamespace(data=SimpleNamespace(projected_gravity_b=ProxyArray(wp.from_torch(gravity))))},
    )
    group = ObservationGroupCfg()
    group.projected_gravity = ObservationTermCfg(func=mdp.bias_projected_gravity)
    env.observation_manager = ObservationManager(SimpleNamespace(policy=group), env)
    env.observation_manager.reset()
    clean = env.observation_manager.compute()["policy"]
    torch.testing.assert_close(clean, gravity)
    curriculum_cfg = CurriculumTermCfg(
        func=mdp.modify_term_cfg,
        params={
            "address": "observations.policy.projected_gravity.params.pitch_bias_range",
            "modify_fn": mdp.linear_ramp,
            "modify_params": {
                "initial_value": (0.0, 0.0),
                "final_value": (-0.2, 0.2),
                "start_step": 10,
                "end_step": 30,
            },
        },
    )
    curriculum = mdp.modify_term_cfg(curriculum_cfg, env)
    for step, limit in [(0, 0.0), (20, 0.1), (40, 0.2)]:
        env.common_step_counter = step
        curriculum(env, [0], **curriculum_cfg.params)
        assert env.observation_manager.cfg.policy.projected_gravity.params["pitch_bias_range"] == (-limit, limit)
    # A curriculum update alone must not change existing episode bias.
    torch.testing.assert_close(env.observation_manager.compute()["policy"], clean)
    selected = torch.arange(0, 32, 2)
    env.observation_manager.reset(selected)
    changed = env.observation_manager.compute()["policy"]
    torch.testing.assert_close(changed[1::2], clean[1::2])
    assert torch.count_nonzero(changed[selected, 0]) == len(selected)
    assert changed[selected, 0].abs().max() <= math.sin(0.2) + 1e-6
    assert changed[selected, 0].std() > 0.01
    torch.testing.assert_close(env.observation_manager.compute()["policy"], changed)
    # Existing sampled bias survives shrinking the range until its own reset.
    env.common_step_counter = 0
    curriculum(env, [0], **curriculum_cfg.params)
    env.observation_manager.reset([0])
    after = env.observation_manager.compute()["policy"]
    torch.testing.assert_close(after[0], clean[0])
    torch.testing.assert_close(after[2:], changed[2:])
    env.observation_manager.reset()
    torch.testing.assert_close(env.observation_manager.compute()["policy"], clean)


@pytest.mark.unit
def test_linear_ramp_step_change():
    """A zero-duration schedule changes exactly at the boundary; reversed bounds fail."""
    env = SimpleNamespace(common_step_counter=9)
    assert mdp.linear_ramp(env, [], 0.0, 0.0, 2.0, 10, 10) == 0.0
    env.common_step_counter = 10
    assert mdp.linear_ramp(env, [], 0.0, 0.0, 2.0, 10, 10) == 2.0
    with pytest.raises(ValueError, match="end_step"):
        mdp.linear_ramp(env, [], 0.0, 0.0, 2.0, 10, 9)
