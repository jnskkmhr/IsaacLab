# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Joint selection and episode history for G1 action penalties."""

from types import SimpleNamespace

import torch

from isaaclab.managers import RewardTermCfg, SceneEntityCfg

from isaaclab_tasks.contrib.velocity.config.g1_29dof_rigid.mdp.rewards import action_rate_l2, action_rate_rate_l2


def test_action_penalties_select_joints_and_reset_episode_history():
    """Unselected joints do not contribute, and resetting one environment preserves the other's history."""
    actions = SimpleNamespace(action=torch.zeros(2, 4), prev_action=torch.zeros(2, 4))
    env = SimpleNamespace(action_manager=actions, num_envs=2, device="cpu")
    asset_cfg = SceneEntityCfg("robot", joint_ids=[1, 3])
    term = action_rate_rate_l2(RewardTermCfg(func=action_rate_rate_l2, weight=-1.0), env)

    actions.action[:] = torch.tensor([[100.0, 1.0, -100.0, 2.0], [200.0, 2.0, -200.0, 3.0]])
    torch.testing.assert_close(action_rate_l2(env, asset_cfg), torch.tensor([5.0, 13.0]))
    torch.testing.assert_close(term(env, asset_cfg), torch.tensor([5.0, 13.0]))

    # Constant action increments have zero second difference, even as other joints jump.
    actions.prev_action.copy_(actions.action)
    actions.action[:] = torch.tensor([[-500.0, 2.0, 500.0, 4.0], [-500.0, 4.0, 500.0, 6.0]])
    torch.testing.assert_close(term(env, asset_cfg), torch.zeros(2))

    term.reset(torch.tensor([0]))
    actions.prev_action.copy_(actions.action)
    actions.action[:] = torch.tensor([[900.0, 3.0, -900.0, 6.0], [900.0, 7.0, -900.0, 7.0]])
    torch.testing.assert_close(term(env, asset_cfg), torch.tensor([5.0, 5.0]))

    term.reset()
    actions.action.zero_()
    actions.prev_action.zero_()
    torch.testing.assert_close(term(env, asset_cfg), torch.zeros(2))

    # Playback may step under inference_mode and reset outside it. Keep the history
    # buffer writable across both contexts rather than replacing it with an inference tensor.
    with torch.inference_mode():
        actions.action.fill_(1.0)
        term(env, asset_cfg)
    term.reset()
    actions.action.zero_()
    torch.testing.assert_close(term(env, asset_cfg), torch.zeros(2))
