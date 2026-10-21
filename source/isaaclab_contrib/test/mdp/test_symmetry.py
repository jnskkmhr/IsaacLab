# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Behavior of shared manager-term symmetry through its public API."""

from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from isaaclab_contrib.mdp import (
    MirrorJointPositionActionCfg,
    MirrorObservationTermCfg,
    compute_mirrored_states,
    mirror_joints,
    mirror_vec3,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("concatenate", [True, False])
def test_augmentation_preserves_originals_and_reflects_history_and_actions(concatenate):
    """New shared configs drive both manager layouts without changing the original samples."""
    observation = MirrorObservationTermCfg(func=lambda env: None, mirror=mirror_vec3, history_length=2)
    action = MirrorJointPositionActionCfg(
        asset_name="robot",
        joint_names=["left", "right"],
        mirror=mirror_joints,
        mirror_params={"permutation": [1, 0], "signs": [-1, -1]},
    )
    env = SimpleNamespace(
        observation_manager=SimpleNamespace(
            cfg={"policy": SimpleNamespace(velocity=observation, concatenate_dim=-1)},
            active_terms={"policy": ["velocity"]},
            group_obs_term_dim={"policy": [(6,)]},
            group_obs_concatenate={"policy": concatenate},
        ),
        action_manager=SimpleNamespace(
            active_terms=["joint_pos"],
            action_term_dim=[2],
            get_term=lambda name: SimpleNamespace(cfg=action),
        ),
    )
    env.unwrapped = env
    values = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
    observations = TensorDict({"policy": values if concatenate else {"velocity": values}}, batch_size=[1])
    actions = torch.tensor([[2.0, -3.0]])

    mirrored_obs, mirrored_actions = compute_mirrored_states(env, observations, actions)

    key = "policy" if concatenate else ("policy", "velocity")
    torch.testing.assert_close(
        mirrored_obs[key], torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [1.0, -2.0, 3.0, 4.0, -5.0, 6.0]])
    )
    torch.testing.assert_close(mirrored_actions, torch.tensor([[2.0, -3.0], [3.0, -2.0]]))
    torch.testing.assert_close(observations[key], values)
    torch.testing.assert_close(actions, torch.tensor([[2.0, -3.0]]))
