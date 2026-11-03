# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Joint-range target mapping through the action term's public processing API."""

from types import SimpleNamespace

import pytest
import torch

from isaaclab.utils.string import resolve_matching_names

from isaaclab_contrib.mdp import JointPositionRangeAction, MirrorJointPositionRangeActionCfg, mirror_identity

pytestmark = pytest.mark.unit


class JointIds:
    def __init__(self, ids):
        self.torch = torch.tensor(ids)

    def __len__(self):
        return len(self.torch)


class ArticulationFixture:
    """Supply articulation properties and capture position targets without simulating physics."""

    num_joints = 3

    def __init__(self):
        self.data = SimpleNamespace(
            default_joint_pos=SimpleNamespace(torch=torch.tensor([[0.0, 0.0, -1.0]]).repeat(3, 1)),
            joint_pos_limits=SimpleNamespace(
                torch=torch.tensor([[[-1.0, 2.0], [-float("inf"), float("inf")], [-4.0, 1.0]]]).repeat(3, 1, 1)
            ),
        )

    def find_joints(self, names, preserve_order, as_proxy):
        ids, names = resolve_matching_names(names, ["j0", "j1", "j2"], preserve_order=preserve_order)
        return JointIds(ids), names

    def set_joint_position_target_index(self, target, joint_ids):
        self.targets = target.clone()
        self.target_ids = joint_ids.clone()


def test_joint_range_targets_keep_default_pose_and_reach_farther_limits():
    """Reordered selected joints use symmetric physical ranges, including overshoot of nearer limits."""
    asset = ArticulationFixture()
    env = SimpleNamespace(scene={"robot": asset}, num_envs=3, device="cpu")
    cfg = MirrorJointPositionRangeActionCfg(
        asset_name="robot", joint_names=["j2", "j0"], preserve_order=True, mirror=mirror_identity
    )
    action = JointPositionRangeAction(cfg, env)
    raw = torch.tensor([[0.0, 0.0], [1.0, -1.0], [-1.0, 1.0]])
    action.process_actions(raw)
    action.apply_actions()
    torch.testing.assert_close(asset.targets, torch.tensor([[-1.0, 0.0], [2.0, -2.0], [-4.0, 2.0]]))
    torch.testing.assert_close(asset.target_ids, torch.tensor([2, 0]))
    torch.testing.assert_close(action.raw_actions, raw)

    amplified = JointPositionRangeAction(cfg.replace(scale={"j2": 2.0, "j0": 0.5}), env)
    amplified.process_actions(raw)
    amplified.apply_actions()
    torch.testing.assert_close(asset.targets, torch.tensor([[-1.0, 0.0], [5.0, -1.0], [-7.0, 1.0]]))


def test_joint_range_rejects_unbounded_joint_limits():
    """Continuous joints cannot define a finite normalized position range."""
    asset = ArticulationFixture()
    env = SimpleNamespace(scene={"robot": asset}, num_envs=3, device="cpu")
    cfg = MirrorJointPositionRangeActionCfg(asset_name="robot", joint_names=["j1"], mirror=mirror_identity)
    with pytest.raises(ValueError, match="finite, ordered"):
        JointPositionRangeAction(cfg, env)
