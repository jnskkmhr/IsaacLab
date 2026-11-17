# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reconstruct teacher inputs without changing the student's joint-position offsets."""

import torch

from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.utils.string import resolve_matching_names_values


class joint_pos_reference(ManagerTermBase):
    """Joint positions relative to explicitly configured teacher defaults, in requested order."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        asset_cfg = cfg.params["asset_cfg"]
        asset = env.scene[asset_cfg.name]
        self.reference = torch.zeros(asset.num_joints, device=env.device)
        ids, _, values = resolve_matching_names_values(cfg.params["reference_positions"], asset.joint_names)
        self.reference[ids] = torch.tensor(values, device=env.device)

    def __call__(self, env, reference_positions: dict[str, float], asset_cfg: SceneEntityCfg):
        return (
            env.scene[asset_cfg.name].data.joint_pos.torch[:, asset_cfg.joint_ids] - self.reference[asset_cfg.joint_ids]
        )


def action_subset(env, indices: list[int]):
    """Read the previous student actions corresponding to one teacher's controlled joints."""
    return env.action_manager.action[:, indices]
