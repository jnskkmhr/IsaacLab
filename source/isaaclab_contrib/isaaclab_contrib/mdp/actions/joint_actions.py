# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Joint position targets scaled by the articulation's joint limits."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.actions import JointPositionAction

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .joint_actions_cfg import JointPositionRangeActionCfg


class JointPositionRangeAction(JointPositionAction):
    """Scale each action around the default pose by the distance to the farther joint limit.

    At scale=1, actions in [-1, 1] can reach either joint limit and may exceed
    the nearer limit. Limits and default positions are captured at initialization.
    Raw actions remain in policy coordinates for observations, rewards and replay.
    """

    def __init__(self, cfg: JointPositionRangeActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        if not cfg.use_default_offset:
            raise ValueError("JointPositionRangeAction requires use_default_offset=True.")
        limits = self._asset.data.joint_pos_limits.torch[:, self._joint_ids]
        if not torch.isfinite(limits).all() or (limits[..., 0] > limits[..., 1]).any():
            raise ValueError("JointPositionRangeAction requires finite, ordered joint position limits.")
        ranges = torch.maximum((limits[..., 0] - self._offset).abs(), (limits[..., 1] - self._offset).abs())
        self._scale = self._scale * ranges
