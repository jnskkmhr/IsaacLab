# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Body-orientation rewards for Particle NeRD locomotion."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def body_flat_orientation_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Sum squared horizontal gravity projections in selected link frames, independent of gravity magnitude."""
    asset = env.scene[asset_cfg.name]
    quaternion = asset.data.body_link_quat_w.torch[:, asset_cfg.body_ids]
    gravity = torch.nn.functional.normalize(asset.data.GRAVITY_VEC_W.torch, dim=-1)
    projected_gravity = quat_apply_inverse(quaternion, gravity[:, None, :].expand(-1, quaternion.shape[1], -1))
    return projected_gravity[..., :2].square().sum(dim=(1, 2))
