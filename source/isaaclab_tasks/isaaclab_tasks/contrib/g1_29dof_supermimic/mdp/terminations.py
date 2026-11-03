# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Illegal contacts measured from both rigid and MPM forces."""

from isaaclab.managers import SceneEntityCfg


def illegal_contact(env, threshold: float, asset_cfg: SceneEntityCfg):
    """Terminate when a selected body has more than threshold N of coupled reaction."""
    forces = env.body_contact_force_history[:, :, asset_cfg.body_ids]
    return (forces.norm(dim=-1).amax(dim=1) > threshold).any(dim=-1)
