# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Events of the G1 29-DoF locomotion tasks among trees."""

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

from ...g1_29dof_rigid.env_cfg.event_cfg import G1EventCfg
from .. import mdp as tree_mdp
from .bar_scene_cfg import ROD_ASSET_NAMES
from .scene_cfg import LOG_ASSET_NAMES


@configclass
class G1TreeEventCfg(G1EventCfg):
    """Events of the rigid flat task, plus the pile reset."""

    # A cable is not reset by the asset itself, so piles the robot kicked apart would stay apart
    # for the rest of training.
    reset_cables = EventTerm(
        func=tree_mdp.reset_cables,
        mode="reset",
        params={"asset_cfgs": [SceneEntityCfg(name) for name in LOG_ASSET_NAMES]},
    )


@configclass
class G1BarEventCfg(G1EventCfg):
    """Events of the rigid flat task, plus the rod reset."""

    # The rods are welded at their ends, but a swing that a step left behind would otherwise carry
    # into the next episode.
    reset_cables = EventTerm(
        func=tree_mdp.reset_cables,
        mode="reset",
        params={"asset_cfgs": [SceneEntityCfg(name) for name in ROD_ASSET_NAMES]},
    )
