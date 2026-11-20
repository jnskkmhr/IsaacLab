# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Events of the locomanip debug task: the streaming task's events plus the object resets."""

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from ...g1_29dof import mdp as mpm_mdp
from ...g1_29dof.env_cfg import G1EventCfg


@configclass
class G1LocomanipEventCfg(G1EventCfg):
    """Robot and sand events of the streaming task, and the objects put back on the sand at every reset."""

    reset_box = EventTerm(
        func=mpm_mdp.reset_root_state_on_terrain,  # type: ignore
        mode="reset",
        params={"pose_range": {}, "velocity_range": {}, "asset_cfg": SceneEntityCfg("box")},
    )
    reset_sphere = EventTerm(
        func=mpm_mdp.reset_root_state_on_terrain,  # type: ignore
        mode="reset",
        params={"pose_range": {}, "velocity_range": {}, "asset_cfg": SceneEntityCfg("sphere")},
    )
    reset_cylinder = EventTerm(
        func=mpm_mdp.reset_root_state_on_terrain,  # type: ignore
        mode="reset",
        params={"pose_range": {}, "velocity_range": {}, "asset_cfg": SceneEntityCfg("cylinder")},
    )
