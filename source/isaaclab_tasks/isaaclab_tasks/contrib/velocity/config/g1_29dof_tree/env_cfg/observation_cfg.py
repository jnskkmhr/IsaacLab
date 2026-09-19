# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observations of the G1 29-DoF locomotion task among trees.

The groups are the rigid task's. Only the three privileged contact terms are replaced, because the
coupled solver carries no contact sensor: they are re-derived from the foot kinematics in
:mod:`..mdp`. The term order, shapes and units are unchanged, so a critic trained on the rigid task
still matches this layout.
"""

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

from ...g1_29dof_rigid.env_cfg.observation_cfg import G1ObservationsCfg, PrivilegedHistoryCfg
from .. import mdp as tree_mdp

FOOT_CFG = SceneEntityCfg("robot", body_names=".*_ankle_roll_link")
"""Feet the contact terms are reported for."""


@configclass
class TreePrivilegedObsCfg(PrivilegedHistoryCfg):
    """Privileged group with the contact terms served by the foot kinematics."""

    foot_contact = ObsTerm(func=tree_mdp.foot_contact, params={"asset_cfg": FOOT_CFG})
    foot_contact_force = ObsTerm(func=tree_mdp.foot_contact_forces, params={"asset_cfg": FOOT_CFG})
    foot_air_time = ObsTerm(func=tree_mdp.foot_air_time, params={"asset_cfg": FOOT_CFG})


@configclass
class G1TreeObservationsCfg(G1ObservationsCfg):
    """Observation specifications for the MDP."""

    privileged: TreePrivilegedObsCfg = TreePrivilegedObsCfg()
