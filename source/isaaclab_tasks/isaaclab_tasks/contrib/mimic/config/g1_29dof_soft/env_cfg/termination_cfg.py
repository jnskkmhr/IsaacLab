# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Soft-task terminations with stance-aware tracking checks."""

from isaaclab.utils import configclass

from isaaclab_tasks.contrib.mimic.config.g1_29dof.env_cfg.termination_cfg import (
    G1TerminationsCfg as RigidTerminationsCfg,
)

from .. import mdp as soft_mdp


@configclass
class G1TerminationsCfg(RigidTerminationsCfg):
    """Retain fall checks while allowing tracking deviations during stance."""

    def __post_init__(self):
        self.anchor_pos.func = soft_mdp.bad_anchor_pos_z_only
        self.anchor_ori.func = soft_mdp.bad_anchor_ori
        self.ee_body_pos.func = soft_mdp.bad_motion_body_pos_z_only
