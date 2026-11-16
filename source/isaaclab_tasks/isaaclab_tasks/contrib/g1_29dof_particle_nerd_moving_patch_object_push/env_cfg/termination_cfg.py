# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fall, invalid-state, and moving-patch bounds checks."""

import math

import isaaclab.envs.mdp as mdp
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from .. import mdp as push_mdp


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": math.pi / 3})
    solver_diverged = DoneTerm(func=push_mdp.invalid_state, params={"asset_name": "robot"})
    object_invalid = DoneTerm(func=push_mdp.invalid_state, params={"asset_name": "object"})
    object_outside_patch = DoneTerm(
        func=push_mdp.object_outside_patch,
        params={"object_name": "object", "half_extents": (0.325, 0.325, 0.325), "margin": 0.05},
    )
