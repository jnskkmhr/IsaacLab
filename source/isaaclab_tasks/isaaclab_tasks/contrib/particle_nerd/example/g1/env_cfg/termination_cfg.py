# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Termination configuration for the `g1` humanoid granular locomotion task."""

from isaaclab.envs.mdp import time_out
from isaaclab.managers import TerminationTermCfg
from isaaclab.utils import configclass


@configclass
class G1TerminationsCfg:
    """Episode timeout termination."""

    time_out = TerminationTermCfg(func=time_out, time_out=True)
