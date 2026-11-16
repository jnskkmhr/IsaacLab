# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Termination configuration for the `box` two-way coupling task."""

from isaaclab.envs.mdp import time_out
from isaaclab.managers import TerminationTermCfg
from isaaclab.utils import configclass


@configclass
class BoxTerminationsCfg:
    """Episode timeout termination."""

    time_out = TerminationTermCfg(func=time_out, time_out=True)
