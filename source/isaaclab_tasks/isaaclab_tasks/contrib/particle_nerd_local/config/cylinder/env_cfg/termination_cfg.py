# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Termination configuration for the `cylinder` two-way coupling task."""

from isaaclab.envs.mdp import time_out
from isaaclab.managers import TerminationTermCfg
from isaaclab.utils import configclass


@configclass
class CylinderTerminationsCfg:
    """Episode timeout termination."""

    time_out = TerminationTermCfg(func=time_out, time_out=True)
