# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task-specific viewer classes; preserve core camera configuration and controls."""

from isaaclab_visualizers.newton import NewtonGLBackendCfg, NewtonRTXBackendCfg

from isaaclab.utils.configclass import configclass


@configclass
class SandGLBackendCfg(NewtonGLBackendCfg):
    class_type: str = "isaaclab_tasks.contrib.g1_29dof_mpm_moving_patch.util.visualizer:SandGLVisualizer"
    show_particles: bool = False


@configclass
class SandRTXBackendCfg(NewtonRTXBackendCfg):
    class_type: str = "isaaclab_tasks.contrib.g1_29dof_mpm_moving_patch.util.visualizer:SandRTXVisualizer"
