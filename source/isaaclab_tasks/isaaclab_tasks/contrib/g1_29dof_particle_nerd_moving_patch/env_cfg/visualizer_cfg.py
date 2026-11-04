# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Viewer settings for the fixed-size moving particle patch."""

from isaaclab_visualizers.kit import KitVisualizerCfg
from isaaclab_visualizers.newton import NewtonGLVisualizerCfg, NewtonRTXVisualizerCfg

from isaaclab.utils import configclass

from isaaclab_tasks.contrib.g1_29dof_mpm_moving_patch.util.visualizer import (
    MovingPatchGLVisualizer,
    MovingPatchKitVisualizer,
    MovingPatchRTXVisualizer,
)


@configclass
class MovingPatchKitVisualizerCfg(KitVisualizerCfg):
    """Use Kit's particle clouds with moving-patch boundary and environment filtering."""

    class_type: type = MovingPatchKitVisualizer
    show_particles: bool = True
    """Whether to display moving-patch particles."""


@configclass
class MovingPatchGLVisualizerCfg(NewtonGLVisualizerCfg):
    """Use the task-local GL subclass while keeping IsaacLab's visualizer configuration."""

    class_type: type = MovingPatchGLVisualizer

    show_particles: bool = True


@configclass
class MovingPatchRTXVisualizerCfg(NewtonRTXVisualizerCfg):
    """Use the task-local RTX subclass while keeping IsaacLab's visualizer configuration."""

    class_type: type = MovingPatchRTXVisualizer

    show_particles: bool = True
