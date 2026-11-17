# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task viewers publish the cached sand surface before both window and offline camera rendering."""

from isaaclab_visualizers.newton import NewtonGLBackendCfg, NewtonRTXBackendCfg
from isaaclab_visualizers.newton.newton_visualizer import NewtonGLVisualizer, NewtonRTXVisualizer

from isaaclab.utils import configclass

_MODULE = "isaaclab_tasks.contrib.velocity_mpm.g1_29dof_mpm_moving_patch_particle_cache.util.visualizer"


class ParticleCacheVisualizerMixin:
    """Render the cached sand surface of ``sand_surface`` before every frame the viewer produces."""

    sand_surface = None
    sand_show_surface = True

    def _render_frame(self):
        if self.sand_surface is not None:
            self.sand_surface.render(self)
        super()._render_frame()

    def render_rgb_array(self):
        if self.sand_surface is not None:
            self.sand_surface.render(self)
        return super().render_rgb_array()

    def _capture_frame(self, *args, **kwargs):
        if self.sand_surface is not None:
            self.sand_surface.render(self)
        return super()._capture_frame(*args, **kwargs)


class MovingPatchGLVisualizer(ParticleCacheVisualizerMixin, NewtonGLVisualizer):
    """GL view of the moving-patch isosurface and the background terrain."""


class MovingPatchRTXVisualizer(ParticleCacheVisualizerMixin, NewtonRTXVisualizer):
    """RTX view, including offline video, of the same surface."""


@configclass
class MovingPatchGLBackendCfg(NewtonGLBackendCfg):
    class_type: str = f"{_MODULE}:MovingPatchGLVisualizer"
    show_particles: bool = False


@configclass
class MovingPatchRTXBackendCfg(NewtonRTXBackendCfg):
    class_type: str = f"{_MODULE}:MovingPatchRTXVisualizer"
