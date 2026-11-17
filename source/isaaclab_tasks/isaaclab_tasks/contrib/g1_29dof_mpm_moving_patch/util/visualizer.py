# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task viewers publish the sand surface before both window and offline camera rendering."""

from isaaclab_visualizers.newton.newton_visualizer import NewtonGLVisualizer, NewtonRTXVisualizer


class SandVisualizerMixin:
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


class SandGLVisualizer(SandVisualizerMixin, NewtonGLVisualizer):
    """GL view of the cached sand surface and optional active particle glyphs."""


class SandRTXVisualizer(SandVisualizerMixin, NewtonRTXVisualizer):
    """RTX view, including offline video, of the same cached sand surface."""
