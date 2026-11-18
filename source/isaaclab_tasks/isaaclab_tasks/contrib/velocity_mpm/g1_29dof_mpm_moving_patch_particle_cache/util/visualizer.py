# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task viewers publish the cached sand surface before both window and offline camera rendering."""

import numpy as np
from isaaclab_visualizers.newton import NewtonGLBackendCfg
from isaaclab_visualizers.newton.newton_visualizer import NewtonGLVisualizer, NewtonRTXVisualizer

from isaaclab.utils import configclass

from ...rtx_visualizer import DLSSFramePresentMixin, InteractiveRTXBackendCfg

_MODULE = "isaaclab_tasks.contrib.velocity_mpm.g1_29dof_mpm_moving_patch_particle_cache.util.visualizer"


class CameraFollow:
    """Keep a visualizer's camera at fixed offsets from a body, as the task's playback view did."""

    def __init__(self, backend, body_index: int, eye_offset, lookat_offset):
        self.backend = backend
        self.body_index = body_index
        self.eye = np.asarray(eye_offset, dtype=np.float32)
        self.lookat = np.asarray(lookat_offset, dtype=np.float32)

    def __call__(self, visualizer) -> None:
        position = self.backend.state_0.body_q[self.body_index : self.body_index + 1].numpy()[0, :3]
        if not np.isfinite(position).all():
            return  # the robot diverged; keep the last view until it resets
        visualizer.set_camera_view(tuple(position + self.eye), tuple(position + self.lookat))


class ParticleCacheVisualizerMixin:
    """Render the cached sand surface of ``sand_surface`` before every frame the viewer produces."""

    sand_surface = None
    sand_show_surface = True
    camera_follow = None

    def _prepare_frame(self):
        if self.camera_follow is not None:
            self.camera_follow(self)
        if self.sand_surface is not None:
            self.sand_surface.render(self)

    def _render_frame(self):
        self._prepare_frame()
        super()._render_frame()

    def render_rgb_array(self):
        self._prepare_frame()
        return super().render_rgb_array()

    def _capture_frame(self, *args, **kwargs):
        self._prepare_frame()
        return super()._capture_frame(*args, **kwargs)


class MovingPatchGLVisualizer(ParticleCacheVisualizerMixin, NewtonGLVisualizer):
    """GL view of the moving-patch isosurface and the background terrain."""


class MovingPatchRTXVisualizer(ParticleCacheVisualizerMixin, DLSSFramePresentMixin, NewtonRTXVisualizer):
    """RTX view, including offline video, of the same surface."""


@configclass
class MovingPatchGLBackendCfg(NewtonGLBackendCfg):
    class_type: str = f"{_MODULE}:MovingPatchGLVisualizer"
    show_particles: bool = False


@configclass
class MovingPatchRTXBackendCfg(InteractiveRTXBackendCfg):
    class_type: str = f"{_MODULE}:MovingPatchRTXVisualizer"
