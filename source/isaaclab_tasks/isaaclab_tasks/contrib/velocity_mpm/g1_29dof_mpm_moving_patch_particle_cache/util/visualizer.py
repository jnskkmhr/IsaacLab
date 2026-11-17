# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task viewers publish the cached sand surface before both window and offline camera rendering."""

import numpy as np
from isaaclab_ov.renderers import OVRTXRendererCfg
from isaaclab_visualizers.newton import NewtonGLBackendCfg, NewtonRTXBackendCfg
from isaaclab_visualizers.newton.newton_visualizer import NewtonGLVisualizer, NewtonRTXVisualizer

from isaaclab.utils import configclass

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


class MovingPatchRTXVisualizer(ParticleCacheVisualizerMixin, NewtonRTXVisualizer):
    """RTX view, including offline video, of the same surface.

    DLSS frame generation yields an extra image for every render. It is presented before the real frame, which
    doubles the displayed frame rate at no simulation cost.
    """

    def _intermediate_image(self):
        camera = self.current_image_view.camera
        if camera is None:
            return None
        path = camera.cfg.prim_path
        for renderer in self._renderers:
            for data in renderer._camera_render_data:
                if data.intermediate_rgba is not None and any(
                    path.startswith(p) or p == path for p in data.camera_paths
                ):
                    return data.intermediate_rgba
        return None

    def _log_streaming_image(self) -> None:
        composite = self.current_image_view.frame.data
        if composite is None or not self._viewer.is_paused():
            composite = self.current_image_view.read(self._sim.get_physics_step_count())
        extra = self._intermediate_image() if not self._viewer.is_paused() else None
        if extra is not None and composite is not None and extra.shape[1:3] == tuple(composite.shape[:2]):
            self._viewer.log_image("Streaming View", extra[0], fullscreen=True)
            # Present the generated image now; the real frame follows in the regular end of the frame.
            self._viewer.end_frame()
            self._viewer.begin_frame(self._sim_time)
        if composite is not None:
            self._viewer.log_image("Streaming View", composite, fullscreen=True)


@configclass
class MovingPatchGLBackendCfg(NewtonGLBackendCfg):
    class_type: str = f"{_MODULE}:MovingPatchGLVisualizer"
    show_particles: bool = False


@configclass
class MovingPatchRTXBackendCfg(NewtonRTXBackendCfg):
    class_type: str = f"{_MODULE}:MovingPatchRTXVisualizer"
    renderer_cfg: OVRTXRendererCfg = OVRTXRendererCfg(async_rendering=True)
    """Pipeline the RTX render with the next physics step, at the price of one capture of latency."""
    render_settings: dict = {
        # The Newton RTX viewer's interactive profile: one sample per pixel without denoising, and DLSS frame
        # generation, which produces an extra image per render.
        "omni:rtx:dlss:frameGeneration": ("Bool", True),
        "omni:rtx:pt:samplesPerPixel": ("UInt", 1),
        "omni:rtx:pt:denoising:enabled": ("Bool", False),
        "omni:rtx:pt:maxSamplesPerLaunch": ("Int", 2073600),
        "omni:rtx:ambientOcclusion:denoiserMode": ("Token", "none"),
        "omni:rtx:directLighting:sampledLighting:denoisingTechnique": ("Token", "None"),
        "omni:rtx:indirectDiffuse:denoiser:enabled": ("Bool", False),
        "omni:rtx:reflections:denoiser:enabled": ("Bool", False),
        "omni:rtx:rt:demoire": ("Bool", False),
        "omni:rtx:rt:sss:samples": ("Int", 1),
        "omni:rtx:post:aa:limitedOps": ("Bool", False),
    }
