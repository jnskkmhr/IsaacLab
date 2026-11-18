# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Interactive RTX viewer shared by the MPM locomotion tasks: asynchronous rendering with DLSS frame generation."""

from isaaclab_ov.renderers import OVRTXRendererCfg
from isaaclab_visualizers.newton import NewtonRTXBackendCfg
from isaaclab_visualizers.newton.newton_visualizer import NewtonRTXVisualizer

from isaaclab.utils import configclass


class DLSSFramePresentMixin:
    """Present the DLSS-generated image before the real frame.

    DLSS frame generation yields an extra image for every render. Presenting it first doubles the displayed frame
    rate at no simulation cost.
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


class InteractiveRTXVisualizer(DLSSFramePresentMixin, NewtonRTXVisualizer):
    """RTX window that presents the DLSS-generated frames."""


@configclass
class InteractiveRTXBackendCfg(NewtonRTXBackendCfg):
    """RTX backend with the Newton viewer's interactive profile: asynchronous rendering and DLSS frame generation."""

    class_type: str = "isaaclab_tasks.contrib.velocity_mpm.rtx_visualizer:InteractiveRTXVisualizer"

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
