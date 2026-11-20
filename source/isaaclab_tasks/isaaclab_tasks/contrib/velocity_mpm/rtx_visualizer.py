# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Interactive RTX viewer shared by the MPM locomotion tasks.

Asynchronous rendering with DLSS frame generation, and right-click dragging of rigid bodies.
"""

import torch
from isaaclab_ov.renderers import OVRTXRendererCfg
from isaaclab_visualizers.newton import NewtonRTXBackendCfg
from isaaclab_visualizers.newton.newton_visualizer import NewtonGLVisualizer, NewtonRTXVisualizer
from newton.viewer import Picking

from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply


class BodyPickingMixin:
    """Right-click dragging of rigid bodies in the RTX window, as in the GL viewer.

    The RTX window shows a rendered image, so the Newton model is not loaded into its viewer. The picker is
    created for the simulated model directly and casts its ray from the window's navigation camera, which
    is first moved onto the camera that renders the image.
    """

    def initialize(self, sim, *, cameras) -> None:
        if self._is_initialized:
            return
        super().initialize(sim, cameras=cameras)
        physics = sim.backend_manager.backend if self.physics_backend == "newton" else None
        supported = physics is not None and physics.solver_ops is not None and physics.solver_ops.applies_body_forces()
        self._picking_physics = physics
        self._picking_enabled = supported and not self._runtime_headless
        self._viewer_picking_binding = NewtonGLVisualizer._ViewerPickingBinding()
        if self._picking_enabled:
            self._viewer.picking = Picking(physics.model)
            self._viewer.picking._load_kernels()
            # The pick force is recorded into the physics graph, so it must be enabled before the first step.
            self._viewer.picking_enabled = True
            self._viewer_picking_binding.bind(self._viewer)
            sim.backend_manager.register_state_force_callback(self._viewer_picking_binding.apply)
            gui = self._viewer.gui
            start_picking = gui.start_picking_from_screen

            def start_picking_from_rendered_view(x, y, to_framebuffer_coords):
                # Only the window's own camera is steered by the navigation camera; sensor views are not pickable.
                if not self.current_image_view.camera.cfg.prim_path.startswith("/World/viewer_camera_"):
                    return False
                self._align_navigation_camera()
                return start_picking(x, y, to_framebuffer_coords)

            gui.start_picking_from_screen = start_picking_from_rendered_view

    def _align_navigation_camera(self) -> None:
        """Move the window's navigation camera onto the camera that renders the displayed image."""
        positions, orientations = self.current_image_view.camera.get_world_poses(convention="opengl")
        # An OpenGL camera looks along its negative Z axis.
        front = quat_apply(orientations[0], torch.tensor([0.0, 0.0, -1.0], device=orientations.device))
        target = positions[0] + front * self._viewer.camera.pivot_distance
        self.set_camera_view(tuple(positions[0].tolist()), tuple(target.tolist()))

    def _render_frame(self) -> None:
        if self._picking_enabled:
            # The picker reads body poses from the state the viewer last saw; this viewer logs none.
            self._viewer._last_state = self._picking_physics.state_0
        super()._render_frame()

    def close(self) -> None:
        if getattr(self, "_picking_enabled", False):
            # Captured graphs retain neutral picking inputs until their last replay.
            self._viewer_picking_binding.deactivate()
        super().close()


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


class InteractiveRTXVisualizer(BodyPickingMixin, DLSSFramePresentMixin, NewtonRTXVisualizer):
    """RTX window that presents the DLSS-generated frames and lets rigid bodies be dragged."""


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
