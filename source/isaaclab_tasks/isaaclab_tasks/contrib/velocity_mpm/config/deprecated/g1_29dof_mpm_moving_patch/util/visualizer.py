# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Standard particle rendering with moving-patch visibility filtering."""

import numpy as np
import warp as wp
from isaaclab_visualizers.newton import NewtonGLBackendCfg
from isaaclab_visualizers.newton.newton_visualizer import NewtonGLVisualizer, NewtonRTXVisualizer

from isaaclab.utils import configclass

from .....rtx_visualizer import DLSSFramePresentMixin, InteractiveRTXBackendCfg
from .kernel import gather_visible_particles, mark_visible_particles

_MODULE = "isaaclab_tasks.contrib.velocity_mpm.config.legacy.g1_29dof_mpm_moving_patch.util.visualizer"


class ParticleVisibilityFilter:
    """Replace a Newton viewer's particle batch by the particles the moving patch actually simulates.

    Parked reserve slots, and optionally the kinematic boundary band, are left out. Bind an instance with
    :func:`attach_particle_filter`; it is called with the state to render.
    """

    def __init__(self, patch, visualizer):
        self.patch = patch
        self.visualizer = visualizer
        self.model = None

    def _allocate(self, state):
        model = self.patch.model
        count = model.particle_count
        device = state.particle_q.device
        self.model = model
        self.mask = wp.empty(count, dtype=int, device=device)
        self.offsets = wp.empty(count, dtype=int, device=device)
        self.positions = wp.empty(count, dtype=wp.vec3, device=device)
        self.radii = wp.empty(count, dtype=float, device=device)
        self.colors = wp.full(count, wp.vec3(0.7, 0.6, 0.4), dtype=wp.vec3, device=device)
        self.world_mask = wp.zeros(model.world_count, dtype=int, device=device)
        self.world_ids = None
        self.empty = wp.empty(0, dtype=wp.vec3, device=device)

    def __call__(self, state):
        viewer = self.visualizer._viewer
        patch = self.patch
        if self.model is not patch.model:
            self._allocate(state)
        model = patch.model
        if not viewer.show_particles or model.particle_count == 0:
            viewer.log_points("/model/particles", points=self.empty, hidden=True)
            return
        visible = getattr(self.visualizer, "_env_ids", None)
        if visible is None and patch.terrain.show_boundary_particles:
            viewer.log_points(
                "/model/particles", points=state.particle_q, radii=model.particle_radius, colors=self.colors
            )
            return
        if visible is not None:
            world_ids = tuple(visible)
            if world_ids != self.world_ids:
                selected = set(world_ids)
                self.world_mask.assign([int(i in selected) for i in range(model.world_count)])
                self.world_ids = world_ids
        wp.launch(
            mark_visible_particles,
            dim=model.particle_count,
            inputs=[
                model.particle_world,
                patch.particle_is_dynamic,
                self.world_mask,
                visible is not None,
                patch.terrain.show_boundary_particles,
                self.mask,
            ],
            device=state.particle_q.device,
        )
        wp.utils.array_scan(self.mask, self.offsets, inclusive=True)
        # Standard log_points needs an array length; only the count returns to the CPU.
        count = int(self.offsets[-1:].numpy()[0])
        if count == 0:
            viewer.log_points("/model/particles", points=self.empty, hidden=True)
            return
        wp.launch(
            gather_visible_particles,
            dim=model.particle_count,
            inputs=[self.mask, self.offsets, state.particle_q, model.particle_radius, self.positions, self.radii],
            device=state.particle_q.device,
        )
        viewer.log_points(
            "/model/particles", points=self.positions[:count], radii=self.radii[:count], colors=self.colors[:count]
        )


def attach_particle_filter(visualizer, patch) -> None:
    """Make a Newton visualizer draw only the moving patch's simulated particles."""
    visualizer._viewer._log_particles = ParticleVisibilityFilter(patch, visualizer)


class CameraFollow:
    """Keep a visualizer's camera at fixed offsets from a body, as the previous ``origin_type="asset"`` view did."""

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


class CameraFollowMixin:
    """Move the camera with the tracked robot before every frame the viewer produces."""

    camera_follow = None

    def _prepare_frame(self):
        if self.camera_follow is not None:
            self.camera_follow(self)

    def _render_frame(self):
        self._prepare_frame()
        super()._render_frame()

    def render_rgb_array(self):
        self._prepare_frame()
        return super().render_rgb_array()

    def _capture_frame(self, *args, **kwargs):
        self._prepare_frame()
        return super()._capture_frame(*args, **kwargs)


class MovingPatchGLVisualizer(CameraFollowMixin, NewtonGLVisualizer):
    """GL view that can follow the robot."""


class MovingPatchRTXVisualizer(CameraFollowMixin, DLSSFramePresentMixin, NewtonRTXVisualizer):
    """RTX view that can follow the robot and presents the DLSS-generated frames."""


@configclass
class MovingPatchGLBackendCfg(NewtonGLBackendCfg):
    """GL backend configuration; the particle filter is attached by the environment."""

    class_type: str = f"{_MODULE}:MovingPatchGLVisualizer"
    follow_robot: bool = False
    """Keep the camera at the configured eye and lookat offsets from the first environment's robot."""


@configclass
class MovingPatchRTXBackendCfg(InteractiveRTXBackendCfg):
    """RTX backend configuration; the particle filter is attached by the environment."""

    class_type: str = f"{_MODULE}:MovingPatchRTXVisualizer"
    follow_robot: bool = False
    """Keep the camera at the configured eye and lookat offsets from the first environment's robot."""
