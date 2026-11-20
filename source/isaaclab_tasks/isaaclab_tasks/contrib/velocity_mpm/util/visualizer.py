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

from ..rtx_visualizer import BodyPickingMixin, DLSSFramePresentMixin, InteractiveRTXBackendCfg

_MODULE = "isaaclab_tasks.contrib.velocity_mpm.util.visualizer"


@wp.kernel
def mark_visible_particles(
    particle_env_ids: wp.array[int],
    particle_flags: wp.array[int],
    particle_is_dynamic: wp.array[int],
    particle_activity: wp.array[int],
    hide_sleeping: bool,
    env_visibility_mask: wp.array[int],
    filter_envs: bool,
    show_boundary_particles: bool,
    particle_visibility_mask: wp.array[int],
):
    """Select particles by environment, simulated/boundary membership and sleep state without changing physics."""
    particle_id = wp.tid()
    # Spare slots of a tile cache are inactive and parked outside the terrain.
    is_visible = (particle_flags[particle_id] & 1) != 0 and (
        show_boundary_particles or particle_is_dynamic[particle_id] != 0
    )
    # Activity 0 is asleep; support (1) and awake (2) particles are drawn.
    if hide_sleeping and particle_activity[particle_id] == 0:
        is_visible = False
    if filter_envs:
        env_id = particle_env_ids[particle_id]
        if env_id >= 0 and env_id < env_visibility_mask.shape[0]:
            is_visible = is_visible and env_visibility_mask[env_id] != 0
        else:
            is_visible = False
    particle_visibility_mask[particle_id] = wp.where(is_visible, 1, 0)


@wp.kernel
def gather_visible_particles(
    particle_visibility_mask: wp.array[int],
    particle_visible_offsets: wp.array[int],
    particle_q: wp.array[wp.vec3],
    particle_radius: wp.array[float],
    visible_particle_q: wp.array[wp.vec3],
    visible_particle_radius: wp.array[float],
):
    """Compact visible particle positions [m] and radii [m] using inclusive prefix-scan offsets."""
    particle_id = wp.tid()
    if particle_visibility_mask[particle_id] != 0:
        visible_particle_id = particle_visible_offsets[particle_id] - 1
        visible_particle_q[visible_particle_id] = particle_q[particle_id]
        visible_particle_radius[visible_particle_id] = particle_radius[particle_id]


class ParticleVisibilityFilter:
    """Replace a Newton viewer's particle batch by the particles the streaming pool actually simulates.

    Parked reserve slots, sleeping particles, and optionally the kinematic boundary band, are left out. Bind an
    instance with :func:`attach_particle_filter`; it is called with the state to render.
    """

    def __init__(self, streaming, visualizer, show_boundary_particles: bool = False):
        self.patch = streaming
        self.visualizer = visualizer
        self.show_boundary_particles = show_boundary_particles
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
        # Sleeping particles are not drawn; without sleeping the solver exposes no activity.
        activity = patch.solver.particle_activity
        if visible is None and self.show_boundary_particles and activity is None:
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
                model.particle_flags,
                patch.particle_is_dynamic,
                patch.particle_is_dynamic if activity is None else activity,
                activity is not None,
                self.world_mask,
                visible is not None,
                self.show_boundary_particles,
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


def attach_particle_filter(visualizer, streaming, show_boundary_particles: bool = False) -> None:
    """Make a Newton visualizer draw only the streaming pool's simulated particles."""
    visualizer._viewer._log_particles = ParticleVisibilityFilter(streaming, visualizer, show_boundary_particles)


class CameraFollow:
    """Move a visualizer's camera with a body, as the previous ``origin_type="asset"`` view did.

    The view starts at the configured offsets from the body. Afterwards the camera is translated by the body's
    displacement, so the view the user sets with the mouse and keyboard is kept.
    """

    def __init__(self, backend, body_index: int, eye_offset, lookat_offset):
        self.backend = backend
        self.body_index = body_index
        self.eye = np.asarray(eye_offset, dtype=np.float32)
        self.lookat = np.asarray(lookat_offset, dtype=np.float32)
        self.previous = None

    def __call__(self, visualizer) -> None:
        position = self.backend.state_0.body_q[self.body_index : self.body_index + 1].numpy()[0, :3]
        if not np.isfinite(position).all():
            return  # the robot diverged; keep the last view until it resets
        viewer = visualizer._viewer
        if self.previous is None or viewer is None:
            visualizer.set_camera_view(tuple(position + self.eye), tuple(position + self.lookat))
        else:
            camera = viewer.camera
            eye = np.asarray(tuple(camera.pos), dtype=float) + position - self.previous
            target = eye + np.asarray(tuple(camera.get_front()), dtype=float) * camera.pivot_distance
            visualizer.set_camera_view(tuple(eye), tuple(target))
        self.previous = position


class CameraFollowMixin:
    """Move the camera with the tracked robot and publish the sand surface before every frame the viewer produces."""

    camera_follow = None
    sand_surface = None
    sand_show_surface = True

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


class MovingPatchGLVisualizer(CameraFollowMixin, NewtonGLVisualizer):
    """GL view that can follow the robot."""


class MovingPatchRTXVisualizer(CameraFollowMixin, BodyPickingMixin, DLSSFramePresentMixin, NewtonRTXVisualizer):
    """RTX view that can follow the robot, presents the DLSS-generated frames and lets rigid bodies be dragged."""


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
