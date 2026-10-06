# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Standard particle rendering with task-local visibility filtering."""

import numpy as np
import torch
import warp as wp
from isaaclab_visualizers.kit import KitVisualizer, KitVisualizerCfg
from isaaclab_visualizers.newton import NewtonGLVisualizerCfg, NewtonRTXVisualizerCfg
from isaaclab_visualizers.newton.newton_visualizer import (
    NewtonGLVisualizer,
    NewtonRTXVisualizer,
    NewtonViewerGL,
    NewtonViewerRTX,
)

from pxr import UsdGeom, Vt

import isaaclab.sim as sim_utils
from isaaclab.scene_data import SceneDataFormat
from isaaclab.utils import configclass

from .kernel import gather_visible_particles, mark_visible_particles
from .particles import MovingPatchParticles


class MovingPatchParticleRendering:
    """Filter the standard particle batch without creating a second point cloud."""

    def log_state(self, state):
        """Render particles and track the configured body's world position."""
        super().log_state(state)
        visualizer = self.moving_patch_visualizer
        body_path = visualizer.cfg.follow_body_path
        if body_path is None or self.model is None:
            return

        follow_key = (id(self.model), body_path)
        if getattr(self, "_follow_body_key", None) != follow_key:
            self._follow_body_index = self.model.body_label.index(body_path)
            self._follow_body_key = follow_key

        body_position = state.body_q[self._follow_body_index : self._follow_body_index + 1].numpy()[0, :3]
        eye = tuple(body_position + np.asarray(visualizer.cfg.eye))
        target = tuple(body_position + np.asarray(visualizer.cfg.lookat))
        visualizer._apply_camera_pose((eye, target))

    def _log_particles(self, state):
        patch = getattr(self.moving_patch_visualizer, "moving_patch_particle", None)
        if patch is None:
            super()._log_particles(state)
            return
        model = patch.model
        if getattr(self, "_particle_filter_model", None) is not model:
            self._particle_filter_model = model
            count = model.particle_count
            device = state.particle_q.device
            self._particle_mask = wp.empty(count, dtype=int, device=device)
            self._particle_offsets = wp.empty(count, dtype=int, device=device)
            self._visible_positions = wp.empty(count, dtype=wp.vec3, device=device)
            self._visible_radii = wp.empty(count, dtype=float, device=device)
            self._particle_colors = wp.full(count, wp.vec3(0.7, 0.6, 0.4), dtype=wp.vec3, device=device)
            self._particle_filter_world_mask = wp.zeros(model.world_count, dtype=int, device=device)
            self._particle_filter_world_ids = None
            self._empty_positions = wp.empty(0, dtype=wp.vec3, device=device)

        if not self.show_particles or model.particle_count == 0:
            self.log_points("/model/particles", points=self._empty_positions, hidden=True)
            return

        visible = self.moving_patch_visualizer._resolved_visible_env_ids
        if visible is None and patch.terrain.show_boundary_particles:
            self.log_points(
                "/model/particles",
                points=state.particle_q,
                radii=model.particle_radius,
                colors=self._particle_colors,
            )
            return

        if visible is not None:
            world_ids = tuple(visible)
            if world_ids != self._particle_filter_world_ids:
                selected = set(world_ids)
                self._particle_filter_world_mask.assign([int(i in selected) for i in range(model.world_count)])
                self._particle_filter_world_ids = world_ids

        wp.launch(
            mark_visible_particles,
            dim=model.particle_count,
            inputs=[
                model.particle_world,  # worlds
                patch.dynamic,  # dynamic
                self._particle_filter_world_mask,  # visible_worlds
                visible is not None,  # filter_worlds
                patch.terrain.show_boundary_particles,  # show_boundary
                self._particle_mask,  # mask
            ],
            device=state.particle_q.device,
        )
        wp.utils.array_scan(self._particle_mask, self._particle_offsets, inclusive=True)
        # Standard log_points needs an array length; only the count returns to the CPU.
        count = int(self._particle_offsets[-1:].numpy()[0])
        if count == 0:
            self.log_points("/model/particles", points=self._empty_positions, hidden=True)
            return
        wp.launch(
            gather_visible_particles,
            dim=model.particle_count,
            inputs=[
                self._particle_mask,  # mask
                self._particle_offsets,  # offsets
                state.particle_q,  # positions
                model.particle_radius,  # radii
                self._visible_positions,  # visible_positions
                self._visible_radii,  # visible_radii
            ],
            device=state.particle_q.device,
        )
        self.log_points(
            "/model/particles",
            points=self._visible_positions[:count],
            radii=self._visible_radii[:count],
            colors=self._particle_colors[:count],
        )


class MovingPatchViewerGL(MovingPatchParticleRendering, NewtonViewerGL):
    """Standard GL viewer with moving-terrain particle filtering."""


class MovingPatchViewerRTX(MovingPatchParticleRendering, NewtonViewerRTX):
    """Standard RTX viewer with moving-terrain particle filtering."""


class MovingPatchGLVisualizer(NewtonGLVisualizer):
    """OpenGL rendering for the experimental moving terrain."""

    moving_patch_particle: MovingPatchParticles | None = None

    def _create_viewer(self, runtime_headless: bool, metadata: dict) -> MovingPatchViewerGL:
        viewer = MovingPatchViewerGL(
            width=self.cfg.window_width,
            height=self.cfg.window_height,
            headless=runtime_headless,
            metadata=metadata,
            update_frequency=self.cfg.update_frequency,
        )
        viewer.moving_patch_visualizer = self
        return viewer


class MovingPatchRTXVisualizer(NewtonRTXVisualizer):
    """RTX rendering for the experimental moving terrain."""

    moving_patch_particle: MovingPatchParticles | None = None

    def _create_viewer(self, runtime_headless: bool, metadata: dict) -> MovingPatchViewerRTX:
        viewer = MovingPatchViewerRTX(
            width=self.cfg.window_width,
            height=self.cfg.window_height,
            headless=runtime_headless,
            up_axis="Z",
            metadata=metadata,
            update_frequency=self.cfg.update_frequency,
            environment=self.cfg.rtx_environment,
            background_color=self.cfg.background_color,
            render_settings=self.cfg.render_settings,
        )
        viewer.moving_patch_visualizer = self
        return viewer


class MovingPatchKitVisualizer(KitVisualizer):
    """Filter existing USD particle clouds while Kit updates their positions through Fabric."""

    moving_patch_particle: MovingPatchParticles | None = None

    def __init__(self, cfg: "MovingPatchKitVisualizerCfg"):
        super().__init__(cfg)
        self._particle_width_bindings = None

    def _update_particle_visibility(self) -> None:
        patch = self.moving_patch_particle
        if patch is None:
            return
        if self._particle_width_bindings is None:
            self._particle_width_bindings = []
            stage = sim_utils.get_current_stage()
            for source, ranges in self._scene_data_provider.backend.get_geometry_batches():
                if source._cls is not SceneDataFormat.Points:
                    continue
                for path, (start, count) in ranges.items():
                    points = UsdGeom.Points(stage.GetPrimAtPath(path))
                    if points:
                        widths = points.GetWidthsAttr()
                        self._particle_width_bindings.append((widths, start, count, widths.Get()))

        widths = 2.0 * wp.to_torch(patch.model.particle_radius)
        if not self.cfg.show_particles:
            widths.zero_()
        else:
            if not patch.terrain.show_boundary_particles:
                widths *= wp.to_torch(patch.dynamic) != 0
            visible_env_ids = self._resolved_visible_env_ids
            if visible_env_ids is not None:
                worlds = wp.to_torch(patch.model.particle_world)
                widths *= torch.isin(worlds, torch.tensor(visible_env_ids, device=worlds.device, dtype=worlds.dtype))
        widths = widths.cpu().numpy()
        for attr, start, count, _ in self._particle_width_bindings:
            attr.Set(Vt.FloatArray.FromNumpy(widths[start : start + count]))

    def step(self, dt: float) -> None:
        """Update particle visibility before Kit renders the scene."""
        if not self._runtime_headless:
            self._update_particle_visibility()
        super().step(dt)

    def render_rgb_array(self) -> np.ndarray:
        """Capture Kit RGB output with current particle visibility, including headless mode."""
        self._update_particle_visibility()
        return super().render_rgb_array()

    def close(self) -> None:
        """Restore shared USD particle widths and release Kit resources."""
        if self._particle_width_bindings is not None:
            for attr, _, _, original in self._particle_width_bindings:
                attr.Set(original)
            self._particle_width_bindings = None
        super().close()


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

    follow_body_path: str | None = None
    """Body prim path to follow, or ``None`` for a fixed camera.

    When set, ``eye`` and ``lookat`` are world-axis offsets from the body's position.
    The camera follows translation without rotating with the body.
    """
    show_particles: bool = True


@configclass
class MovingPatchRTXVisualizerCfg(NewtonRTXVisualizerCfg):
    """Use the task-local RTX subclass while keeping IsaacLab's visualizer configuration."""

    class_type: type = MovingPatchRTXVisualizer

    follow_body_path: str | None = None
    """Body prim path to follow, or ``None`` for a fixed camera.

    When set, ``eye`` and ``lookat`` are world-axis offsets from the body's position.
    The camera follows translation without rotating with the body.
    """
    show_particles: bool = True
