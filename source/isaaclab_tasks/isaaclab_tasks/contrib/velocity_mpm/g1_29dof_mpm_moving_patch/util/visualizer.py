# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Standard particle rendering with moving-patch visibility filtering."""

import warp as wp
from isaaclab_visualizers.newton import NewtonGLVisualizerCfg, NewtonRTXVisualizerCfg

from isaaclab.utils import configclass

from .kernel import gather_visible_particles, mark_visible_particles


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
            viewer.log_points("/model/particles", points=state.particle_q, radii=model.particle_radius, colors=self.colors)
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


@configclass
class MovingPatchGLVisualizerCfg(NewtonGLVisualizerCfg):
    """GL visualizer configuration; the particle filter is attached by the environment."""


@configclass
class MovingPatchRTXVisualizerCfg(NewtonRTXVisualizerCfg):
    """RTX visualizer configuration; the particle filter is attached by the environment."""
