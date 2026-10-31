# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Viewer-only material-on-body force arrows for the last contact solve."""

import torch
import warp as wp
from isaaclab_visualizers.newton.newton_visualization_markers import NewtonVisualizationMarkers

from isaaclab.markers.config import RED_ARROW_X_MARKER_CFG
from isaaclab.utils.math import quat_from_euler_xyz

from particle_nerd.force_estimation.kernels import geometry
from particle_nerd.shovel.sdf import surface


@wp.kernel
def contact_arrows(
    ids: wp.array(dtype=int),
    position: wp.array(dtype=wp.vec3),
    pose: wp.array(dtype=wp.transform),
    size: wp.array(dtype=wp.vec3),
    shape: wp.array(dtype=int),
    frames: wp.array(dtype=wp.mat33d),
    impulses: wp.array(dtype=wp.vec3d),
    sdf: wp.array3d(dtype=float),
    sdf_origin: wp.vec3,
    sdf_spacing: float,
    dt: float,
    scale: float,
    starts: wp.array(dtype=wp.vec3),
    ends: wp.array(dtype=wp.vec3),
):
    k = wp.tid()
    i = ids[k]
    distance, normal, point = geometry(wp.vec3d(position[i]), pose[0], size[0], shape[0])
    if shape[0] == 3:
        local = wp.transform_point(wp.transform_inverse(pose[0]), position[i])
        d, n = surface(local, sdf, sdf_origin, sdf_spacing)
        normal = wp.vec3d(wp.quat_rotate(wp.transform_get_rotation(pose[0]), n))
        point = wp.vec3d(position[i]) - wp.float64(d) * normal
    force = wp.vec3(wp.transpose(frames[i]) * impulses[i]) / dt
    starts[k] = wp.vec3(point)
    ends[k] = starts[k] + scale * force


@wp.kernel
def resultant_arrow(
    pose: wp.array(dtype=wp.transform),
    wrench: wp.array(dtype=wp.spatial_vector),
    scale: float,
    starts: wp.array(dtype=wp.vec3),
    ends: wp.array(dtype=wp.vec3),
):
    starts[0] = wp.transform_get_translation(pose[0])
    ends[0] = starts[0] + scale * wp.spatial_top(wrench[0])


class ContactForceVisualizer:
    """Material force vectors using IsaacLab's standard Newton arrow markers."""

    def __init__(self, viewer, simulation, enabled=False):
        self.simulation = simulation
        self.enabled, self.show_total, self.scale = enabled, True, 0.001
        self.pose = wp.clone(simulation.rigid.state.body_q)
        self.starts = wp.zeros(simulation.estimator.n, dtype=wp.vec3, device=simulation.device)
        self.ends = wp.zeros_like(self.starts)
        self.total_start = wp.empty(1, dtype=wp.vec3, device=simulation.device)
        self.total_end = wp.empty_like(self.total_start)
        self.capacity = 0
        self.contact_markers = None
        self.total_markers = None
        viewer.register_ui_callback(self.gui, position="rendering")

    def gui(self, ui):
        _, self.enabled = ui.checkbox("Estimated material contact forces", self.enabled)
        if self.enabled:
            _, self.show_total = ui.checkbox("Total material force at body COM", self.show_total)
            _, self.scale = ui.slider_float("Force arrow scale (m/N)", self.scale, 0.0001, 0.01, "%.4f")
            ui.text("Red: contact forces. Thicker arrow: total force.")
            ui.text("Last solve; excludes mouse and rigid-ground forces.")

    def _create_markers(self):
        # This runner owns a Newton viewer, not an IsaacLab SimulationContext.
        # Use the same marker backend as VisualizationMarkers without creating
        # a second simulation context. The backend owns all meshes and instances.
        cfg = RED_ARROW_X_MARKER_CFG.copy()
        cfg.prim_path = "/Visuals/material_contact_forces"
        cfg.markers["arrow"].scale = (1.0, 0.0125, 0.0125)
        cfg.markers["arrow"].visual_material.diffuse_color = (1.0, 0.0, 0.0)
        self.contact_markers = NewtonVisualizationMarkers(cfg)
        total_cfg = cfg.copy()
        total_cfg.prim_path = "/Visuals/total_material_force"
        total_cfg.markers["arrow"].scale = (1.0, 0.025, 0.025)
        total_cfg.markers["arrow"].visual_material.diffuse_color = (1.0, 0.0, 0.0)
        self.total_markers = NewtonVisualizationMarkers(total_cfg)

    def _update_markers(self, markers, starts, ends, count, capacity):
        # Like velocity-command markers: rotate the standard X-axis arrow toward
        # the vector and scale its X dimension by magnitude. Include pitch for
        # three-dimensional contact forces. All vector operations stay on GPU.
        positions = wp.to_torch(starts)[:capacity]
        vectors = wp.to_torch(ends)[:capacity] - positions
        vectors = torch.where((torch.arange(capacity, device=vectors.device) < count).unsqueeze(-1), vectors, 0.0)
        lengths = torch.linalg.vector_norm(vectors, dim=-1)
        yaw = torch.atan2(vectors[:, 1], vectors[:, 0])
        pitch = -torch.atan2(vectors[:, 2], torch.linalg.vector_norm(vectors[:, :2], dim=-1))
        orientations = quat_from_euler_xyz(torch.zeros_like(yaw), pitch, yaw)
        scales = torch.ones_like(vectors)
        scales[:, 0] = lengths
        scales = torch.where((lengths > 1.0e-8).unsqueeze(-1), scales, 0.0)
        markers.visualize(positions, orientations, scales, marker_indices=None)

    def draw(self, viewer, count):
        if self.contact_markers is None:
            if not self.enabled:
                return
            self._create_markers()
        self.contact_markers.set_visibility(self.enabled)
        self.total_markers.set_visibility(self.enabled and self.show_total)
        sim = self.simulation
        estimator = sim.estimator
        with torch.cuda.stream(sim.torch_stream):
            if self.enabled:
                # RTX rebuilds a marker batch if its instance count changes.
                # Retain slots; unused arrows have zero scale. Grow only when
                # needed, rather than reallocating at every contact transition.
                self.capacity = min(estimator.n, max(self.capacity, 512, 1 << (max(count, 1) - 1).bit_length()))
                if count:
                    wp.launch(
                        contact_arrows,
                        count,
                        [
                            estimator.ids,
                            sim.old_position,
                            self.pose,
                            estimator.sizes,
                            estimator.shapes,
                            estimator.frames,
                            estimator.impulses,
                            *estimator.sdf,
                            estimator.dt,
                            self.scale,
                            self.starts,
                            self.ends,
                        ],
                        device=sim.device,
                    )
                self._update_markers(self.contact_markers, self.starts, self.ends, count, self.capacity)
                if self.show_total:
                    wp.launch(
                        resultant_arrow,
                        1,
                        [self.pose, estimator.wrench, self.scale, self.total_start, self.total_end],
                        device=sim.device,
                    )
                    self._update_markers(self.total_markers, self.total_start, self.total_end, 1, 1)
            self.contact_markers.render(viewer, visible_env_ids=None, num_envs=1)
            self.total_markers.render(viewer, visible_env_ids=None, num_envs=1)
