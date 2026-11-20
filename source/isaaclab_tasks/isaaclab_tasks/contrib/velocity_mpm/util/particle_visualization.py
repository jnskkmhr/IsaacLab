# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""GPU particle activity colors for the experimental GL and RTX viewers."""

import re

import numpy as np
import warp as wp
from isaaclab_newton.assets.mpm_object.mpm_object import MPMObject
from isaaclab_newton.sim.spawners.mpm.mpm import _SIMULATION_POINTS_SUFFIX

from pxr import UsdGeom, Vt

from isaaclab.sim import get_current_stage


@wp.kernel
def color_particle_activity(activity: wp.array[int], indices: wp.array[int], colors: wp.array[wp.vec3]):
    i = wp.tid()
    color = wp.vec3(0.35, 0.35, 0.35)  # Sleeping.
    if activity[i] == 1:
        color = wp.vec3(0.0, 0.8, 1.0)  # Stationary support.
    elif activity[i] == 2:
        color = wp.vec3(1.0, 0.25, 0.02)  # Awake.
    colors[indices[i]] = color


class ActivityMPMObject(MPMObject):
    """Declare per-particle USD colors before the RTX renderer imports the scene."""

    def _initialize_impl(self):
        super()._initialize_impl()
        expression = re.compile(self.cfg.prim_path + _SIMULATION_POINTS_SUFFIX)
        stage = get_current_stage()
        for path, (_, count) in self._backend_manager.backend.particle_ranges.items():
            if expression.fullmatch(path):
                points = UsdGeom.Points(
                    stage.GetPrimAtPath(path.removesuffix(_SIMULATION_POINTS_SUFFIX) + "/Particles")
                )
                if points:
                    colors = np.tile(np.array([1.0, 0.25, 0.02], dtype=np.float32), (count, 1))
                    points.CreateDisplayColorPrimvar(UsdGeom.Tokens.vertex).Set(Vt.Vec3fArray.FromNumpy(colors))


class ParticleActivityColors:
    """Publish activity colors at render cadence without changing particle physics.

    Colors are prepared after rendering for the next frame. The full color buffer and
    its per-object views persist; no particle activity is copied back to the CPU.
    """

    def __init__(self, env):
        self.sim = env.sim
        backend = self.sim.backend_manager.backend
        self.entry = backend.solver._entries["sand"]
        self.colors = wp.full(backend.model.particle_count, wp.vec3(1.0, 0.25, 0.02), device=env.device)
        self.paths = []
        self.views = []
        expression = re.compile(env.cfg.scene.sand.prim_path + _SIMULATION_POINTS_SUFFIX)
        stage = get_current_stage()
        for path, (offset, count) in backend.particle_ranges.items():
            if not expression.fullmatch(path):
                continue
            visual_path = path.removesuffix(_SIMULATION_POINTS_SUFFIX) + "/Particles"
            points = UsdGeom.Points(stage.GetPrimAtPath(visual_path))
            if not points:
                continue
            colors = self.colors[offset : offset + count]
            self.paths.append(visual_path)
            self.views.append(colors)
        self.update()

    def update(self, event=None):
        """Map current activity IDs to colors and publish them to initialized viewers."""
        activity = self.entry.solver.particle_activity
        if activity is not None:
            wp.launch(
                color_particle_activity,
                dim=self.entry.particle_indices.shape[0],
                inputs=[activity, self.entry.particle_indices, self.colors],
                device=self.colors.device,
            )
        renderers = set()
        for visualizer in self.sim.visualizers:
            if not visualizer.is_initialized:
                continue
            if visualizer.cfg.backend.visualizer_type == "newton_gl":
                visualizer._viewer.particle_colors = self.colors
            elif visualizer.cfg.backend.visualizer_type == "newton_rtx":
                renderers.update(visualizer._renderers)
        if not self.paths:
            return
        stream = wp.get_stream(self.colors.device).cuda_stream or 1
        for renderer in renderers:
            if renderer._use_ovstage:
                import ovstage

                tensors = [
                    ovstage.make_dltensor(
                        colors, dtype=ovstage.DLDataType(code=ovstage.DLDataTypeCode.kDLFloat, bits=32, lanes=3)
                    )
                    for colors in self.views
                ]
                with renderer.scene.query(self.paths) as query:
                    renderer.scene.stage.write_attribute(
                        query,
                        "primvars:displayColor",
                        ordinal=renderer.scene.ordinal,
                        tensors=tensors,
                        is_array=True,
                        cuda_stream=stream,
                    ).wait()
            else:
                from ovrtx import DataAccess

                renderer.backend.renderer.write_array_attribute(
                    self.paths, "primvars:displayColor", self.views, data_access=DataAccess.ASYNC, cuda_stream=stream
                )
