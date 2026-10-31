# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Identical scene presentation for learned and MPM reference play."""

import warp as wp


def create_viewer(kind, model):
    if kind == "newton_rtx":
        from newton.viewer import ViewerRTX

        viewer = ViewerRTX(width=1200, height=800, environment="studio", async_rendering=False)
    else:
        from newton.viewer import ViewerGL

        viewer = ViewerGL(width=1200, height=800)
    viewer.set_model(model)
    viewer.set_camera(wp.vec3(1.8, -2.4, 1.8))
    viewer.camera.look_at((0.0, 0.0, 0.18))
    return viewer


def particle_colors(count, device):
    return wp.full(count, wp.vec3(0.32, 0.24, 0.21), dtype=wp.vec3, device=device)


def draw_scene(viewer, state, positions, colors):
    viewer.log_state(state)
    if getattr(viewer, "_SandPointsInitialized", False):
        if type(viewer).__name__ == "ViewerRTX":
            viewer.log_points("sand", positions)
        else:
            viewer.log_points("sand", positions, radii=0.012)
    else:
        viewer.log_points("sand", positions, radii=0.012, colors=colors)
        viewer._SandPointsInitialized = True
