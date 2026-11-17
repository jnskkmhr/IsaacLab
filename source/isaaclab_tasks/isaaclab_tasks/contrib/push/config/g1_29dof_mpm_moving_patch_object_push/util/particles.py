# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Recycling particle storage following the robot-object midpoint on the GPU."""

import newton
import numpy as np
import warp as wp

from isaaclab_tasks.contrib.velocity_mpm.g1_29dof_mpm_moving_patch.util.kernel import update_patch_center
from isaaclab_tasks.contrib.velocity_mpm.g1_29dof_mpm_moving_patch.util.particles import MovingPatchParticles


@wp.kernel
def midpoint_transforms(
    body_q: wp.array[wp.transform],
    robot_bodies: wp.array[int],
    object_bodies: wp.array[int],
    centers: wp.array[wp.transform],
):
    """Compute one virtual tracking pose per world without modifying rigid-body state."""
    i = wp.tid()
    robot = wp.transform_get_translation(body_q[robot_bodies[i]])
    obj = wp.transform_get_translation(body_q[object_bodies[i]])
    centers[i] = wp.transform(0.5 * (robot + obj), wp.quat_identity())


class ObjectPushMovingPatchParticles(MovingPatchParticles):
    """Reuse ordinary recycling with one robot-object-centered patch per world."""

    def _setup_tracked_bodies(self) -> None:
        super()._setup_tracked_bodies()
        worlds = self.model.body_world.numpy()
        bodies = []
        for world in range(self.model.world_count):
            matches = [
                i
                for i, label in enumerate(self.model.body_label)
                if worlds[i] == world and label and label.rsplit("/", 1)[-1] == self.terrain.tracked_object_body
            ]
            if len(matches) != 1:
                raise ValueError(f"Expected one {self.terrain.tracked_object_body!r} in world {world}, got {matches}")
            bodies.append(matches[0])
        self.object_bodies = wp.array(bodies, dtype=int, device=self.model.device)
        self._midpoint_q = wp.empty(self.model.world_count, dtype=wp.transform, device=self.model.device)
        self._midpoint_ids = wp.array(np.arange(self.model.world_count), dtype=int, device=self.model.device)

    def _update_patch_centers(self, state: newton.State) -> None:
        wp.launch(
            midpoint_transforms,
            dim=self.model.world_count,
            inputs=[state.body_q, self.bodies, self.object_bodies],
            outputs=[self._midpoint_q],
            device=self.model.device,
        )
        bounds = self.background_mesh.bounds
        wp.launch(
            update_patch_center,
            dim=self.model.world_count,
            inputs=[
                self._midpoint_q,
                self._midpoint_ids,
                self.origins,
                wp.vec2(*((bounds[0, :2] + bounds[1, :2]) / 2)),
                wp.vec2(*(bounds[1, :2] - bounds[0, :2])),
                wp.vec2(*self.terrain.total_patch_size),
                self.terrain.patch_discretization_step,
            ],
            outputs=[self.patch_centers],
            device=self.model.device,
        )
