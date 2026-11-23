# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Kinematic object action that does not disturb the MPM sand between steps."""

from ..core.mdp.actions import ObjectStateAction


class QuietObjectStateAction(ObjectStateAction):
    """Set the kinematic pose and velocity with ``skip_forward=True``.

    The default writes invalidate the asset's cached data after every step. With the MPM coupler that
    perturbs the sand: the whole bed keeps sinking (about 4 cm in 6 s) even when the object is parked far away.
    Skipping the invalidation removes the effect. Nothing here reads the asset's cached data back.
    """

    def apply_actions(self):
        pose = self._actions[:, :7].clone()
        pose[:, :3] += self._env.scene.env_origins
        self._asset.write_root_pose_to_sim_index(root_pose=pose, skip_forward=True)
        self._asset.write_root_velocity_to_sim_index(root_velocity=self._actions[:, 7:].contiguous(), skip_forward=True)
