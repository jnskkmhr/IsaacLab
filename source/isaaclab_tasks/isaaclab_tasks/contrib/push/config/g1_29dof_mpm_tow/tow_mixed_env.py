# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""G1 towing a sled over mixed rigid/sand terrain: the mixed walking environment plus the rope and the sled."""

from __future__ import annotations

from isaaclab_newton.physics.newton_builder_cfg import NewtonBuilderCfg

from isaaclab_tasks.contrib.velocity_mpm.config.g1_29dof.streaming_mpm_mixed_env import G1MPMMixedEnv

from .env_cfg.scene_cfg import SLED_NAME
from .tow_env import G1TowEnv
from .tow_mixed_env_cfg import G1TowMixedEnvCfg


class G1TowMixedEnv(G1TowEnv, G1MPMMixedEnv):
    """The tow environment on terrain whose columns alternate between rigid ground and streaming MPM sand."""

    cfg: G1TowMixedEnvCfg

    def _set_contact_margin(self, payload=None):
        """Use a fixed margin for each ground model on the feet and on the sled."""
        super()._set_contact_margin(payload)
        mpm = self.scene.terrain.mpm_env_mask.cpu().tolist()
        builder = self.sim.get_or_create_backend(NewtonBuilderCfg(physics_cfg=self.cfg.sim.physics))
        matched_envs = set()
        for shape_index, body_index in enumerate(builder.shape_body):
            if body_index < 0 or builder.body_label[body_index].rsplit("/", 1)[-1] != SLED_NAME:
                continue
            env_id = builder.body_world[body_index]
            if 0 <= env_id < self.num_envs:
                builder.shape_margin[shape_index] = (
                    self.cfg.mpm_contact_margin if mpm[env_id] else self.cfg.rigid_contact_margin
                )
                matched_envs.add(env_id)
        if len(matched_envs) != self.num_envs:
            raise ValueError("Could not find the sled shape in every environment when setting contact margins.")
