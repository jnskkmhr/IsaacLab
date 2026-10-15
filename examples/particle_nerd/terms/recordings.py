# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Named recorder terms with step-aligned states and validation impulses."""

import torch
import warp as wp
from newton_access import NewtonData
from scene_cfg import BODY_FRICTION, BODY_SIZES

from isaaclab.managers import RecorderManagerBaseCfg, RecorderTerm, RecorderTermCfg
from isaaclab.utils import configclass


class StateRecorder(RecorderTerm):
    """Record the reset state and every post-step state, giving T+1 states."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.access = NewtonData(env)

    def record_post_reset(self, env_ids):
        return "states", {key: value[env_ids] for key, value in self.access.state().items()}

    def record_post_step(self):
        return "states", self.access.state()


class InputRecorder(RecorderTerm):
    def record_pre_step(self):
        return "actions", self._env.action_manager.action


class ReferenceRecorder(RecorderTerm):
    """MPM impulses are evaluation targets, never reconstruction inputs."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.access = NewtonData(env)

    def record_pre_step(self):
        self.access.cache_rigid_contact_poses()
        return "reference/applied_body_impulse", self.access.feedback_impulse()

    def record_post_step(self):
        return "reference", {
            "body_impulse": self.access.body_impulse(),
            "rigid_contact_impulse": self.access.rigid_contact_impulse(),
            "next_body_impulse": self.access.feedback_impulse(),
        }

    def record_post_reset(self, env_ids):
        model = self.access.model
        indices = self.access.particle_indices
        count = len(BODY_SIZES)
        n = self._env.num_envs
        device = self._env.device
        values = {
            "particle_mass": wp.to_torch(model.particle_mass)[indices],
            "particle_radius": wp.to_torch(model.particle_radius)[indices],
            "particle_id": torch.arange(indices.shape[1], device=device).expand(n, -1),
            "body_mass": wp.to_torch(model.body_mass)[self.access.body_indices],
            "body_inertia": wp.to_torch(model.body_inertia)[self.access.body_indices],
            "body_com": wp.to_torch(model.body_com)[self.access.body_indices],
            "body_object_id": torch.arange(count, device=device).expand(n, -1),
            "body_size": torch.tensor(BODY_SIZES, device=device).expand(n, -1, -1),
            "body_friction": torch.full((n, count), BODY_FRICTION, device=device),
            "environment_origin": self._env.scene.env_origins,
            "environment_id": torch.arange(n, device=device),
            "dt": torch.full((n,), self._env.physics_dt, device=device),
        }
        return "constants", {key: value[env_ids] for key, value in values.items()}


@configclass
class ParticleRecorderCfg(RecorderManagerBaseCfg):
    states = RecorderTermCfg(class_type=StateRecorder)
    actions = RecorderTermCfg(class_type=InputRecorder)
    reference = RecorderTermCfg(class_type=ReferenceRecorder)
