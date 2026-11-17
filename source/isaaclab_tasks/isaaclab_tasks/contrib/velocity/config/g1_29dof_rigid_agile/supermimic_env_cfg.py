# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


# Adapted from Agile. Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Agile environment with privileged physical observations for SuperMimic."""

import isaaclab.envs.mdp as mdp
from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg
from isaaclab.utils import configclass

import isaaclab_tasks.contrib.velocity.config.g1_29dof_rigid_agile.mdp.supermimic as supermimic_mdp
from isaaclab_tasks.core.velocity.mdp.observations import velocity_context

from .rough_env_cfg import G1RoughEnvCfg


@configclass
class G1RoughSuperMimicEnvCfg(G1RoughEnvCfg):
    """Agile physical task plus privileged SuperMimic training observations."""

    final_obs_every_step = True

    @configclass
    class StateCfg(ObservationGroupCfg):
        state = ObservationTermCfg(func=supermimic_mdp.AgileState)
        enable_corruption = False

    @configclass
    class LabelsCfg(ObservationGroupCfg):
        events = ObservationTermCfg(func=supermimic_mdp.AgileLabels)
        enable_corruption = False

    @configclass
    class ContextCfg(ObservationGroupCfg):
        masses = ObservationTermCfg(func=velocity_context)
        enable_corruption = False

    @configclass
    class GoalCfg(ObservationGroupCfg):
        command = ObservationTermCfg(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        enable_corruption = False

    @configclass
    class ActionCfg(ObservationGroupCfg):
        action = ObservationTermCfg(func=mdp.last_action)
        enable_corruption = False

    @configclass
    class PreviousActionCfg(ObservationGroupCfg):
        action = ObservationTermCfg(func=supermimic_mdp.previous_action)
        enable_corruption = False

    def __post_init__(self):
        super().__post_init__()
        # Retain the agreed 16 mm rigid-contact baseline for this experiment.
        self.scene.robot.newton_collision_shape_margin = 0.016
        self.sim.physics.newton_mjwarp.default_shape_cfg.margin = 0.016
        self.observations.state = self.StateCfg()
        self.observations.labels = self.LabelsCfg()
        self.observations.context = self.ContextCfg()
        self.observations.goal = self.GoalCfg()
        self.observations.prev_action = self.ActionCfg()
        self.observations.prev_prev_action = self.PreviousActionCfg()
