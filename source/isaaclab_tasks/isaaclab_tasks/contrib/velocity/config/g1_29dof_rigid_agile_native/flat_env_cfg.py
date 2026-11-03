# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Agile G1 with leg and waist policy actions and delayed DC motors."""

from isaaclab.utils import configclass

from isaaclab_tasks.contrib.velocity.config.g1_29dof_rigid_agile.env_cfg.action_cfg import CONTROLLED_JOINT_NAMES
from isaaclab_tasks.contrib.velocity.config.g1_29dof_rigid_agile.env_cfg.scene_cfg import G1_29DOF_DELAYED_DC_MOTOR
from isaaclab_tasks.contrib.velocity.config.g1_29dof_rigid_agile.rough_env_cfg import G1RoughEnvCfg


@configclass
class G1FlatEnvCfg(G1RoughEnvCfg):
    """Native observation/reward contract with the corrected collection-step curricula."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 2048
        self.scene.robot = G1_29DOF_DELAYED_DC_MOTOR.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # Keep the successful delayed-DC sweep setting; robot shapes retain their asset margin.
        self.sim.physics.newton_mjwarp.default_shape_cfg.margin = 0.016
        self.sim.physics.default = self.sim.physics.newton_mjwarp

        joints = CONTROLLED_JOINT_NAMES.copy()
        self.actions.joint_pos.joint_names = joints
        self.actions.joint_pos.mirror_params = {"joint_names": joints}
        for term in (self.observations.policy.controlled_joint_pos, self.observations.policy.controlled_joint_vel):
            term.params["asset_cfg"].joint_names = joints
            term.mirror_params = {"joint_names": joints}
        self.observations.policy.actions.mirror_params = {"joint_names": joints}
        self.observations.critic.actions.mirror_params = {"joint_names": joints}
        for term in (
            self.rewards.torques,
            self.rewards.dof_vel,
            self.rewards.action_rate,
            self.rewards.action_rate_rate,
            self.rewards.dof_pos_limits,
            self.rewards.dof_vel_limits,
            self.rewards.torque_limits,
        ):
            term.params["asset_cfg"].joint_names = joints

        self.observations.height_scan = None
        self.scene.height_scanner = None
        self.rewards.foot_clearance = None


@configclass
class G1FlatEnvCfg_PLAY(G1FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.curriculum.command_vel = None
