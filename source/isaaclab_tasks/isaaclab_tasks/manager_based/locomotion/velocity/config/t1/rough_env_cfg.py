# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import math 

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab.envs.mdp as mdp
import isaaclab_tasks.manager_based.locomotion.velocity.config.t1.mdp as t1_mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg, RewardsCfg

from .env_cfg import (
    T1ActionsCfg, 
    T1ObservationsCfg, 
    T1RewardsCfg, 
    T1SceneCfg,
    T1TerminationsCfg,
    T1CurriculumsCfg, 
    T1EventsCfg,
    T1CommandsCfg, 
)


@configclass
class T1RoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    scene: T1SceneCfg = T1SceneCfg(num_envs=4096, env_spacing=2.5)
    observations: T1ObservationsCfg = T1ObservationsCfg()
    rewards: T1RewardsCfg = T1RewardsCfg()
    terminations: T1TerminationsCfg = T1TerminationsCfg()
    events: T1EventsCfg = T1EventsCfg()
    actions: T1ActionsCfg = T1ActionsCfg()
    curriculum: T1CurriculumsCfg = T1CurriculumsCfg()
    commands: T1CommandsCfg = T1CommandsCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # gait duration in sec
        self.phase_dt = 0.2 * 2
        self.scene.env_spacing = 2.5

        # physics dt
        self.sim.dt = 0.002 # 500 Hz
        self.decimation = 10 # 50 Hz
        self.sim.render_interval = self.decimation

        # sim settings
        self.sim.gravity = (0.0, 0.0, -9.806)
        self.sim.physx.gpu_found_lost_agregate_pairs_capacity = 2**26
        self.sim.physx.gpu_total_aggregatge_pairs_capacity = 2**22

        # Scene
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/Trunk"

        # Randomization
        self.events.push_robot = None
        # self.events.push_robot.params["velocity_range"] = {"x": (-1.0, 1.0), "y": (-1.0, 1.0)}
        self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 3.0)
        self.events.add_end_effector_mass.params["mass_distribution_params"] = (0.0, 1.0)
        self.events.scale_actuator_gains.params["stiffness_distribution_params"] = (0.95, 1.05)
        self.events.scale_actuator_gains.params["damping_distribution_params"] = (0.95, 1.05)
        # self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.base_external_force_torque.params["asset_cfg"].body_names = ["Trunk"]
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-math.pi, math.pi)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }
        self.events.base_com = None

        # Rewards
        self.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot",
            joint_names=[
                ".*Hip.*",
                ".*Knee.*",
            ],
        )
        self.rewards.dof_acc_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot",
            joint_names=[
                ".*Hip.*",
                ".*Knee.*",
            ],
        )
#------------------------------------------------------------#
        self.rewards.alive.weight = 10
        self.rewards.termination_penalty.weight = 0.0
        self.rewards.track_lin_vel_xy_exp.weight = 10
        self.rewards.track_ang_vel_z_exp.weight = 5
        self.rewards.base_height_l2.weight = -100
        self.rewards.flat_orientation_exp.weight = 4.0
#------------------------------------------------------------#
        self.rewards.dof_torques_l2.weight = -2e-4
        self.rewards.lin_vel_z_l2.weight = -2.0
        self.rewards.ang_vel_xy_l2.weight = -1
        self.rewards.dof_vel_l2.weight = -1e-4
        self.rewards.dof_acc_l2.weight = -1.25e-7
        self.rewards.base_acc_l2.weight = -1e-4
        self.rewards.action_rate_l2.weight = -0.5
        self.rewards.dof_pos_limits.weight = -3
        self.rewards.dof_torque_limits.weight = -1.0
#------------------------------------------------------------#
        self.rewards.feet_swing.weight = 20
        self.rewards.feet_roll.weight = -5.0
        self.rewards.feet_pitch.weight = -5.0
        self.rewards.feet_yaw_diff.weight = -1.0
        self.rewards.feet_yaw_mean.weight = -1.0
        self.rewards.foot_distance.weight = -1.0
        self.rewards.feet_slide.weight = -1
        self.rewards.feet_air_time.weight = 1.5
        self.rewards.foot_clearance.weight = 1.0
        self.rewards.joint_deviation_hip.weight = -1.0
#------------------------------------------------------------#
        # Commands
        self.commands.base_velocity.ranges.lin_vel_x = (-0.5, 0.5)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.5, 0.5)

        # Curriculum
        # self.curriculum.command_range = None
        self.curriculum.track_lin_vel.params['std'] = 0.25
        self.curriculum.track_ang_vel.params['std'] = 0.25
        self.curriculum.track_lin_vel.params["num_steps"] = 7000 * 24
        self.curriculum.track_ang_vel.params["num_steps"] = 7000 * 24



@configclass
class T1RoughEnvCfg_PLAY(T1RoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.base_external_force_torque = None
        self.events.push_robot = None
