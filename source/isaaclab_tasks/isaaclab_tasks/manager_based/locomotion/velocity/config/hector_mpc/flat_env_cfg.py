# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import math
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.envs.common import ViewerCfg

from .rough_env_cfg import HECTORRoughEnvCfg
import isaaclab_tasks.manager_based.locomotion.velocity.config.hector_mpc.mdp as hector_mdp


@configclass
class HECTORFlatEnvCfg(HECTORRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None
        
        # Randomization 
        self.events.reset_base.params = {
            "pose_range": 
                {"x": (-0.5, 0.5), 
                 "y": (-0.5, 0.5), 
                "yaw": (-math.pi, math.pi),
                 },
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }
        self.events.base_external_force_torque = None
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)

        # # Rewards
        # self.rewards.track_ang_vel_z_exp.weight = 1.0
        # self.rewards.lin_vel_z_l2.weight = -0.2
        # self.rewards.action_rate_l2.weight = -0.005
        # self.rewards.dof_acc_l2.weight = -1.0e-7
        # self.rewards.feet_air_time.weight = 0.75
        # self.rewards.feet_air_time.params["threshold"] = 0.4
        # self.rewards.dof_torques_l2.weight = -2.0e-6
        # self.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg(
        #     "robot", joint_names=[".*_hip_.*", ".*_knee_joint"]
        # )
        
        # Commands
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        
        # light and view settings
        self.scene.sky_light.init_state.rot = (0.86603, 0, 0, 0.5)  # yaw=60deg
        self.viewer = ViewerCfg(
            eye=(-0.0, -2.0, -0.1), 
            lookat=(0.0, -0.0, 0.0),
            resolution=(1920, 1080), 
            origin_type="asset_root", 
            asset_name="robot"
        )


class HECTORFlatEnvCfg_PLAY(HECTORFlatEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.seed = 0
        self.scene.num_envs = 50
        self.scene.env_spacing = 5.0
        self.episode_length_s = 5.0
        
        # make soft terrain
        self.scene.terrain = hector_mdp.FlatTerrain
        self.scene.terrain.disable_collider = True  # soft terrain
        
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        
        # remove random pushing
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.events.physics_material = None
        self.events.reset_robot_joints.params["position_range"] = (0.0, 0.0)
        
        # Randomization 
        self.events.reset_base.params = {
            "pose_range": 
                {"x": (-2.5, 2.5), 
                 "y": (-2.5, 2.5), 
                 "yaw": (-math.pi, math.pi),
                 },
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }
        
        # Commands
        self.commands.base_velocity.ranges.lin_vel_x = (0.3, 0.3)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0,0)
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.resampling_time_range = (self.episode_length_s, self.episode_length_s)
        # self.commands.base_velocity.resampling_time_range = (self.episode_length_s/5, self.episode_length_s/5)
        # self.commands.base_velocity.debug_vis = False