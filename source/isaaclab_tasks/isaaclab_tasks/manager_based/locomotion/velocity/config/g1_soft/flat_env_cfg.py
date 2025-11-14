# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import math
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.envs.common import ViewerCfg

from .rough_env_cfg import G1RoughEnvCfg
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as vel_mdp
# import isaaclab_tasks.manager_based.locomotion.velocity.config.g1_soft.mdp as g1_mdp


@configclass
class G1FlatEnvCfg(G1RoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make curriculum soft terrain
        self.scene.terrain = vel_mdp.CurriculumSoftTerrain
        
        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        
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

        # Rewards
        self.rewards.track_ang_vel_z_exp.weight = 1.0
        self.rewards.lin_vel_z_l2.weight = -0.2
        self.rewards.action_rate_l2.weight = -0.005
        self.rewards.dof_acc_l2.weight = -1.0e-7
        self.rewards.dof_torques_l2.weight = -2.0e-6
        self.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*_hip_.*", ".*_knee_joint"]
        )
        # Commands
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        
        # light and view settings
        self.scene.sky_light.init_state.rot = (0.86603, 0, 0, 0.5)  # yaw=60deg


class G1FlatEnvCfg_PLAY(G1FlatEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # change timestep
        self.sim.dt = 1/200 # 200Hz
        self.decimation = 4 # 50Hz
        self.sim.render_interval = self.decimation
        self.episode_length_s = 20.0
        
        # make soft terrain 
        self.scene.terrain = vel_mdp.SoftTerrain
        self.scene.terrain.disable_collider = True  # soft terrain
        # self.actions.physics_callback.disable = True # disable soft contact

        # disable curriculum 
        self.curriculum.terrain_levels = None
        
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        
        # remove random pushing
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.events.physics_material = None
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.randomize_friction = None
        self.events.randomize_stiffness = None
        
        # Commands
        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        # self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.0, 0.0)
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.resampling_time_range = (self.episode_length_s, self.episode_length_s)
        # self.commands.base_velocity.debug_vis = False
        
        # Randomization 
        self.events.reset_base.params = {
            "pose_range": {
                "x": (-0.5, 0.5), 
                "y": (-0.5, 0.5),
                # "yaw": (-math.pi, math.pi),
                "yaw": (0, 0),
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

        # rendering 
        self.viewer = ViewerCfg(
            eye=(-0.0, -2.5, 0.0), 
            lookat=(0.0, -0.8, 0.0),
            resolution=(1920, 1080), 
            origin_type="asset_root", 
            asset_name="robot"
        )
        
        # # rendering 
        # self.viewer = ViewerCfg(
        #     eye=(-0.0, -2.5, 0.5), 
        #     lookat=(0.0, -0.3, 0.5),
        #     resolution=(1920, 1080), 
        #     # origin_type="asset_root", 
        #     # asset_name="robot"
        # )