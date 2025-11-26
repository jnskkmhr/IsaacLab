# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.envs.common import ViewerCfg

from .rough_env_cfg import T1RoughEnvCfg

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as vel_mdp
# import isaaclab_tasks.manager_based.locomotion.velocity.config.t1_soft.mdp as t1_mdp


@configclass
class T1FlatEnvCfg(T1RoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # physics dt
        self.sim.dt = 0.005 # 200Hz
        self.decimation = 4 # 50Hz
        self.sim.render_interval = self.decimation

        # make curriculum soft terrain
        self.scene.terrain = vel_mdp.CurriculumSoftTerrain

        # no end effector mass randomization
        self.events.add_end_effector_mass = None

        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None

        # Commands
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)


@configclass
class T1FlatEnvCfg_PLAY(T1FlatEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # physics dt
        self.sim.dt = 0.005 # 200Hz
        self.decimation = 4 # 50Hz
        self.sim.render_interval = self.decimation
        self.episode_length_s = 10.0

        # make soft terrain 
        self.scene.terrain = vel_mdp.SoftTerrain
        # self.scene.terrain.disable_collider = True  # soft terrain
        self.actions.physics_callback.disable = True # disable soft contact

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5

        # disable randomization for play
        self.observations.policy.enable_corruption = False

        # disable curriculum 
        self.curriculum.terrain_levels = None
        self.curriculum.track_ang_vel = None
        self.curriculum.track_lin_vel = None

        # remove random events
        self.events.add_end_effector_mass = None
        self.events.add_base_mass = None
        self.events.scale_actuator_gains = None
        self.events.base_com = None
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.events.randomize_friction = None
        self.events.randomize_stiffness = None

        # Commands
        # self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
        # self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        # self.commands.base_velocity.heading_command = False
        # self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        
        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.3, -0.3)
        self.commands.base_velocity.heading_command = False

        # track specific yaw angle
        # self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        # self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        # self.commands.base_velocity.heading_command = True
        # self.commands.base_velocity.resampling_time_range = (self.episode_length_s/10, self.episode_length_s/10)
        self.commands.base_velocity.debug_vis = False

        # pose initialization
        self.events.reset_base.params = {
            "pose_range": 
                {"x": (-0.5, 0.5), 
                 "y": (-0.5, 0.5),
                # "yaw": (-math.pi, math.pi),
                # "yaw": (0, 0),
                # "yaw": (-math.pi/2, -math.pi/2),
                "yaw": (-math.pi/4, -math.pi/4),
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
        
        # rendering settings
        self.viewer = ViewerCfg(
            eye=(-0.0, -3.5, 0.25), 
            lookat=(0.0, -1.5, 0.15),
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