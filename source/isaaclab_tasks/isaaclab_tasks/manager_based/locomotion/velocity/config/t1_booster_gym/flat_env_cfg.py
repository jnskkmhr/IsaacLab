# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.envs.common import ViewerCfg

from .rough_env_cfg import T1RoughEnvCfg


@configclass
class T1FlatEnvCfg(T1RoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # # change step dt
        # self.sim.dt = 0.002 # 500 Hz
        # self.decimation = 10 # 50 Hz
        # self.sim.render_interval = self.decimation

        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None

        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None

        # no terrain curriculum
        self.curriculum.terrain_levels = None

        # Commands
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-1.0, 1.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)


@configclass
class T1FlatEnvCfg_PLAY(T1FlatEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()
        # self.episode_length_s = 20
        self.episode_length_s = 1000000.0  # non-stop playing

        # disable randomization for play
        self.observations.policy.enable_corruption = False

        # remove random pushing
        self.events.base_external_force_torque = None
        self.events.push_robot = None

        # disable curriculum
        self.curriculum.command_curriculum_x = None
        self.curriculum.command_curriculum_y = None
        self.curriculum.command_curriculum_z = None
        self.curriculum.push_curriculum = None

        # disable events
        self.events.push_robot = None
        self.events.base_external_force_torque = None
        self.events.scale_actuator_gains = None
        self.events.physics_material = None
        self.events.add_base_com_xyz = None
        self.events.scale_base_mass = None


        # Commands
        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 0.5)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.0, -0.0)
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.rel_rotonly_envs = 0.0

        # pose initialization
        self.events.reset_base.params = {
            "pose_range": 
                {"x": (-0.5, 0.5), 
                 "y": (-0.5, 0.5),
                # "yaw": (-math.pi, math.pi),
                "yaw": (0, 0),
                # "yaw": (-math.pi/2, -math.pi/2),
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
        
        # track robot's motion
        self.viewer = ViewerCfg(
            eye=(-0.0, -3.5, 0.4), 
            lookat=(0.0, -0.8, 0.3),
            resolution=(1920, 1080), 
            origin_type="asset_root", 
            asset_name="robot"
        )

        # self.sim.render.enable_dlssg = True
        # self.sim.render.dlss_mode = "performance"
    