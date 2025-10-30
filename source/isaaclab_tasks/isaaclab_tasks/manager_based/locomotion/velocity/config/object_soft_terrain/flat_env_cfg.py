# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import math
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.envs.common import ViewerCfg

import isaaclab_tasks.manager_based.locomotion.velocity.config.object_soft_terrain.mdp as object_mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

from .env_cfg import (
    ObjectActionsCfg, 
    ObjectObservationsCfg, 
    ObjectRewardsCfg, 
    ObjectSceneCfg,
    ObjectTerminationsCfg,
)


@configclass
class ObjectFlatEnvCfg(LocomotionVelocityRoughEnvCfg):
    rewards: ObjectRewardsCfg = ObjectRewardsCfg()
    actions: ObjectActionsCfg = ObjectActionsCfg()
    observations: ObjectObservationsCfg = ObjectObservationsCfg()
    scene: ObjectSceneCfg = ObjectSceneCfg(num_envs=4096, env_spacing=2.5)
    terminations: ObjectTerminationsCfg = ObjectTerminationsCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 5.0
        self.episode_length_s = 5.0

        # Randomization
        self.events.push_robot = None
        self.events.physics_material = None
        self.events.add_base_mass = None
        self.events.reset_robot_joints = None
        self.events.base_external_force_torque = None
        self.events.base_com = None
        self.events.reset_base.params = {
            "asset_cfg": SceneEntityCfg("object"),
            "pose_range": 
                {
                "x": (-0., 0.), 
                "y": (-0., 0.), 
                "z": (0.05, 0.05), 
                # "yaw": (-math.pi, math.pi),
                "yaw": (0.0, 0.0),
                 },
            "velocity_range": {
                "x": (0.5, 0.5),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0), 
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }
        # disable randomization for play
        self.observations.policy.enable_corruption = False

        # change terrain to flat
        self.scene.terrain = object_mdp.FlatTerrain
        self.scene.terrain.disable_collider = True
        self.actions.physics_callback.max_terrain_level = 1 # fully soft
        
        # no terrain curriculum
        self.curriculum.terrain_levels = None
        
        # Commands
        self.commands.base_velocity.asset_name = "object"
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.0, 0.0)
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.resampling_time_range = (self.episode_length_s, self.episode_length_s)
        self.commands.base_velocity.debug_vis = False
        
        # light and view settings
        self.scene.sky_light.init_state.rot = (0, 0, 0, 1.0)  # roll=60deg
        
        # viewer 
        self.viewer = ViewerCfg(
            eye=(-0.0, -0.6, 0.5), 
            lookat=(0.0, -0.3, 0.3),
            resolution=(1920, 1080), 
            # origin_type="asset_root", 
            # asset_name="object"
        )