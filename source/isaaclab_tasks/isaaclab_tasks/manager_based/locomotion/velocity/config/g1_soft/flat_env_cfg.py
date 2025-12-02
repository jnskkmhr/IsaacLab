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


@configclass
class G1FlatEnvCfg(G1RoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # physics dt
        self.sim.dt = 0.005 # 200Hz
        self.decimation = 4 # 50Hz
        self.sim.render_interval = self.decimation

        # make curriculum soft terrain
        self.scene.terrain = vel_mdp.CurriculumSoftTerrain
        
        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.observations.critic.height_scan = None
        
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
        self.events.base_com = None
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)

        # Commands (no vel curriculum)
        self.curriculum.command_vel = None
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (-math.pi, math.pi)


class G1FlatEnvCfg_PLAY(G1FlatEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # change timestep
        self.sim.dt = 1/200 # 200Hz
        self.decimation = 4 # 50Hz
        self.sim.render_interval = self.decimation
        self.episode_length_s = 20.0

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        
        # make soft terrain 
        self.scene.terrain = vel_mdp.SoftTerrain
        self.scene.terrain.disable_collider = True  # soft terrain
        # self.actions.physics_callback.disable = True # disable soft contact

        # disable curriculum 
        self.curriculum.terrain_levels = None
        self.curriculum.command_vel = None
        
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        
        # remove random pushing
        self.events.add_base_mass = None
        self.events.base_com = None
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.events.physics_material = None
        self.events.randomize_friction = None
        self.events.randomize_stiffness = None
        
        # Commands
        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 0.5)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        # self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.0, 0.0)
        
        # self.commands.base_velocity.ranges.lin_vel_x = (0.6, 0.6)
        # self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        # self.commands.base_velocity.ranges.ang_vel_z = (-0.5, -0.5)
        
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.resampling_time_range = (self.episode_length_s, self.episode_length_s)
        # self.commands.base_velocity.debug_vis = False
        
        # Randomization 
        self.events.reset_base.params = {
            "pose_range": {
                "x": (-0.5, 0.5), 
                "y": (-0.5, 0.5),
                "yaw": (-math.pi, math.pi),
                # "yaw": (-math.pi/2, -math.pi/2),
                # "yaw": (0, 0),
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
        self.sim.render.enable_dlssg = True
        self.sim.render.dlss_mode = "performance"
        self.viewer = ViewerCfg(
            eye=(-0.0, -3.5, 0.0), 
            lookat=(0.0, -0.0, 0.0),
            resolution=(1920, 1080), 
            # resolution=(1080, 720),
            origin_type="asset_root", 
            asset_name="robot"
        )
        
        # # rendering 
        # self.viewer = ViewerCfg(
        #     eye=(-0.0, -15.0, 1.0), 
        #     lookat=(0.0, -0.0, 1.0),
        #     resolution=(1920, 1080), 
        #     # origin_type="asset_root", 
        #     # asset_name="robot"
        # )