# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

from isaaclab.envs.common import ViewerCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as vel_mdp

from .rough_env_cfg import G1RoughEnvCfg


@configclass
class G1FlatEnvCfg(G1RoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Optionally override physics speed
        self.sim.dt = 0.005  # 200Hz
        self.decimation = 4  # 50Hz
        self.sim.render_interval = self.decimation

        # make curriculum soft terrain
        self.scene.terrain = vel_mdp.CurriculumSoftTerrain

        # no height scan
        self.scene.height_scanner = None  # type: ignore
        self.observations.policy.height_scan = None  # type: ignore
        self.observations.critic.height_scan = None  # type: ignore

        # edit randomization
        self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 3.0)
        self.events.reset_base.params = {
            "pose_range": {
                "x": (-0.5, 0.5),
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
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)

        # disable curriculum for walking only
        # self.curriculum.command_vel = None

        # edit command range
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.5)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (-math.pi, math.pi)

        # disable for non rough terrain
        self.terminations.terrain_out_of_bounds = None # type: ignore

        # rendering
        self.sim.render.enable_dlssg = True
        self.sim.render.dlss_mode = "performance"  # type: ignore
        self.viewer = ViewerCfg(
            eye=(20.0, 20.0, 0.5),
            lookat=(0.0, 0.0, 0.2),
            resolution=(1080, 720),
            origin_type="asset_root",
            asset_name="robot",
        )


class G1FlatEnvCfg_FINETUNE(G1FlatEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        self.curriculum.command_vel.params["velocity_stages"] = [
            {"step": 0, "lin_vel_x": (-1.0, 1.0), "ang_vel_z": (-0.5, 0.5)},
            {"step": 4000 * 24, "lin_vel_x": (-1.0, 1.7), "ang_vel_z": (-0.7, 0.7)},
            {"step": 8000 * 24, "lin_vel_x": (-1.0, 2.5), "ang_vel_z": (-1.0, 1.0)},
        ]
        self.curriculum.track_lin_vel.params["step_1"] = 12000 * 24
        self.curriculum.track_ang_vel.params["step_1"] = 12000 * 24
        self.curriculum.track_heading.params["step_1"] = 12000 * 24

        # self.scene.terrain = vel_mdp.SoftTerrainVisual


class G1FlatEnvCfg_PLAY(G1FlatEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # change timestep
        self.sim.dt = 1 / 200  # 200Hz
        self.decimation = 4  # 50Hz
        self.episode_length_s = 10.0

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 0.0

        # # terrain with hole
        # self.scene.terrain = vel_mdp.SoftTerrainVisual
        # self.scene.rigid_floor = vel_mdp.RigidSoftTerrain

        self.scene.terrain = vel_mdp.SoftTerrain
        # self.scene.rigid_floor = vel_mdp.RigidPatch

        # disable curriculum
        self.curriculum.terrain_levels = None # type: ignore
        self.curriculum.command_vel = None # type: ignore

        # disable randomization for play
        self.observations.policy.enable_corruption = False

        # remove random events
        self.events.add_base_mass = None # type: ignore
        self.events.push_robot = None # type: ignore
        self.events.physics_material = None # type: ignore
        self.events.scale_actuator_gains = None # type: ignore
        # self.events.distance_based_sample_terrain_property.mode = "interval"
        # self.events.distance_based_sample_terrain_property.interval_range_s = (0.02, 0.02)

        # Commands
        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.0, 0.0)

        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.resampling_time_range = (self.episode_length_s, self.episode_length_s)
        self.commands.base_velocity.debug_vis = True

        # Randomization
        self.events.reset_base.params = {
            "pose_range": {
                "x": (-0.0, 0.0),
                "y": (-0.0, 0.0),
                # "yaw": (-math.pi, math.pi),
                # "yaw": (-math.pi / 2, -math.pi / 2),
                # "yaw": (-math.pi / 6, math.pi / 6),
                "yaw": (0, 0),
                # "yaw": (math.pi / 2, math.pi / 2),
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
        # self.sim.render.enable_dlssg = True
        # self.sim.render.dlss_mode = "performance"  # type: ignore
        self.viewer = ViewerCfg(
            eye=(-0.0, -2.5, -0.1),
            lookat=(0.0, -0.0, -0.1),
            # eye=(3.5, 0.0, 0.5),
            # lookat=(0.0, 0.0, 0.2),
            resolution=(1920, 1080),
            # resolution=(1080, 720),
            origin_type="asset_root",
            asset_name="robot",
        )


"""
Recurrent PPO env
"""


class G1FlatRecurrentPPOEnvCfg(G1FlatEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # no explicit history concatenation
        self.observations.policy.history_length = 1
        self.observations.critic.history_length = 1


class G1FlatRecurrentPPOEnvCfg_PLAY(G1FlatEnvCfg_PLAY):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # no explicit history concatenation
        self.observations.policy.history_length = 1
        self.observations.critic.history_length = 1
