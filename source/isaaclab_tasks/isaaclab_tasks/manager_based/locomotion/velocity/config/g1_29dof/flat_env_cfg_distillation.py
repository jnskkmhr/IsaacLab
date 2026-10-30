# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import math

from isaaclab.envs.common import ViewerCfg
from isaaclab.utils import configclass

from .rough_env_cfg_distillation import G1RoughStudentEnvCfg, G1RoughTeacherEnvCfg

"""
teacher
"""


@configclass
class G1FlatTeacherEnvCfg(G1RoughTeacherEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Optionally override physics speed
        self.sim.dt = 0.005  # 200Hz
        self.decimation = 4  # 50Hz
        self.sim.render_interval = self.decimation

        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None

        # curriculum settings
        self.curriculum.terrain_levels = None  # type: ignore
        # self.curriculum.command_vel = None  # no running

        # no height scan
        self.scene.height_scanner = None
        if hasattr(self.observations.policy, "height_scan"):
            self.observations.policy.height_scan = None
        if hasattr(self.observations.critic, "height_scan"):
            self.observations.critic.height_scan = None

        # Randomization
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

        # Commands
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.5)
        self.commands.base_velocity.ranges.lin_vel_y = (-1.0, 1.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (-math.pi, math.pi)

        # rendering
        self.sim.render.enable_dlssg = True
        self.sim.render.dlss_mode = "performance"  # type: ignore
        self.viewer = ViewerCfg(
            eye=(-0.0, -3.5, 0.5),
            lookat=(0.0, -0.0, 0.2),
            # eye=(3.5, 0.0, 0.5),
            # lookat=(0.0, 0.0, 0.2),
            # resolution=(1920, 1080),
            resolution=(1080, 720),
            origin_type="asset_root",
            asset_name="robot",
        )


class G1FlatTeacherEnvCfg_PLAY(G1FlatTeacherEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # change timestep
        # self.sim.dt = 1/200 # 200Hz
        # self.decimation = 4 # 50Hz
        # self.sim.render_interval = self.decimation
        self.episode_length_s = 20.0

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5

        # disable curriculum
        self.curriculum.terrain_levels = None  # type: ignore
        self.curriculum.command_vel = None  # type: ignore

        # disable randomization for play
        self.observations.policy.enable_corruption = False

        # remove random pushing
        self.events.add_base_mass = None  # type: ignore
        self.events.push_robot = None  # type: ignore
        self.events.physics_material = None  # type: ignore
        self.events.scale_actuator_gains = None  # type: ignore

        # Commands
        self.commands.base_velocity.ranges.lin_vel_x = (2.5, 2.5)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.0, 0.0)

        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.resampling_time_range = (self.episode_length_s / 4, self.episode_length_s / 4)
        # self.commands.base_velocity.debug_vis = False

        # Randomization
        self.events.reset_base.params = {
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                # "yaw": (-math.pi, math.pi),
                # "yaw": (-math.pi/2, -math.pi/2),
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
        self.sim.render.enable_dlssg = True
        self.sim.render.dlss_mode = "performance"
        self.viewer = ViewerCfg(
            eye=(-0.0, -3.5, 0.2),
            lookat=(0.0, -0.0, 0.0),
            # resolution=(1920, 1080),
            resolution=(1080, 720),
            origin_type="asset_root",
            asset_name="robot",
        )


"""
student
"""


@configclass
class G1FlatStudentEnvCfg(G1RoughStudentEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Optionally override physics speed
        self.sim.dt = 0.005  # 200Hz
        self.decimation = 4  # 50Hz
        self.sim.render_interval = self.decimation

        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None

        # curriculum settings
        self.curriculum.terrain_levels = None  # type: ignore
        # self.curriculum.command_vel = None  # no running
        self.curriculum.command_vel.params["velocity_stages"] = [
            {"step": 0, "lin_vel_x": (-1.0, 1.0), "ang_vel_z": (-0.5, 0.5)},
            {"step": 3000 * 24, "lin_vel_x": (-1.0, 1.7), "ang_vel_z": (-0.7, 0.7)},
            {"step": 6000 * 24, "lin_vel_x": (-1.0, 2.5), "ang_vel_z": (-1.0, 1.0)},
        ]
        # does not have effect for pure DAgger
        # self.curriculum.track_lin_vel.params["num_steps"] = 12000 * 24
        # self.curriculum.track_ang_vel.params["num_steps"] = 12000 * 24
        # self.curriculum.track_heading.params["num_steps"] = 12000 * 24

        # no height scan
        self.scene.height_scanner = None
        if hasattr(self.observations.policy, "height_scan"):
            self.observations.policy.height_scan = None
        if hasattr(self.observations.student_encoder, "height_scan"):
            self.observations.student_encoder.height_scan = None

        # Randomization
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

        # Commands
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.5)
        self.commands.base_velocity.ranges.lin_vel_y = (-1.0, 1.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (-math.pi, math.pi)


class G1FlatStudentEnvCfg_PLAY(G1FlatStudentEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # change timestep
        # self.sim.dt = 1/200 # 200Hz
        # self.decimation = 4 # 50Hz
        # self.sim.render_interval = self.decimation
        self.episode_length_s = 10.0

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5

        # disable curriculum
        self.curriculum.terrain_levels = None  # type: ignore
        self.curriculum.command_vel = None  # type: ignore

        # disable randomization for play
        self.observations.policy.enable_corruption = False

        # remove random pushing
        self.events.add_base_mass = None  # type: ignore
        self.events.push_robot = None  # type: ignore
        self.events.physics_material = None  # type: ignore
        self.events.scale_actuator_gains = None  # type: ignore

        # Commands
        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)

        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_standing_envs = 0.2
        self.commands.base_velocity.resampling_time_range = (self.episode_length_s / 4, self.episode_length_s / 4)
        # self.commands.base_velocity.debug_vis = False

        # Randomization
        self.events.reset_base.params = {
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                # "yaw": (-math.pi, math.pi),
                # "yaw": (-math.pi/2, -math.pi/2),
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
        self.sim.render.enable_dlssg = True
        self.sim.render.dlss_mode = "performance"
        self.viewer = ViewerCfg(
            eye=(-0.0, -3.5, 0.2),
            lookat=(0.0, -0.0, 0.0),
            # resolution=(1920, 1080),
            resolution=(1080, 720),
            origin_type="asset_root",
            asset_name="robot",
        )
