# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

##
# Pre-defined configs
##
from .env_cfg import (
    G1ActionsCfg,
    G1CommandsCfg,
    G1CurriculumCfg,
    G1EventCfg,
    G1RewardsCfg,
    G1SceneCfg,
    G1StudentObservationsCfg,
    G1TeacherObservationsCfg,
    G1TerminationsCfg,
)


@configclass
class G1RoughTeacherEnvCfg(LocomotionVelocityRoughEnvCfg):
    rewards: G1RewardsCfg = G1RewardsCfg()
    actions: G1ActionsCfg = G1ActionsCfg()
    observations: G1TeacherObservationsCfg = G1TeacherObservationsCfg()
    scene: G1SceneCfg = G1SceneCfg(num_envs=4096, env_spacing=2.5)
    terminations: G1TerminationsCfg = G1TerminationsCfg()
    curriculum: G1CurriculumCfg = G1CurriculumCfg()
    events: G1EventCfg = G1EventCfg()
    commands: G1CommandsCfg = G1CommandsCfg()
    seed: int = 42

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Optionally override physics speed
        # self.sim.dt = 0.002 # 500 Hz
        # self.decimation = 10 # 50 Hz
        # self.sim.render_interval = self.decimation

        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        # self.sim.physx.enable_external_forces_every_iteration = True

        # Randomization
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }


@configclass
class G1RoughTeacherEnvCfg_PLAY(G1RoughTeacherEnvCfg):
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
        # self.events.base_external_force_torque = None
        self.events.push_robot = None  # type: ignore


@configclass
class G1RoughStudentEnvCfg(LocomotionVelocityRoughEnvCfg):
    rewards: G1RewardsCfg = G1RewardsCfg()
    actions: G1ActionsCfg = G1ActionsCfg()
    observations: G1StudentObservationsCfg = G1StudentObservationsCfg()
    scene: G1SceneCfg = G1SceneCfg(num_envs=4096, env_spacing=2.5)
    terminations: G1TerminationsCfg = G1TerminationsCfg()
    curriculum: G1CurriculumCfg = G1CurriculumCfg()
    events: G1EventCfg = G1EventCfg()
    commands: G1CommandsCfg = G1CommandsCfg()
    seed: int = 42

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Optionally override physics speed
        # self.sim.dt = 0.002 # 500 Hz
        # self.decimation = 10 # 50 Hz
        # self.sim.render_interval = self.decimation

        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        # self.sim.physx.enable_external_forces_every_iteration = True

        # Randomization
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }


@configclass
class G1RoughStudentEnvCfg_PLAY(G1RoughStudentEnvCfg):
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
        # self.events.base_external_force_torque = None
        self.events.push_robot = None  # type: ignore
