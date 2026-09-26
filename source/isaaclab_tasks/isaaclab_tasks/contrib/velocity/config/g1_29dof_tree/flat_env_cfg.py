# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for G1 29-DoF locomotion on flat ground across fallen trees.

The task is the flat rigid-terrain task with piles of cable logs added to every environment and
the physics backend replaced by the coupled MJWarp/VBD preset. Actions, terminations and the
observation layout are inherited unchanged, so a policy trained on flat ground transfers directly
and the piles are the only new thing it has to deal with. The contact-sensor terms are the one
exception: coupled solvers report no contacts, so those rewards and observations are re-derived
from the foot kinematics in :mod:`.mdp`.
"""

from __future__ import annotations

from isaaclab_visualizers.newton import NewtonGLVisualizerCfg

from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass

from ..g1_29dof_rigid.flat_env_cfg import G1FlatEnvCfg
from .env_cfg import G1TreeEventCfg, G1TreeObservationsCfg, G1TreePhysicsCfg, G1TreeRewardsCfg, G1TreeSceneCfg


@configclass
class G1TreeFlatEnvCfg(G1FlatEnvCfg):
    """Velocity-tracking locomotion of the G1 across a pile of fallen trees."""

    sim: SimulationCfg = SimulationCfg(physics=G1TreePhysicsCfg())  # type: ignore
    # The piles ring the robot at 1.5 m and the logs get kicked around, so the environments are
    # spaced to hold the cable-entry ground of `scene_cfg.PILE_GROUND_SIZE` without overlapping
    # their neighbours; cable piles are expensive enough that the environment count is cut
    # accordingly, and raising `scene_cfg.PILE_COUNT` should be paired with cutting it further.
    scene: G1TreeSceneCfg = G1TreeSceneCfg(num_envs=512, env_spacing=5.0)
    observations: G1TreeObservationsCfg = G1TreeObservationsCfg()
    rewards: G1TreeRewardsCfg = G1TreeRewardsCfg()
    events: G1TreeEventCfg = G1TreeEventCfg()

    def __post_init__(self) -> None:
        super().__post_init__()

        # The flat task drives the robot in a straight line for as long as the episode lasts,
        # which walks it past the pile within seconds. Resample often enough that it keeps
        # coming back to the logs.
        self.commands.base_velocity.resampling_time_range = (4.0, 4.0)


@configclass
class G1TreeFlatEnvCfg_PLAY(G1TreeFlatEnvCfg):
    """Small playback scene with randomization disabled."""

    def __post_init__(self) -> None:
        super().__post_init__()

        self.episode_length_s = 5.0

        # make a smaller scene for play
        self.scene.num_envs = 16

        # disable curriculum and observation corruption
        self.curriculum.terrain_levels = None  # type: ignore
        self.curriculum.command_vel = None  # type: ignore
        self.observations.policy.enable_corruption = False

        # remove random events
        self.events.add_base_mass = None  # type: ignore
        self.events.push_robot = None  # type: ignore
        self.events.physics_material = None  # type: ignore
        self.events.scale_actuator_gains = None  # type: ignore

        # Randomization
        self.events.reset_base.params = {
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "yaw": (-3.14, 3.14),
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

        # walk straight into the pile
        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 0.5)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.resampling_time_range = (self.episode_length_s, self.episode_length_s)
        self.commands.base_velocity.debug_vis = True

        self.sim.visualizer_cfgs = [
            NewtonGLVisualizerCfg(eye=(0.0, -6.0, 2.0)),
        ]
        self.video_recorders = []
