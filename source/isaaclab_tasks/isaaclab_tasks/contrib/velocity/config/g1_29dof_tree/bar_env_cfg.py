# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for G1 29-DoF locomotion on flat ground across low bars.

The task is the flat rigid-terrain task with rods strung between pillars added to every environment
and the physics backend replaced by the coupled MJWarp/VBD preset. A rod catches the swing foot
instead of stopping it, so the policy has to notice the snag, retract and reorient the foot, and
keep tracking the velocity command. Actions, terminations and the observation layout are inherited
unchanged; the contact-sensor terms are the one exception, since coupled solvers report no
contacts, so those rewards and observations are re-derived from the foot kinematics in :mod:`.mdp`.
"""

from __future__ import annotations

from isaaclab_visualizers.newton import NewtonGLVisualizerCfg

from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass

from ..g1_29dof_rigid.flat_env_cfg import G1FlatEnvCfg
from .env_cfg import G1BarEventCfg, G1BarPhysicsCfg, G1BarSceneCfg, G1TreeObservationsCfg, G1TreeRewardsCfg


@configclass
class G1BarFlatEnvCfg(G1FlatEnvCfg):
    """Velocity-tracking locomotion of the G1 across rods strung between pillars."""

    sim: SimulationCfg = SimulationCfg(physics=G1BarPhysicsCfg())  # type: ignore
    # The bars ring the robot at 1.5 m, so the environments are spaced to hold them with room to
    # walk on either side.
    scene: G1BarSceneCfg = G1BarSceneCfg(num_envs=1024, env_spacing=5.0)
    observations: G1TreeObservationsCfg = G1TreeObservationsCfg()
    rewards: G1TreeRewardsCfg = G1TreeRewardsCfg()
    events: G1BarEventCfg = G1BarEventCfg()

    def __post_init__(self) -> None:
        super().__post_init__()

        # A straight command walks the robot past the bars within seconds; resample often enough
        # that it keeps running into them.
        self.commands.base_velocity.resampling_time_range = (4.0, 4.0)


@configclass
class G1BarFlatEnvCfg_PLAY(G1BarFlatEnvCfg):
    """Small playback scene with randomization disabled."""

    def __post_init__(self) -> None:
        super().__post_init__()

        self.episode_length_s = 10.0

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

        # walk straight into a bar
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
