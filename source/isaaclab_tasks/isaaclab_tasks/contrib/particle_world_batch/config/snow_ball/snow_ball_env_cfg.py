# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Standalone ManagerBasedRLEnvCfg for the `snow_ball` slope rolling task (`example_mpm_snow_ball.py`)."""

from isaaclab_visualizers.newton import NewtonGLVisualizerCfg, NewtonRTXVisualizerCfg

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from .env_cfg import (
    SnowBallActionsCfg,
    SnowBallCommandsCfg,
    SnowBallCurriculumCfg,
    SnowBallEventsCfg,
    SnowBallObservationsCfg,
    SnowBallRewardsCfg,
    SnowBallSceneCfg,
    SnowBallTerminationsCfg,
    snow_ball_mpm_physics_cfg,
    snow_ball_particle_world_physics_cfg,
)


@configclass
class SnowBallParticleWorldEnvCfg(ManagerBasedRLEnvCfg):
    """Manager-based RL environment using `WorldParticle` (`ParticleWorldSolverCfg`) for snow-ball rolling."""

    scene: SnowBallSceneCfg = SnowBallSceneCfg(num_envs=4, env_spacing=4.0)
    observations: SnowBallObservationsCfg = SnowBallObservationsCfg()
    actions: SnowBallActionsCfg = SnowBallActionsCfg()
    commands: SnowBallCommandsCfg = SnowBallCommandsCfg()
    rewards: SnowBallRewardsCfg = SnowBallRewardsCfg()
    terminations: SnowBallTerminationsCfg = SnowBallTerminationsCfg()
    events: SnowBallEventsCfg = SnowBallEventsCfg()
    curriculum: SnowBallCurriculumCfg = SnowBallCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        self.decimation = 1
        self.episode_length_s = 6.0
        self.sim = SimulationCfg(
            dt=0.005,
            render_interval=4,
            physics=snow_ball_particle_world_physics_cfg(),
        )

        self.sim.visualizer_cfgs = [
            NewtonGLVisualizerCfg(
                eye=(1.65, -2.0, 1.55),
                lookat=(0.0, 0.0, -0.05),
                show_particles=True,
                particle_color=(0.82, 0.85, 0.94),
                streaming_view=False,
            ),
            NewtonRTXVisualizerCfg(
                eye=(1.65, -2.0, 1.55),
                lookat=(0.0, 0.0, -0.05),
                streaming_view=False,
            ),
        ]


@configclass
class SnowBallMPMEnvCfg(SnowBallParticleWorldEnvCfg):
    """Reference manager-based RL environment using native `SolverImplicitMPM` (`example_mpm_snow_ball.py`)."""

    def __post_init__(self):
        super().__post_init__()
        self.sim = SimulationCfg(
            dt=0.005,
            render_interval=4,
            physics=snow_ball_mpm_physics_cfg(),
        )
        self.sim.visualizer_cfgs = [
            NewtonGLVisualizerCfg(
                eye=(1.65, -2.0, 1.55),
                lookat=(0.0, 0.0, -0.05),
                show_particles=True,
                particle_color=(0.82, 0.85, 0.94),
                streaming_view=False,
            ),
            NewtonRTXVisualizerCfg(
                eye=(1.65, -2.0, 1.55),
                lookat=(0.0, 0.0, -0.05),
                streaming_view=False,
            ),
        ]
