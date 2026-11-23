# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Standalone ManagerBasedRLEnvCfg for the `box` two-way coupling task (`example_mpm_twoway_coupling.py`)."""

from isaaclab_visualizers.newton import NewtonGLVisualizerCfg, NewtonRTXVisualizerCfg

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from .env_cfg import (
    BoxActionsCfg,
    BoxCommandsCfg,
    BoxCurriculumCfg,
    BoxEventsCfg,
    BoxObservationsCfg,
    BoxRewardsCfg,
    BoxSceneCfg,
    BoxTerminationsCfg,
    box_mpm_physics_cfg,
    box_particle_world_physics_cfg,
)


@configclass
class BoxParticleWorldEnvCfg(ManagerBasedRLEnvCfg):
    """Manager-based RL environment using `WorldParticle` (`ParticleWorldSolverCfg`) for 2-way box-sand coupling."""

    scene: BoxSceneCfg = BoxSceneCfg(num_envs=4, env_spacing=4.0)
    observations: BoxObservationsCfg = BoxObservationsCfg()
    actions: BoxActionsCfg = BoxActionsCfg()
    commands: BoxCommandsCfg = BoxCommandsCfg()
    rewards: BoxRewardsCfg = BoxRewardsCfg()
    terminations: BoxTerminationsCfg = BoxTerminationsCfg()
    events: BoxEventsCfg = BoxEventsCfg()
    curriculum: BoxCurriculumCfg = BoxCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        self.decimation = 1
        self.episode_length_s = 6.0
        self.sim = SimulationCfg(
            dt=0.005,
            render_interval=4,
            physics=box_particle_world_physics_cfg(),
        )

        self.sim.visualizer_cfgs = [
            NewtonGLVisualizerCfg(
                eye=(1.65, -2.0, 1.55),
                lookat=(0.0, 0.0, -0.05),
                show_particles=True,
                particle_color=(0.70, 0.60, 0.40),
                streaming_view=False,
            ),
            NewtonRTXVisualizerCfg(
                eye=(1.65, -2.0, 1.55),
                lookat=(0.0, 0.0, -0.05),
                streaming_view=False,
            ),
        ]


@configclass
class BoxMPMEnvCfg(BoxParticleWorldEnvCfg):
    """Reference manager-based RL environment using native `SolverImplicitMPM` (`example_mpm_twoway_coupling.py`)."""

    def __post_init__(self):
        super().__post_init__()
        self.sim = SimulationCfg(
            dt=0.005,
            render_interval=4,
            physics=box_mpm_physics_cfg(),
        )
        self.sim.visualizer_cfgs = [
            NewtonGLVisualizerCfg(
                eye=(1.65, -2.0, 1.55),
                lookat=(0.0, 0.0, -0.05),
                show_particles=True,
                particle_color=(0.70, 0.60, 0.40),
                streaming_view=False,
            ),
            NewtonRTXVisualizerCfg(
                eye=(1.65, -2.0, 1.55),
                lookat=(0.0, 0.0, -0.05),
                streaming_view=False,
            ),
        ]
