# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Top-level environment configurations for the batched `franka` sand-pushing task."""

from isaaclab_visualizers.newton import NewtonGLVisualizerCfg, NewtonRTXVisualizerCfg

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils import configclass

from .env_cfg.action_cfg import FrankaActionsCfg
from .env_cfg.commands_cfg import FrankaCommandsCfg
from .env_cfg.curriculum_cfg import FrankaCurriculumCfg
from .env_cfg.event_cfg import FrankaEventsCfg
from .env_cfg.observation_cfg import FrankaObservationsCfg
from .env_cfg.physics_cfg import make_mpm_coupler_sim_cfg, make_particle_nerd_local_sim_cfg
from .env_cfg.reward_cfg import FrankaRewardsCfg
from .env_cfg.scene_cfg import MPM_VISUAL_COLOR, FrankaSceneCfg
from .env_cfg.termination_cfg import FrankaTerminationsCfg


@configclass
class FrankaParticleNeRDLocalEnvCfg(ManagerBasedRLEnvCfg):
    """Two-way coupled Franka Panda arm pushing granular sand using `NewtonParticleNeRDLocalCouplerManager`."""

    scene: FrankaSceneCfg = FrankaSceneCfg(num_envs=4, env_spacing=4.0)
    observations: FrankaObservationsCfg = FrankaObservationsCfg()
    actions: FrankaActionsCfg = FrankaActionsCfg()
    commands: FrankaCommandsCfg = FrankaCommandsCfg()
    rewards: FrankaRewardsCfg = FrankaRewardsCfg()
    terminations: FrankaTerminationsCfg = FrankaTerminationsCfg()
    events: FrankaEventsCfg = FrankaEventsCfg()
    curriculum: FrankaCurriculumCfg = FrankaCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        self.decimation = 1
        self.episode_length_s = 6.0
        self.sim = make_particle_nerd_local_sim_cfg()
        self.sim.visualizer_cfgs = [
            NewtonGLVisualizerCfg(
                eye=(2.2, -2.0, 2.0),
                lookat=(0.55, 0.0, -0.1),
                focal_length=24.0,
                origin_type="asset",
                origin_track_path="robot",
                show_particles=True,
                particle_color=MPM_VISUAL_COLOR,
                streaming_view=False,
            ),
            NewtonRTXVisualizerCfg(
                eye=(2.2, -2.0, 2.0),
                lookat=(0.55, 0.0, -0.1),
                origin_type="asset",
                origin_track_path="robot",
                streaming_view=False,
            ),
        ]


@configclass
class FrankaMPMCouplerEnvCfg(FrankaParticleNeRDLocalEnvCfg):
    """Ground-truth Implicit MPM counterpart using `NewtonMPMCouplerManager`."""

    def __post_init__(self):
        super().__post_init__()
        self.sim = make_mpm_coupler_sim_cfg()
        self.sim.visualizer_cfgs = [
            NewtonGLVisualizerCfg(
                eye=(2.2, -2.0, 2.0),
                lookat=(0.55, 0.0, -0.1),
                focal_length=24.0,
                origin_type="asset",
                origin_track_path="robot",
                show_particles=True,
                particle_color=MPM_VISUAL_COLOR,
                streaming_view=False,
            ),
            NewtonRTXVisualizerCfg(
                eye=(2.2, -2.0, 2.0),
                lookat=(0.55, 0.0, -0.1),
                origin_type="asset",
                origin_track_path="robot",
                streaming_view=False,
            ),
        ]
