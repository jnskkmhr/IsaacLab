# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Top-level environment configurations for the batched `go2` quadruped granular locomotion task."""

from isaaclab_visualizers.newton import NewtonGLVisualizerCfg, NewtonRTXVisualizerCfg

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils import configclass

from .env_cfg.action_cfg import Go2ActionsCfg
from .env_cfg.commands_cfg import Go2CommandsCfg
from .env_cfg.curriculum_cfg import Go2CurriculumCfg
from .env_cfg.event_cfg import Go2EventsCfg
from .env_cfg.observation_cfg import Go2ObservationsCfg
from .env_cfg.physics_cfg import make_mpm_coupler_sim_cfg, make_particle_nerd_sim_cfg
from .env_cfg.reward_cfg import Go2RewardsCfg
from .env_cfg.scene_cfg import MPM_VISUAL_COLOR, Go2SceneCfg
from .env_cfg.termination_cfg import Go2TerminationsCfg


@configclass
class Go2ParticleNeRDEnvCfg(ManagerBasedRLEnvCfg):
    """Two-way coupled Unitree Go2 quadruped on granular sand using `NewtonParticleNeRDCouplerManager`."""

    scene: Go2SceneCfg = Go2SceneCfg(num_envs=4, env_spacing=4.0)
    observations: Go2ObservationsCfg = Go2ObservationsCfg()
    actions: Go2ActionsCfg = Go2ActionsCfg()
    commands: Go2CommandsCfg = Go2CommandsCfg()
    rewards: Go2RewardsCfg = Go2RewardsCfg()
    terminations: Go2TerminationsCfg = Go2TerminationsCfg()
    events: Go2EventsCfg = Go2EventsCfg()
    curriculum: Go2CurriculumCfg = Go2CurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        self.decimation = 1
        self.episode_length_s = 6.0
        self.sim = make_particle_nerd_sim_cfg()
        self.sim.visualizer_cfgs = [
            NewtonGLVisualizerCfg(
                eye=(1.65, -2.0, 1.65),
                lookat=(0.0, 0.0, -0.25),
                focal_length=24.0,
                origin_type="asset",
                origin_track_path="robot",
                show_particles=True,
                particle_color=MPM_VISUAL_COLOR,
                streaming_view=False,
            ),
            NewtonRTXVisualizerCfg(
                eye=(1.65, -2.0, 1.65),
                lookat=(0.0, 0.0, -0.25),
                origin_type="asset",
                origin_track_path="robot",
                streaming_view=False,
            ),
        ]


@configclass
class Go2MPMCouplerEnvCfg(Go2ParticleNeRDEnvCfg):
    """Ground-truth Implicit MPM counterpart using `NewtonMPMCouplerManager`."""

    def __post_init__(self):
        super().__post_init__()
        self.sim = make_mpm_coupler_sim_cfg()
        self.sim.visualizer_cfgs = [
            NewtonGLVisualizerCfg(
                eye=(1.65, -2.0, 1.65),
                lookat=(0.0, 0.0, -0.25),
                focal_length=24.0,
                origin_type="asset",
                origin_track_path="robot",
                show_particles=True,
                particle_color=MPM_VISUAL_COLOR,
                streaming_view=False,
            ),
            NewtonRTXVisualizerCfg(
                eye=(1.65, -2.0, 1.65),
                lookat=(0.0, 0.0, -0.25),
                origin_type="asset",
                origin_track_path="robot",
                streaming_view=False,
            ),
        ]
