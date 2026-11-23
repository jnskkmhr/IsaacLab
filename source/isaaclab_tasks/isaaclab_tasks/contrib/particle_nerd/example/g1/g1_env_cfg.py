# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Top-level environment configurations for the batched `g1` humanoid granular locomotion task."""

from isaaclab_visualizers.newton import NewtonGLVisualizerCfg, NewtonRTXVisualizerCfg

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils import configclass

from .env_cfg.action_cfg import G1ActionsCfg
from .env_cfg.commands_cfg import G1CommandsCfg
from .env_cfg.curriculum_cfg import G1CurriculumCfg
from .env_cfg.event_cfg import G1EventsCfg
from .env_cfg.observation_cfg import G1ObservationsCfg
from .env_cfg.physics_cfg import make_mpm_coupler_sim_cfg, make_particle_nerd_sim_cfg
from .env_cfg.reward_cfg import G1RewardsCfg
from .env_cfg.scene_cfg import MPM_VISUAL_COLOR, G1SceneCfg
from .env_cfg.termination_cfg import G1TerminationsCfg


@configclass
class G1ParticleNeRDEnvCfg(ManagerBasedRLEnvCfg):
    """Two-way coupled Unitree G1 29-DoF humanoid on granular sand using `NewtonParticleNeRDCouplerManager`."""

    scene: G1SceneCfg = G1SceneCfg(num_envs=4, env_spacing=5.0)
    observations: G1ObservationsCfg = G1ObservationsCfg()
    actions: G1ActionsCfg = G1ActionsCfg()
    commands: G1CommandsCfg = G1CommandsCfg()
    rewards: G1RewardsCfg = G1RewardsCfg()
    terminations: G1TerminationsCfg = G1TerminationsCfg()
    events: G1EventsCfg = G1EventsCfg()
    curriculum: G1CurriculumCfg = G1CurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        # The reference policy samples actions and observation history at 50 Hz.
        self.decimation = 4
        self.episode_length_s = 12.0
        self.sim = make_particle_nerd_sim_cfg()
        self.sim.visualizer_cfgs = [
            NewtonGLVisualizerCfg(
                eye=(2.4, -2.8, 1.5),
                lookat=(0.0, 0.0, -0.35),
                focal_length=24.0,
                origin_type="asset",
                origin_track_path="robot",
                show_particles=True,
                particle_color=MPM_VISUAL_COLOR,
                streaming_view=False,
            ),
            NewtonRTXVisualizerCfg(
                eye=(2.4, -2.8, 1.5),
                lookat=(0.0, 0.0, -0.35),
                origin_type="asset",
                origin_track_path="robot",
                streaming_view=False,
            ),
        ]


@configclass
class G1MPMCouplerEnvCfg(G1ParticleNeRDEnvCfg):
    """Ground-truth Implicit MPM counterpart using `NewtonMPMCouplerManager`."""

    def __post_init__(self):
        super().__post_init__()
        self.sim = make_mpm_coupler_sim_cfg()
        self.sim.visualizer_cfgs = [
            NewtonGLVisualizerCfg(
                eye=(2.4, -2.8, 1.5),
                lookat=(0.0, 0.0, -0.35),
                focal_length=24.0,
                origin_type="asset",
                origin_track_path="robot",
                show_particles=True,
                particle_color=MPM_VISUAL_COLOR,
                streaming_view=False,
            ),
            NewtonRTXVisualizerCfg(
                eye=(2.4, -2.8, 1.5),
                lookat=(0.0, 0.0, -0.35),
                origin_type="asset",
                origin_track_path="robot",
                streaming_view=False,
            ),
        ]
