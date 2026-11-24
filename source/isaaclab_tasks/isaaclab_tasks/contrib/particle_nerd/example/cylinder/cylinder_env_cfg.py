# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Top-level environment configurations for the batched `cylinder` two-way coupling task."""

from isaaclab_visualizers.newton import NewtonGLVisualizerCfg, NewtonRTXVisualizerCfg

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils import configclass

from .env_cfg.action_cfg import CylinderActionsCfg
from .env_cfg.commands_cfg import CylinderCommandsCfg
from .env_cfg.curriculum_cfg import CylinderCurriculumCfg
from .env_cfg.event_cfg import CylinderEventsCfg
from .env_cfg.observation_cfg import CylinderObservationsCfg
from .env_cfg.physics_cfg import make_mpm_coupler_sim_cfg, make_particle_nerd_sim_cfg
from .env_cfg.reward_cfg import CylinderRewardsCfg
from .env_cfg.scene_cfg import MPM_VISUAL_COLOR, CylinderSceneCfg
from .env_cfg.termination_cfg import CylinderTerminationsCfg


@configclass
class CylinderParticleNeRDEnvCfg(ManagerBasedRLEnvCfg):
    """Two-way coupled rigid cylinder on granular sand bed using `NewtonParticleNeRDCouplerManager`."""

    scene: CylinderSceneCfg = CylinderSceneCfg(num_envs=4, env_spacing=4.0)
    observations: CylinderObservationsCfg = CylinderObservationsCfg()
    actions: CylinderActionsCfg = CylinderActionsCfg()
    commands: CylinderCommandsCfg = CylinderCommandsCfg()
    rewards: CylinderRewardsCfg = CylinderRewardsCfg()
    terminations: CylinderTerminationsCfg = CylinderTerminationsCfg()
    events: CylinderEventsCfg = CylinderEventsCfg()
    curriculum: CylinderCurriculumCfg = CylinderCurriculumCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        self.decimation = 1
        self.episode_length_s = 6.0
        self.sim = make_particle_nerd_sim_cfg()
        self.sim.visualizer_cfgs = [
            NewtonGLVisualizerCfg(
                eye=(1.65, -2.0, 1.55),
                lookat=(0.0, 0.0, -0.05),
                focal_length=24.0,
                origin_type="asset",
                origin_track_path="cylinder",
                show_particles=True,
                particle_color=MPM_VISUAL_COLOR,
                streaming_view=False,
            ),
            NewtonRTXVisualizerCfg(
                eye=(1.65, -2.0, 1.55),
                lookat=(0.0, 0.0, -0.05),
                focal_length=20.0,
                origin_type="asset",
                origin_track_path="cylinder",
                streaming_view=False,
            ),
        ]


@configclass
class CylinderMPMCouplerEnvCfg(CylinderParticleNeRDEnvCfg):
    """Ground-truth Implicit MPM counterpart using `NewtonMPMCouplerManager`."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.sim = make_mpm_coupler_sim_cfg()
        self.sim.visualizer_cfgs = [
            NewtonGLVisualizerCfg(
                eye=(1.65, -2.0, 1.55),
                lookat=(0.0, 0.0, -0.05),
                focal_length=24.0,
                origin_type="asset",
                origin_track_path="cylinder",
                show_particles=True,
                particle_color=MPM_VISUAL_COLOR,
                streaming_view=False,
            ),
            NewtonRTXVisualizerCfg(
                eye=(1.65, -2.0, 1.55),
                lookat=(0.0, 0.0, -0.05),
                focal_length=20.0,
                origin_type="asset",
                origin_track_path="cylinder",
                streaming_view=False,
            ),
        ]
