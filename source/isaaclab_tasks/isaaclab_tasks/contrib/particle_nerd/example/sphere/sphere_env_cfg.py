# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Top-level environment configurations for the batched `sphere` two-way coupling task."""

from isaaclab_visualizers.newton import NewtonGLVisualizerCfg, NewtonRTXVisualizerCfg

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils import configclass

from .env_cfg.action_cfg import SphereActionsCfg
from .env_cfg.commands_cfg import SphereCommandsCfg
from .env_cfg.curriculum_cfg import SphereCurriculumCfg
from .env_cfg.event_cfg import SphereEventsCfg
from .env_cfg.observation_cfg import SphereObservationsCfg
from .env_cfg.physics_cfg import make_mpm_coupler_sim_cfg, make_particle_nerd_sim_cfg
from .env_cfg.reward_cfg import SphereRewardsCfg
from .env_cfg.scene_cfg import MPM_VISUAL_COLOR, SphereSceneCfg
from .env_cfg.termination_cfg import SphereTerminationsCfg


@configclass
class SphereParticleNeRDEnvCfg(ManagerBasedRLEnvCfg):
    """Two-way coupled rigid sphere on granular sand bed using `NewtonParticleNeRDCouplerManager`."""

    scene: SphereSceneCfg = SphereSceneCfg(num_envs=4, env_spacing=4.0)
    observations: SphereObservationsCfg = SphereObservationsCfg()
    actions: SphereActionsCfg = SphereActionsCfg()
    commands: SphereCommandsCfg = SphereCommandsCfg()
    rewards: SphereRewardsCfg = SphereRewardsCfg()
    terminations: SphereTerminationsCfg = SphereTerminationsCfg()
    events: SphereEventsCfg = SphereEventsCfg()
    curriculum: SphereCurriculumCfg = SphereCurriculumCfg()

    def __post_init__(self):
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
                origin_track_path="sphere",
                show_particles=True,
                particle_color=MPM_VISUAL_COLOR,
                streaming_view=False,
            ),
            NewtonRTXVisualizerCfg(
                eye=(1.65, -2.0, 1.55),
                lookat=(0.0, 0.0, -0.05),
                focal_length=20.0,
                origin_type="asset",
                origin_track_path="sphere",
                streaming_view=False,
            ),
        ]


@configclass
class SphereMPMCouplerEnvCfg(SphereParticleNeRDEnvCfg):
    """Ground-truth Implicit MPM counterpart using `NewtonMPMCouplerManager`."""

    def __post_init__(self):
        super().__post_init__()
        self.sim = make_mpm_coupler_sim_cfg()
        self.sim.visualizer_cfgs = [
            NewtonGLVisualizerCfg(
                eye=(1.65, -2.0, 1.55),
                lookat=(0.0, 0.0, -0.05),
                focal_length=24.0,
                origin_type="asset",
                origin_track_path="sphere",
                show_particles=True,
                particle_color=MPM_VISUAL_COLOR,
                streaming_view=False,
            ),
            NewtonRTXVisualizerCfg(
                eye=(1.65, -2.0, 1.55),
                lookat=(0.0, 0.0, -0.05),
                focal_length=20.0,
                origin_type="asset",
                origin_track_path="sphere",
                streaming_view=False,
            ),
        ]
