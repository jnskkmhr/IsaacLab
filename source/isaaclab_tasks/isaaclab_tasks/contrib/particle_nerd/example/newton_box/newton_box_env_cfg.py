# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Environment configurations for the `newton_box` task (Newton's `mpm_twoway_coupling` scene)."""

from isaaclab_visualizers.newton import NewtonGLVisualizerCfg, NewtonRTXVisualizerCfg

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils import configclass

from ..particle.env_cfg.action_cfg import NullActionCfg
from ..particle.env_cfg.commands_cfg import ParticleCommandsCfg
from ..particle.env_cfg.curriculum_cfg import ParticleCurriculumCfg
from ..particle.env_cfg.event_cfg import ParticleEventsCfg
from ..particle.env_cfg.observation_cfg import ParticleObservationsCfg
from ..particle.env_cfg.reward_cfg import ParticleRewardsCfg
from ..particle.env_cfg.termination_cfg import ParticleTerminationsCfg
from .env_cfg.physics_cfg import make_mpm_coupler_sim_cfg, make_particle_nerd_sim_cfg
from .env_cfg.scene_cfg import MPM_VISUAL_COLOR, NewtonBoxSceneCfg


def _visualizers():
    eye, lookat = (3.5, -4.0, 2.5), (0.0, 0.0, 0.8)
    return [
        NewtonGLVisualizerCfg(
            eye=eye,
            lookat=lookat,
            focal_length=24.0,
            show_particles=True,
            particle_color=MPM_VISUAL_COLOR,
            streaming_view=False,
        ),
        NewtonRTXVisualizerCfg(eye=eye, lookat=lookat, focal_length=20.0, streaming_view=False),
    ]


@configclass
class NewtonBoxActionsCfg:
    """Placeholder action; the boxes fall passively."""

    null = NullActionCfg(asset_name="box0")


@configclass
class NewtonBoxEnvCfg(ManagerBasedRLEnvCfg):
    """Six boxes dropped onto a sand bed, simulated with `ParticleNeRDSolver`."""

    scene: NewtonBoxSceneCfg = NewtonBoxSceneCfg(num_envs=1, env_spacing=30.0)
    observations: ParticleObservationsCfg = ParticleObservationsCfg()
    actions: NewtonBoxActionsCfg = NewtonBoxActionsCfg()
    commands: ParticleCommandsCfg = ParticleCommandsCfg()
    rewards: ParticleRewardsCfg = ParticleRewardsCfg()
    terminations: ParticleTerminationsCfg = ParticleTerminationsCfg()
    events: ParticleEventsCfg = ParticleEventsCfg()
    curriculum: ParticleCurriculumCfg = ParticleCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        self.decimation = 1
        self.episode_length_s = 6.0
        self.sim = make_particle_nerd_sim_cfg()
        self.sim.visualizer_cfgs = _visualizers()


@configclass
class NewtonBoxMPMEnvCfg(NewtonBoxEnvCfg):
    """The same scene with the implicit MPM ground-truth solver."""

    def __post_init__(self):
        super().__post_init__()
        self.sim = make_mpm_coupler_sim_cfg()
        self.sim.visualizer_cfgs = _visualizers()
