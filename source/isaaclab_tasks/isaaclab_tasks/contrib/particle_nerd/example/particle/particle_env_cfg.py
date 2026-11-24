# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Top-level environment configurations for the `particle` task: a sand block dropped or tossed onto flat ground."""

from isaaclab_visualizers.newton import NewtonGLVisualizerCfg, NewtonRTXVisualizerCfg

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils import configclass

from .env_cfg.action_cfg import ParticleActionsCfg
from .env_cfg.commands_cfg import ParticleCommandsCfg
from .env_cfg.curriculum_cfg import ParticleCurriculumCfg
from .env_cfg.event_cfg import ParticleEventsCfg
from .env_cfg.observation_cfg import ParticleObservationsCfg
from .env_cfg.physics_cfg import make_mpm_coupler_sim_cfg, make_particle_nerd_sim_cfg
from .env_cfg.reward_cfg import ParticleRewardsCfg
from .env_cfg.scene_cfg import (
    MPM_VISUAL_COLOR,
    ParticleCubeSceneCfg,
    ParticleCylinderSceneCfg,
    ParticleSphereSceneCfg,
)
from .env_cfg.termination_cfg import ParticleTerminationsCfg

THROW_VELOCITY = (1.0, 0.0, 0.25)
"""World-frame release velocity of the tossed blocks (m/s)."""


def _visualizers() -> list[NewtonGLVisualizerCfg | NewtonRTXVisualizerCfg]:
    eye, lookat = (1.05, -1.7, 0.45), (0.2, 0.0, -0.22)
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
class ParticleEnvCfg(ManagerBasedRLEnvCfg):
    """Sand block simulated with `ParticleNeRDSolver` on open flat ground (cube, dropped by default)."""

    scene: ParticleCubeSceneCfg = ParticleCubeSceneCfg(num_envs=1, env_spacing=30.0)
    observations: ParticleObservationsCfg = ParticleObservationsCfg()
    actions: ParticleActionsCfg = ParticleActionsCfg()
    commands: ParticleCommandsCfg = ParticleCommandsCfg()
    rewards: ParticleRewardsCfg = ParticleRewardsCfg()
    terminations: ParticleTerminationsCfg = ParticleTerminationsCfg()
    events: ParticleEventsCfg = ParticleEventsCfg()
    curriculum: ParticleCurriculumCfg = ParticleCurriculumCfg()
    release_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.decimation = 1
        self.episode_length_s = 6.0
        self.sim = make_particle_nerd_sim_cfg()
        self.sim.visualizer_cfgs = _visualizers()
        self.events.reset_sand.params["velocity"] = self.release_velocity


@configclass
class ParticleCubeDropEnvCfg(ParticleEnvCfg):
    """Cube of sand dropped from rest; Particle NeRD Local."""

    scene: ParticleCubeSceneCfg = ParticleCubeSceneCfg(num_envs=1, env_spacing=30.0)
    release_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)


@configclass
class ParticleCubeThrowEnvCfg(ParticleEnvCfg):
    """Cube of sand tossed at 1.0 m/s along +x and 0.25 m/s up; Particle NeRD Local."""

    scene: ParticleCubeSceneCfg = ParticleCubeSceneCfg(num_envs=1, env_spacing=30.0)
    release_velocity: tuple[float, float, float] = THROW_VELOCITY


@configclass
class ParticleSphereDropEnvCfg(ParticleEnvCfg):
    """Sphere of sand dropped from rest; Particle NeRD Local."""

    scene: ParticleSphereSceneCfg = ParticleSphereSceneCfg(num_envs=1, env_spacing=30.0)
    release_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)


@configclass
class ParticleSphereThrowEnvCfg(ParticleEnvCfg):
    """Sphere of sand tossed at 1.0 m/s along +x and 0.25 m/s up; Particle NeRD Local."""

    scene: ParticleSphereSceneCfg = ParticleSphereSceneCfg(num_envs=1, env_spacing=30.0)
    release_velocity: tuple[float, float, float] = THROW_VELOCITY


@configclass
class ParticleCylinderDropEnvCfg(ParticleEnvCfg):
    """Cylinder of sand dropped from rest; Particle NeRD Local."""

    scene: ParticleCylinderSceneCfg = ParticleCylinderSceneCfg(num_envs=1, env_spacing=30.0)
    release_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)


@configclass
class ParticleCylinderThrowEnvCfg(ParticleEnvCfg):
    """Cylinder of sand tossed at 1.0 m/s along +x and 0.25 m/s up; Particle NeRD Local."""

    scene: ParticleCylinderSceneCfg = ParticleCylinderSceneCfg(num_envs=1, env_spacing=30.0)
    release_velocity: tuple[float, float, float] = THROW_VELOCITY


@configclass
class ParticleCubeDropMPMEnvCfg(ParticleCubeDropEnvCfg):
    """Cube of sand dropped from rest; implicit MPM ground truth."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.sim = make_mpm_coupler_sim_cfg()
        self.sim.visualizer_cfgs = _visualizers()


@configclass
class ParticleCubeThrowMPMEnvCfg(ParticleCubeThrowEnvCfg):
    """Cube of sand tossed at 1.0 m/s along +x and 0.25 m/s up; implicit MPM ground truth."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.sim = make_mpm_coupler_sim_cfg()
        self.sim.visualizer_cfgs = _visualizers()


@configclass
class ParticleSphereDropMPMEnvCfg(ParticleSphereDropEnvCfg):
    """Sphere of sand dropped from rest; implicit MPM ground truth."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.sim = make_mpm_coupler_sim_cfg()
        self.sim.visualizer_cfgs = _visualizers()


@configclass
class ParticleSphereThrowMPMEnvCfg(ParticleSphereThrowEnvCfg):
    """Sphere of sand tossed at 1.0 m/s along +x and 0.25 m/s up; implicit MPM ground truth."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.sim = make_mpm_coupler_sim_cfg()
        self.sim.visualizer_cfgs = _visualizers()


@configclass
class ParticleCylinderDropMPMEnvCfg(ParticleCylinderDropEnvCfg):
    """Cylinder of sand dropped from rest; implicit MPM ground truth."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.sim = make_mpm_coupler_sim_cfg()
        self.sim.visualizer_cfgs = _visualizers()


@configclass
class ParticleCylinderThrowMPMEnvCfg(ParticleCylinderThrowEnvCfg):
    """Cylinder of sand tossed at 1.0 m/s along +x and 0.25 m/s up; implicit MPM ground truth."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.sim = make_mpm_coupler_sim_cfg()
        self.sim.visualizer_cfgs = _visualizers()
