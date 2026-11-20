# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_visualizers.kit import KitVisualizerCfg

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from .env_cfg.action_cfg import G1ActionsCfg
from .env_cfg.commands_cfg import G1CommandsCfg, G1RoughPlayCommandsCfg
from .env_cfg.curriculum_cfg import G1CurriculumCfg
from .env_cfg.event_cfg import G1RoughEventCfg, G1RoughPlayEventCfg
from .env_cfg.observation_cfg import G1ObservationsCfg, G1RoughPlayObservationsCfg
from .env_cfg.physics_cfg import G1PhysicsCfg
from .env_cfg.reward_cfg import G1RewardsCfg
from .env_cfg.scene_cfg import G1SceneCfg
from .env_cfg.termination_cfg import G1TerminationsCfg


@configclass
class G1RoughEnvCfg(ManagerBasedRLEnvCfg):
    scene = G1SceneCfg(num_envs=4096, env_spacing=2.5)
    actions = G1ActionsCfg()
    observations = G1ObservationsCfg()
    commands = G1CommandsCfg()
    events = G1RoughEventCfg()
    rewards = G1RewardsCfg()
    curriculum = G1CurriculumCfg()
    terminations = G1TerminationsCfg()
    seed = 42
    decimation = 4
    episode_length_s = 20.0
    sim = SimulationCfg(
        dt=0.005,
        render_interval=4,
        physics=G1PhysicsCfg(),
        physics_material=G1SceneCfg().terrain.physics_material,
        visualizer_cfgs=[KitVisualizerCfg(eye=(7.5, 7.5, 7.5), lookat=(0.0, 0.0, 0.0))],
    )
    video_recorders = []

    def __post_init__(self):
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        self.scene.contact_forces.update_period = self.sim.dt
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.curriculum = self.curriculum.terrain_levels is not None


@configclass
class G1RoughEnvCfg_PLAY(G1RoughEnvCfg):
    scene = G1SceneCfg(num_envs=50, env_spacing=2.5)
    events = G1RoughPlayEventCfg()
    commands = G1RoughPlayCommandsCfg()
    observations = G1RoughPlayObservationsCfg()
    episode_length_s = 40.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.max_init_terrain_level = None
        self.scene.terrain.terrain_generator.num_rows = 5
        self.scene.terrain.terrain_generator.num_cols = 5
        self.scene.terrain.terrain_generator.curriculum = False
