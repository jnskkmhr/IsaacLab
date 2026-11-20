# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_visualizers.kit import KitVisualizerCfg

from isaaclab.utils import configclass

from .env_cfg.commands_cfg import G1FlatCommandsCfg, G1FlatPlayCommandsCfg
from .env_cfg.curriculum_cfg import G1FlatCurriculumCfg, G1FlatPlayCurriculumCfg
from .env_cfg.event_cfg import G1FlatEventCfg, G1FlatPlayEventCfg
from .env_cfg.observation_cfg import G1FlatObservationsCfg, G1FlatPlayObservationsCfg
from .env_cfg.scene_cfg import G1FlatSceneCfg
from .rough_env_cfg import G1RoughEnvCfg


@configclass
class G1FlatEnvCfg(G1RoughEnvCfg):
    scene = G1FlatSceneCfg(num_envs=4096, env_spacing=2.5)
    observations = G1FlatObservationsCfg()
    events = G1FlatEventCfg()
    commands = G1FlatCommandsCfg()
    curriculum = G1FlatCurriculumCfg()


@configclass
class G1FlatEnvCfg_PLAY(G1FlatEnvCfg):
    scene = G1FlatSceneCfg(num_envs=50, env_spacing=2.5)
    observations = G1FlatPlayObservationsCfg()
    events = G1FlatPlayEventCfg()
    commands = G1FlatPlayCommandsCfg()
    curriculum = G1FlatPlayCurriculumCfg()
    episode_length_s = 10.0
    sim = G1FlatEnvCfg().sim.replace(
        visualizer_cfgs=[
            KitVisualizerCfg(
                eye=(0.0, -3.5, 0.2), lookat=(0.0, 0.0, 0.0), origin_type="asset", origin_track_path="robot"
            )
        ]
    )
