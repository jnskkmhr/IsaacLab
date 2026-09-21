# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Soft-contact variants of the rigid G1 motion-tracking configurations."""

from isaaclab_visualizers.newton import NewtonGLVisualizerCfg

from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg
from isaaclab.utils import configclass

from isaaclab_tasks.contrib.mimic.config.g1_29dof.mimic_env_cfg import EnvCfg as RigidEnvCfg
from isaaclab_tasks.contrib.mimic.config.g1_29dof.mimic_env_cfg import EnvCfg_PLAY as RigidPlayEnvCfg

from .env_cfg.action_cfg import G1ActionsCfg
from .env_cfg.curriculum_cfg import G1CurriculumCfg
from .env_cfg.event_cfg import G1EventCfg
from .env_cfg.reward_cfg import G1RewardsCfg
from .env_cfg.scene_cfg import G1SceneCfg


@configclass
class EnvCfg(RigidEnvCfg):
    """Train the rigid mimic policy interface on randomized soft foot contact."""

    actions: G1ActionsCfg = G1ActionsCfg()
    scene: G1SceneCfg = G1SceneCfg(num_envs=4096, env_spacing=2.5)
    events: G1EventCfg = G1EventCfg()
    rewards: G1RewardsCfg = G1RewardsCfg()
    curriculum: G1CurriculumCfg = G1CurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        # Require the full motion before a completion can advance soft-layer depth.
        self.commands.motion.start_from_beginning = True


@configclass
class EnvCfg_PLAY(RigidPlayEnvCfg):
    """Play the mimic policy on a fixed-depth soft layer with randomized material."""

    actions: G1ActionsCfg = G1ActionsCfg()
    scene: G1SceneCfg = G1SceneCfg(num_envs=4096, env_spacing=2.5)
    events: G1EventCfg = G1EventCfg()
    rewards: G1RewardsCfg = G1RewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator.num_rows = 1
        self.scene.terrain.terrain_generator.num_cols = 1
        self.scene.terrain.terrain_generator.curriculum = False
        self.scene.terrain.terrain_generator.difficulty_range = (1.0, 1.0)

        self.sim.visualizer_cfgs = [
            NewtonGLVisualizerCfg(eye=(0.0, -4.0, 1.0)),
            # NewtonRTXVisualizerCfg(eye=(0.0, -4.0, 1.0)),
            # KitVisualizerCfg(eye=(0.0, -4.0, 1.0)),
        ]
        self.video_recorders = [
            VideoRecorderCfg(source="visualizer:newton_gl", output_dir=None, video_length=1600),
            # VideoRecorderCfg(source="visualizer:newton_rtx", output_dir=None, video_length=600),
            # VideoRecorderCfg(source="visualizer:kit", output_dir="videos/"),
        ]
