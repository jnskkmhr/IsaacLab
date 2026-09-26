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

from .env_cfg.action_cfg import G1ActionsCfg, G1ActionsFinetuneCfg
from .env_cfg.curriculum_cfg import G1CurriculumCfg
from .env_cfg.event_cfg import G1EventCfg, G1EventFinetuneCfg
from .env_cfg.reward_cfg import G1RewardsCfg, G1RewardsFinetuneCfg
from .env_cfg.scene_cfg import G1SceneCfg


@configclass
class EnvCfg(RigidEnvCfg):
    """Fine-tune the rigid mimic policy with stance pushes on a fixed-depth soft layer."""

    actions: G1ActionsCfg = G1ActionsCfg()
    scene: G1SceneCfg = G1SceneCfg(num_envs=4096, env_spacing=2.5)
    events: G1EventCfg = G1EventCfg()
    rewards: G1RewardsCfg = G1RewardsCfg()
    curriculum: G1CurriculumCfg = G1CurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        # Evaluate complete jumps at a fixed depth while learning push recovery.
        self.commands.motion.start_from_beginning = True


@configclass
class EnvCfgFinetune(EnvCfg):
    actions: G1ActionsFinetuneCfg = G1ActionsFinetuneCfg()
    events: G1EventFinetuneCfg = G1EventFinetuneCfg()
    rewards: G1RewardsFinetuneCfg = G1RewardsFinetuneCfg()
    """Fine-tune the rigid mimic policy with stance pushes on a fixed-depth soft layer."""

    def __post_init__(self):
        super().__post_init__()

        self.events.assistive_wrench = None  # type: ignore
        self.events.reset_base.params["pose_range"]["x"] = (-80.0, 80.0)
        self.events.reset_base.params["pose_range"]["y"] = (-80.0, 80.0)

        # disable tracking based termination
        self.terminations.anchor_pos = None
        self.terminations.anchor_ori = None
        # self.terminations.ee_body_pos = None

        # finetuning
        self.curriculum.terrain_levels = None
        self.scene.terrain.terrain_generator.num_rows = 1
        self.scene.terrain.terrain_generator.num_cols = 1
        self.scene.terrain.terrain_generator.curriculum = False
        self.scene.terrain.terrain_generator.difficulty_range = (1.0, 1.0)
        self.scene.terrain.terrain_generator.size = (100.0, 100.0)


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
