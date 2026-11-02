# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reference-free stair ascent with the pretrained actor and critic interfaces."""

import isaaclab.envs.mdp as mdp
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.configclass import configclass

from . import mdp as goal_mdp
from .env_cfg.commands_cfg import BODY_NAMES, JOINT_NAMES
from .env_cfg.terrain_cfg import STAIRS_TERRAIN_CFG
from .mimic_env_cfg import EnvCfg, EnvCfg_PLAY


@configclass
class StairsFinetuneEnvCfg(EnvCfg):
    """Climb randomized stairs from ground level to the top platform without motion data."""

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 15.0
        self.curriculum.task_resets = None
        self.scene.terrain.class_type = TerrainImporterCfg().class_type
        # Keep destination rewards without explicitly rewarding approach speed.
        self.rewards.goal_progress = None
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = STAIRS_TERRAIN_CFG.copy()
        self.scene.terrain.max_init_terrain_level = 2
        self.commands.motion = goal_mdp.StairGoalCommandCfg(
            body_names=BODY_NAMES,
            resampling_time_range=(1.0e9, 1.0e9),
            debug_vis=True,
        )
        self.events.reset_joints = EventTerm(
            func=mdp.reset_joints_by_offset,
            mode="reset",
            params={"position_range": (-0.05, 0.05), "velocity_range": (0.0, 0.0)},
        )
        self.events.reset_base = EventTerm(
            func=goal_mdp.reset_stair_root_state, mode="reset", params={"command_name": "motion"}
        )

        # Keep observation ordering, dimensions, and symmetry mappings for checkpoint loading.
        for name, dimension in (
            ("reference_joint_pos", len(JOINT_NAMES)),
            ("reference_phase", 1),
            ("motion_anchor_pos_b", 3),
            ("motion_anchor_ori_b", 6),
        ):
            term = getattr(self.observations.critic, name)
            term.func = goal_mdp.zero_reference
            term.params = {"dimension": dimension}

        self.rewards.task_motion_style = None
        self.rewards.motion_global_anchor_pos = None
        self.rewards.motion_global_anchor_ori = None
        self.rewards.motion_global_anchor_lin_vel = None
        self.rewards.motion_global_anchor_ang_vel = None
        self.rewards.motion_body_pos = None
        self.rewards.motion_body_ori = None
        self.rewards.motion_feet_lin_vel = None
        self.terminations.end_of_reference = None
        self.terminations.anchor_pos = None
        self.terminations.anchor_ori = None
        self.terminations.ee_body_pos = None
        self.curriculum.stair_levels = CurrTerm(func=goal_mdp.stair_levels, params={"command_name": "motion"})


@configclass
class StairsFinetuneEnvCfg_PLAY(StairsFinetuneEnvCfg):
    """Evaluate ascending stairs at fixed sampled difficulty, retaining the existing play camera."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.terrain.max_init_terrain_level = None
        self.curriculum.stair_levels = None
        self.events.reset_joints.params["position_range"] = (0.0, 0.0)
        self.observations.policy.enable_corruption = False
        play_cfg = EnvCfg_PLAY()
        self.sim.visualizer_cfgs = play_cfg.sim.visualizer_cfgs
        self.sim.visualizer_cfgs[0].max_visible_envs = 50
        self.sim.visualizer_cfgs[0].visible_env_indices = None
        self.video_recorders = play_cfg.video_recorders


@configclass
class MixedStairsFinetuneEnvCfg(EnvCfg):
    """Retain reference imitation while generalizing ascent on generated staircases."""

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 15.0
        self.curriculum.task_resets = None
        self.scene.terrain.class_type = TerrainImporterCfg().class_type
        # Keep destination rewards without explicitly rewarding approach speed.
        self.rewards.goal_progress = None
        generator = STAIRS_TERRAIN_CFG.copy()
        generator.num_cols = 10
        # One reference column and three columns per generated step depth.
        generator.sub_terrains = {
            "reference": goal_mdp.ReferenceStairsTerrainCfg(
                proportion=1.0, motion_files=self.commands.motion.motion_files
            ),
            **{name: cfg.replace(proportion=3.0) for name, cfg in generator.sub_terrains.items()},
        }
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = generator
        self.scene.terrain.max_init_terrain_level = 2
        self.commands.motion = goal_mdp.MixedStairGoalCommandCfg(
            asset_name="robot",
            resampling_time_range=(1.0e9, 1.0e9),
            debug_vis=True,
            joint_names=JOINT_NAMES,
            body_names=BODY_NAMES,
            anchor_body_name="torso_link",
            motion_files=self.commands.motion.motion_files,
        )
        self.events.reset_joints.func = goal_mdp.reset_mixed_stair_joints
        self.events.reset_base.func = goal_mdp.reset_mixed_stair_root_state
        self.curriculum.stair_levels = CurrTerm(func=goal_mdp.stair_levels, params={"command_name": "motion"})


@configclass
class MixedStairsFinetuneEnvCfg_PLAY(MixedStairsFinetuneEnvCfg):
    """Evaluate the mixed-trained policy on generated stairs without references."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.terrain.max_init_terrain_level = None
        self.commands.motion.imitation_probability = 0.0
        self.curriculum.stair_levels = None
        self.events.reset_joints.params["position_range"] = (0.0, 0.0)
        self.observations.policy.enable_corruption = False
        play_cfg = EnvCfg_PLAY()
        self.sim.visualizer_cfgs = play_cfg.sim.visualizer_cfgs
        self.sim.visualizer_cfgs[0].max_visible_envs = 50
        self.sim.visualizer_cfgs[0].visible_env_indices = None
        self.video_recorders = play_cfg.video_recorders
