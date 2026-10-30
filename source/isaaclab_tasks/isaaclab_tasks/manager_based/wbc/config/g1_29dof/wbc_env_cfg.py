# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math 
import carb

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import CurriculumCfg
from isaaclab.envs.common import ViewerCfg

##
# Pre-defined configs
##

from .env_cfg import (
    G1ActionsCfg, 
    G1ObservationsCfg, 
    G1RewardsCfg, 
    G1SceneCfg,
    G1TerminationsCfg,
    G1CurriculumCfg, 
    G1EventCfg,
    G1CommandsCfg, 
)


@configclass
class G1WBCEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the motion tracking environment."""

    # Scene settings
    scene: G1SceneCfg = G1SceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: G1ObservationsCfg = G1ObservationsCfg()
    actions: G1ActionsCfg = G1ActionsCfg()
    commands: G1CommandsCfg = G1CommandsCfg()
    # MDP settings
    rewards: G1RewardsCfg = G1RewardsCfg()
    terminations: G1TerminationsCfg = G1TerminationsCfg()
    events: G1EventCfg = G1EventCfg()
    curriculum: G1CurriculumCfg = G1CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 30.0

        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        self.viewer = ViewerCfg(
            eye=(-1.0, -10.5, 0.2), 
            lookat=(-1.0, -0.0, 0.0),
            resolution=(1920, 1080), 
            origin_type="asset_root", 
            asset_name="robot"
        )

@configclass
class G1WBCEnvCfg_PLAY(G1WBCEnvCfg):
    """Configuration for the motion tracking environment in PLAY mode."""

    def __post_init__(self):
        """Post initialization."""
        super().__post_init__()
        
        self.episode_length_s = 1.0

        self.terminations.anchor_pos = None # type: ignore
        self.terminations.anchor_ori = None # type: ignore
        self.terminations.ee_body_pos = None # type: ignore
        self.terminations.base_ang_vel_exceed = None # type: ignore

        self.commands.motion.start_from_beginning = True
        self.commands.motion.joint_position_range = (0.0, 0.0)
        
        self.viewer = ViewerCfg(
            eye=(-1.0, -3.5, 0.2), 
            lookat=(-1.0, -0.0, 0.0),
            resolution=(1920, 1080), 
            origin_type="asset_root", 
            asset_name="robot"
        )