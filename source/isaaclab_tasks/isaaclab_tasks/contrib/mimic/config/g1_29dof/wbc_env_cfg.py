# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.common import ViewerCfg
from isaaclab.sim import SimulationCfg

##
# Pre-defined configs
##

from .env_cfg import (
    G1ActionsCfg,
    G1ObservationsCfg,
    G1PhysicsCfg,
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

    sim: SimulationCfg = SimulationCfg(physics=G1PhysicsCfg())  # type: ignore
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
        self.episode_length_s = 30.0

        # simulation settings
        self.sim.dt = 1 / 240
        self.decimation = 4  # 60hz
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material

        self.viewer = ViewerCfg(
            eye=(0.0, -12.0, 0.4),
            lookat=(0.0, -0.0, 0.0),
            resolution=(1920, 1080),
            origin_type="asset_root",
            asset_name="robot",
        )


@configclass
class G1WBCEnvCfg_PLAY(G1WBCEnvCfg):
    """Configuration for the motion tracking environment in PLAY mode."""

    def __post_init__(self):
        """Post initialization."""
        super().__post_init__()

        self.terminations.anchor_pos = None  # type: ignore
        self.terminations.anchor_ori = None  # type: ignore
        self.terminations.ee_body_pos = None  # type: ignore
        self.terminations.base_ang_vel_exceed = None  # type: ignore

        self.commands.motion.start_from_beginning = True
        self.events.reset_joints.params["position_range"] = (0.0, 0.0)
        self.events.assistive_wrench = None  # type: ignore

        self.viewer = ViewerCfg(
            eye=(-3.0, -3.0, 1.0),
            lookat=(0.0, -0.0, 0.0),
            resolution=(1920, 1080),
            origin_type="asset_root",
            asset_name="robot",
        )
