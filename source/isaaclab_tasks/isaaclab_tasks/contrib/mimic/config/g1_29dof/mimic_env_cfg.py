# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

from isaaclab_visualizers.newton import NewtonGLVisualizerCfg, NewtonRTXVisualizerCfg

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass

##
# Pre-defined configs
##
from .env_cfg import (
    G1ActionsCfg,
    G1CommandsCfg,
    G1CurriculumCfg,
    G1EventCfg,
    G1ObservationsCfg,
    G1PhysicsCfg,
    G1RewardsCfg,
    G1SceneCfg,
    G1TerminationsCfg,
)


@configclass
class EnvCfg(ManagerBasedRLEnvCfg):
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
        self.sim.physics_material = self.scene.terrain.physics_material  # type: ignore

        self.sim.visualizer_cfgs = [
            NewtonGLVisualizerCfg(eye=(12.0, 0.0, 3.0), headless=True),
            # KitVisualizerCfg(eye=(12.0, 0.0, 6.0), headless=True),
        ]

        self.video_recorders = [
            VideoRecorderCfg(source="visualizer:newton_gl", output_dir=None, video_length=200, video_interval=2000),
            # VideoRecorderCfg(source="visualizer:newton_gl", output_dir="videos/"),
            # VideoRecorderCfg(source="visualizer:kit", output_dir="videos/"),
        ]

        # self.events.assistive_wrench = None  # type: ignore


@configclass
class EnvCfg_PLAY(EnvCfg):
    """Configuration for the motion tracking environment in PLAY mode."""

    def __post_init__(self):
        """Post initialization."""
        super().__post_init__()

        self.terminations.anchor_pos = None  # type: ignore
        self.terminations.anchor_ori = None  # type: ignore
        self.terminations.ee_body_pos = None  # type: ignore
        self.terminations.base_ang_vel_exceed = None  # type: ignore

        self.commands.motion.start_from_beginning = True
        # self.events.reset_joints.params["position_range"] = (0.0, 0.0)
        self.events.assistive_wrench = None  # type: ignore

        self.sim.visualizer_cfgs = [
            NewtonGLVisualizerCfg(eye=(0.0, -4.0, 1.0)),
            # NewtonRTXVisualizerCfg(eye=(0.0, -4.0, 1.0)),
            # KitVisualizerCfg(eye=(0.0, -4.0, 1.0)),
        ]
        self.video_recorders = [
            # VideoRecorderCfg(source="visualizer:newton_gl", output_dir=None, video_length=600),
            VideoRecorderCfg(source="visualizer:newton_rtx", output_dir=None, video_length=600),
            # VideoRecorderCfg(source="visualizer:kit", output_dir="videos/"),
        ]
