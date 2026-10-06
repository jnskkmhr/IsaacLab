# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Standalone G1 whole-body environment composed from its term configurations."""

from isaaclab_visualizers.newton import NewtonGLVisualizerCfg, NewtonRTXVisualizerCfg

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

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
class G1WholeBodyEnvCfg(ManagerBasedRLEnvCfg):
    """Newton MJWarp baseline; set ``commands.whole_body.dataset_path`` to a generated NPZ."""

    scene: G1SceneCfg = G1SceneCfg(num_envs=1024, env_spacing=3.0)
    actions: G1ActionsCfg = G1ActionsCfg()
    observations: G1ObservationsCfg = G1ObservationsCfg()
    commands: G1CommandsCfg = G1CommandsCfg()
    rewards: G1RewardsCfg = G1RewardsCfg()
    events: G1EventCfg = G1EventCfg()
    terminations: G1TerminationsCfg = G1TerminationsCfg()
    curriculum: G1CurriculumCfg = G1CurriculumCfg()
    sim: SimulationCfg = SimulationCfg(dt=1.0 / 240.0, physics=G1PhysicsCfg())
    decimation = 4
    episode_length_s = 20.0

    def __post_init__(self) -> None:
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.visualizer_cfgs = [
            NewtonGLVisualizerCfg(eye=(12.0, 0.0, 6.0), headless=True),
            NewtonRTXVisualizerCfg(eye=(12.0, 0.0, 6.0), headless=True),
            ]
        self.video_recorders = [
            VideoRecorderCfg(
                source="visualizer:newton_gl",
                output_dir=None,
                video_length=200,
                video_interval=2000
            )
        ]

    def play_mode(self) -> None:
        """Enable the target ghost for interactive playback and video capture."""
        super().play_mode()
        self.scene.ghost.enabled = True


@configclass
class G1WholeBodyEnvCfgPlay(G1WholeBodyEnvCfg):
    """Play mode configuration for the G1 whole-body environment."""

    def __post_init__(self) -> None:
        super().__post_init__()

        self.sim.visualizer_cfgs = [
            NewtonGLVisualizerCfg(eye=(1.5, 0.0, 1.0), lookat=(0.0, 0.0, 0.8), headless=True),
            NewtonRTXVisualizerCfg(eye=(1.5, 0.0, 1.0), lookat=(0.0, 0.0, 0.8), headless=True),
            ]
        self.video_recorders = [
            VideoRecorderCfg(
                source="visualizer:newton_rtx",
                output_dir=None,
                video_length=200,
                video_interval=2000
            )
        ]
        self.scene.ghost.enabled = True
        self.events.push_foot = None
        self.events.push_robot = None

        self.commands.whole_body.resampling_time_range = (3.0, 4.0)
        self.commands.whole_body.initial_max_joint_rms = 2.0
        self.curriculum.target_joint_change.params["final_max_joint_rms"] = 2.0
