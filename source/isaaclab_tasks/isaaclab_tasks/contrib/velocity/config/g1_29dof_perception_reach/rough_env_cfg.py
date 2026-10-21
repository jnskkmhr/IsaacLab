# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Standalone G1 reach task, with CNN and AME observation layouts."""

from isaaclab_visualizers.newton import NewtonGLVisualizerCfg, NewtonRTXVisualizerCfg

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass

from .env_cfg import (
    G1ActionsCfg,
    G1CommandsCfg,
    G1CurriculumCfg,
    G1EventCfg,
    G1PerceptionObservationsCfg,
    G1PerceptionSceneCfg,
    G1PhysicsCfg,
    G1RewardsCfg,
    G1TerminationsCfg,
)


@configclass
class G1RoughPerceptionReachEnvCfg(ManagerBasedRLEnvCfg):
    """Reach a supported base pose and remain there until the episode ends."""

    scene: G1PerceptionSceneCfg = G1PerceptionSceneCfg(num_envs=128, env_spacing=2.5)
    observations: G1PerceptionObservationsCfg = G1PerceptionObservationsCfg()
    sim: SimulationCfg = SimulationCfg(physics=G1PhysicsCfg())
    actions: G1ActionsCfg = G1ActionsCfg()
    commands: G1CommandsCfg = G1CommandsCfg()
    rewards: G1RewardsCfg = G1RewardsCfg()
    terminations: G1TerminationsCfg = G1TerminationsCfg()
    events: G1EventCfg = G1EventCfg()
    curriculum: G1CurriculumCfg = G1CurriculumCfg()
    seed: int = 42
    height_only: bool = False
    """Use a single channel for CNN; AME receives the structured XYZ grid."""

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 10.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        self.scene.left_foot_height_scanner.update_period = self.decimation * self.sim.dt
        self.scene.right_foot_height_scanner.update_period = self.decimation * self.sim.dt
        self.scene.contact_forces.update_period = self.sim.dt
        self.observations.elevation_map.map.params["height_only"] = self.height_only
        self.observations.elevation_map.map.mirror_params["height_only"] = self.height_only
        self.observations.elevation_map.concatenate_dim = 0 if self.height_only else -1
        self.sim.visualizer_cfgs = [
            NewtonGLVisualizerCfg(eye=(12.0, 0.0, 6.0), headless=True),
            # NewtonRTXVisualizerCfg(eye=(12.0, 0.0, 6.0), headless=True),
        ]
        self.video_recorders = [
            VideoRecorderCfg(source="visualizer:newton_gl", output_dir=None, video_length=200, video_interval=2000),
        ]

    def play_mode(self):
        super().play_mode()
        self.episode_length_s = 4.0
        self.scene.num_envs = min(self.scene.num_envs, 4)
        # self.curriculum.terrain_levels = None
        self.scene.terrain.max_init_terrain_level = 5
        self.observations.policy.enable_corruption = False
        self.sim.visualizer_cfgs = [
            # NewtonGLVisualizerCfg(
            #     eye=(0.0, -4.0, 1.0),
            #     origin_type="asset",
            #     origin_track_path="robot",
            #     # headless=True,
            # ),
            NewtonRTXVisualizerCfg(
                eye=(0.0, -2.0, 1.5),
                # origin_type="asset",
                # origin_track_path="robot",
                # headless=True,
            ),
        ]
        self.video_recorders = [
            # VideoRecorderCfg(source="visualizer:newton_rtx", output_dir=None, video_length=200, video_interval=2000),
        ]


def cnn_env_cfg() -> G1RoughPerceptionReachEnvCfg:
    """Construct the standalone task with height-only CNN observations."""
    return G1RoughPerceptionReachEnvCfg(height_only=True)


def ame_play_env_cfg() -> G1RoughPerceptionReachEnvCfg:
    """Construct AME playback with a fixed terrain level throughout each run."""
    cfg = G1RoughPerceptionReachEnvCfg()
    cfg.play_mode()
    return cfg


def cnn_play_env_cfg() -> G1RoughPerceptionReachEnvCfg:
    """Construct CNN playback using the same height layout as training."""
    cfg = G1RoughPerceptionReachEnvCfg(height_only=True)
    cfg.play_mode()
    return cfg
