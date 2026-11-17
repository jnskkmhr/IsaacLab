# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pinned upper-body object-pushing evaluation with a frozen reaching policy."""

from isaaclab_visualizers.newton import NewtonGLVisualizerCfg, NewtonRTXVisualizerCfg

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from .env_cfg.action_cfg import ActionsCfg
from .env_cfg.commands_cfg import CommandsCfg
from .env_cfg.event_cfg import EventsCfg
from .env_cfg.observation_cfg import ObservationsCfg
from .env_cfg.physics_cfg import G1PhysicsCfg
from .env_cfg.scene_cfg import SceneCfg
from .env_cfg.termination_cfg import TerminationsCfg


@configclass
class G1ManipulationObjectEnvCfg(ManagerBasedRLEnvCfg):
    scene = SceneCfg(num_envs=1, env_spacing=3.0)
    sim = SimulationCfg(dt=0.005, render_interval=4, use_newton_actuators=True, physics=G1PhysicsCfg())
    decimation = 4
    episode_length_s = 10.0
    seed = 42
    observations = ObservationsCfg()
    actions = ActionsCfg()
    commands = CommandsCfg()
    rewards = {}
    events = EventsCfg()
    curriculum = {}
    terminations = TerminationsCfg()

    def __post_init__(self):
        self.sim.visualizer_cfgs = [
            NewtonGLVisualizerCfg(
                eye=(1.5, -1.8, 0.8), lookat=(0.3, 0.0, 0.1), origin_type="asset", origin_track_path="robot"
            ),
            NewtonRTXVisualizerCfg(
                eye=(1.5, -1.8, 0.8),
                lookat=(0.3, 0.0, 0.1),
                origin_type="asset",
                origin_track_path="robot",
                streaming_view=False,
            ),
        ]
        self.video_recorders = []
        self.observations.policy.enable_corruption = False


@configclass
class G1ManipulationObjectEnvCfg_PLAY(G1ManipulationObjectEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.observations.policy.enable_corruption = False
