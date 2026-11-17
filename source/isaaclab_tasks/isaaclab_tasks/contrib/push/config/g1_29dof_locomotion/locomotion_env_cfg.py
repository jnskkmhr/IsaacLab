# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Standalone G1 locomotion task."""

from isaaclab_visualizers.newton import NewtonGLVisualizerCfg, NewtonRTXVisualizerCfg

import isaaclab.envs.mdp as mdp
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from .env_cfg.action_cfg import ActionsCfg
from .env_cfg.commands_cfg import CommandsCfg
from .env_cfg.curriculum_cfg import CurriculumCfg
from .env_cfg.event_cfg import EventsCfg
from .env_cfg.observation_cfg import ObservationsCfg
from .env_cfg.physics_cfg import G1PhysicsCfg
from .env_cfg.reward_cfg import RewardsCfg
from .env_cfg.scene_cfg import SceneCfg
from .env_cfg.termination_cfg import TerminationsCfg


@configclass
class G1LocomotionEnvCfg(ManagerBasedRLEnvCfg):
    scene = SceneCfg(num_envs=1024, env_spacing=3.0)
    sim = SimulationCfg(dt=0.005, render_interval=4, use_newton_actuators=True, physics=G1PhysicsCfg())
    decimation = 4
    episode_length_s = 20.0
    seed = 42
    observations = ObservationsCfg()
    actions = ActionsCfg()
    commands = CommandsCfg()
    rewards = RewardsCfg()
    events = EventsCfg()
    curriculum = CurriculumCfg()
    terminations = TerminationsCfg()

    def __post_init__(self):
        self.sim.visualizer_cfgs = [
            NewtonGLVisualizerCfg(
                eye=(2.0, -2.0, 1.0), lookat=(0.0, 0.0, 0.0), origin_type="asset", origin_track_path="robot"
            ),
            NewtonRTXVisualizerCfg(
                eye=(2.0, -2.0, 1.0),
                lookat=(0.0, 0.0, 0.0),
                origin_type="asset",
                origin_track_path="robot",
                streaming_view=False,
            ),
        ]
        self.video_recorders = []


@configclass
class G1LocomotionEnvCfg_PLAY(G1LocomotionEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.observations.policy.enable_corruption = False
        self.observations.policy.projected_gravity.func = mdp.projected_gravity
        self.observations.policy.projected_gravity.params = {}
        self.curriculum.gravity_roll_bias = None
        self.curriculum.gravity_pitch_bias = None
        self.events.push_robot = None
        self.events.gains = None
