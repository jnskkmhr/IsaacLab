# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Manager-based Franka IK pushes a closed gripper through native MPM sand."""

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from .env_cfg.action_cfg import FrankaActionsCfg
from .env_cfg.event_cfg import RobotEventsCfg
from .env_cfg.observation_cfg import RobotObservationsCfg
from .env_cfg.physics_cfg import franka_physics_cfg
from .env_cfg.scene_cfg import FrankaSceneCfg


@configclass
class FrankaEnvCfg(ManagerBasedRLEnvCfg):
    scene = FrankaSceneCfg(num_envs=1, env_spacing=4.0)
    actions = FrankaActionsCfg()
    observations = RobotObservationsCfg()
    events = RobotEventsCfg()
    decimation = 1
    seed = 42
    episode_length_s = 6.0
    rewards = {}
    terminations = {}
    sim = SimulationCfg(dt=0.005, render_interval=4, physics=franka_physics_cfg(), use_newton_actuators=True)
