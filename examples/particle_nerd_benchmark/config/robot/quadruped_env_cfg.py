# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Manager-based quadruped crawl with a free base and native sand reaction."""

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from particle_nerd_benchmark.core.physics import physics_cfg

from .env_cfg.event_cfg import RobotEventsCfg
from .env_cfg.observation_cfg import RobotObservationsCfg
from .env_cfg.quadruped_action_cfg import QuadrupedActionsCfg
from .env_cfg.quadruped_scene_cfg import QuadrupedSceneCfg


@configclass
class QuadrupedEnvCfg(ManagerBasedRLEnvCfg):
    scene = QuadrupedSceneCfg(num_envs=1, env_spacing=4.0)
    actions = QuadrupedActionsCfg()
    observations = RobotObservationsCfg()
    events = RobotEventsCfg()
    decimation = 1
    seed = 42
    episode_length_s = 6.0
    rewards = {}
    terminations = {}
    sim = SimulationCfg(
        dt=0.005,
        render_interval=4,
        use_newton_actuators=True,
        physics=physics_cfg([r"/World/envs/env_.*/Robot"], [r"/World/envs/env_.*/Robot/.*foot"]),
    )

    def __post_init__(self):
        super().__post_init__()
        self.sim.physics.default_shape_cfg.margin = 0.045
