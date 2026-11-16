# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Standalone manager-based native-MPM scene for closed-loop G1 policy playback."""

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from particle_nerd_benchmark.core.physics import physics_cfg

from .env_cfg.event_cfg import RobotEventsCfg
from .env_cfg.g1_action_cfg import G1ActionsCfg
from .env_cfg.g1_observation_cfg import G1ObservationsCfg
from .env_cfg.g1_scene_cfg import G1SceneCfg


@configclass
class G1EnvCfg(ManagerBasedRLEnvCfg):
    scene = G1SceneCfg(num_envs=1, env_spacing=5.0)
    actions = G1ActionsCfg()
    observations = G1ObservationsCfg()
    events = RobotEventsCfg()
    decimation = 4
    seed = 42
    episode_length_s = 6.0
    rewards = {}
    terminations = {}
    forward_speed = 0.35
    sim = SimulationCfg(
        dt=0.005,
        render_interval=4,
        use_newton_actuators=True,
        physics=physics_cfg([r"/World/envs/env_.*/Robot"], [r"/World/envs/env_.*/Robot/.*ankle_roll_link"]),
    )
