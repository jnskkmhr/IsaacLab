# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Manager-based object/MPM benchmark; one action per physics step."""

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from .env_cfg.action_cfg import ActionsCfg
from .env_cfg.event_cfg import EventsCfg
from .env_cfg.observation_cfg import ObservationsCfg
from .env_cfg.physics_cfg import object_physics_cfg
from .env_cfg.scene_cfg import SceneCfg


@configclass
class ObjectEnvCfg(ManagerBasedRLEnvCfg):
    scene = SceneCfg(num_envs=1, env_spacing=4.0)
    actions = ActionsCfg()
    observations = ObservationsCfg()
    events = EventsCfg()
    decimation = 1
    seed = 42
    episode_length_s = 6.0
    rewards = {}
    terminations = {}
    sim = SimulationCfg(dt=0.005, render_interval=4, physics=object_physics_cfg())
