# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


# Adapted from Agile. Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from .env_cfg import (
    G1ActionsCfg,
    G1CommandsCfg,
    G1CurriculumCfg,
    G1EventCfg,
    G1ObservationsCfg,
    G1PhysicsCfg,
    G1RewardsRoughCfg,
    G1SceneCfg,
    G1TerminationsCfg,
)


@configclass
class G1RoughEnvCfg(ManagerBasedRLEnvCfg):
    """Agile G1VelocityHistoryNewtonFlashSACEnvCfg with compositional observations."""

    sim: SimulationCfg = SimulationCfg(physics=G1PhysicsCfg(), use_newton_actuators=False)
    scene: G1SceneCfg = G1SceneCfg(num_envs=2048, env_spacing=2.5)
    actions: G1ActionsCfg = G1ActionsCfg()
    observations: G1ObservationsCfg = G1ObservationsCfg()
    rewards: G1RewardsRoughCfg = G1RewardsRoughCfg()
    commands: G1CommandsCfg = G1CommandsCfg()
    events: G1EventCfg = G1EventCfg()
    terminations: G1TerminationsCfg = G1TerminationsCfg()
    curriculum: G1CurriculumCfg = G1CurriculumCfg()
    seed: int = 1

    def __post_init__(self):
        self.decimation = 4
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.episode_length_s = 30.0
        self.sim.physics_material = self.scene.terrain.physics_material
        self.scene.terrain.terrain_generator.curriculum = True
        for sensor in (
            self.scene.contact_forces,
            self.scene.height_measurement_sensor,
            self.scene.height_measurement_sensor_left_foot,
            self.scene.height_measurement_sensor_right_foot,
        ):
            sensor.update_period = self.sim.dt
        self.video_recorders = []


@configclass
class G1RoughEnvCfg_PLAY(G1RoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
