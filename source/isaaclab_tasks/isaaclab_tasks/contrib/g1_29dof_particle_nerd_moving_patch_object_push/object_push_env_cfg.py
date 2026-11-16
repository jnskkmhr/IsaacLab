# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Standalone direction-conditioned pushing on a non-cached moving particle patch."""

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from isaaclab_tasks.contrib.g1_29dof_particle_nerd_moving_patch.env_cfg.visualizer_cfg import (
    MovingPatchGLVisualizerCfg,
    MovingPatchRTXVisualizerCfg,
)

from .env_cfg.action_cfg import ActionsCfg
from .env_cfg.commands_cfg import CommandsCfg
from .env_cfg.curriculum_cfg import CurriculumCfg
from .env_cfg.event_cfg import EventsCfg
from .env_cfg.observation_cfg import ObservationsCfg
from .env_cfg.physics_cfg import PhysicsCfg
from .env_cfg.reward_cfg import RewardsCfg
from .env_cfg.scene_cfg import SceneCfg
from .env_cfg.termination_cfg import TerminationsCfg


@configclass
class ObjectPushEnvCfg(ManagerBasedRLEnvCfg):
    scene = SceneCfg(num_envs=4, env_spacing=6.0)
    sim = SimulationCfg(dt=0.005, render_interval=4, use_newton_actuators=True, physics=PhysicsCfg())
    decimation = 4
    episode_length_s = 20.0
    seed = 42
    observations = ObservationsCfg()
    actions = ActionsCfg()
    commands = CommandsCfg()
    events = EventsCfg()
    rewards = RewardsCfg()
    terminations = TerminationsCfg()
    curriculum = CurriculumCfg()
    push_object_name = "object"
    foot_body_expr = ".*_ankle_roll_link"
    foot_contact_force_threshold = 5.0
    foot_contact_margin_range = None
    reset_particle_jitter = 0.0
    particle_max_velocity = 20.0
    proxy_mass_scale = 1.0

    def validate_config(self):
        self.scene.configure_terrain()
        self.scene.sand.spawn.visible = False
        self.sim.physics.configure_terrain(self.scene.terrain.moving_patch_terrain, self.proxy_mass_scale)
        if self.sim.physics.use_cuda_graph:
            raise ValueError("Neural inference requires use_cuda_graph=False; use capture_solver instead")

    def __post_init__(self):
        self.sim.visualizer_cfgs = [
            MovingPatchGLVisualizerCfg(
                eye=(-2.0, -1.5, 1.0),
                lookat=(0, 0, 0),
                origin_type="asset",
                origin_track_path="robot",
                streaming_view=False,
                show_particles=True,
            ),
            MovingPatchRTXVisualizerCfg(
                eye=(0, -2.0, 1.0),
                lookat=(0, 0, 0),
                origin_type="asset",
                origin_track_path="robot",
                streaming_view=False,
                show_particles=True,
            ),
        ]
        self.video_recorders = [VideoRecorderCfg(source="visualizer:newton_gl", video_length=400, video_interval=2000)]


@configclass
class ObjectPushEnvCfg_PLAY(ObjectPushEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.commands.object_direction.angle_range = (0.0, 0.0)
        self.commands.object_direction.standing_probability = 0.0
        self.video_recorders = []
