# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Whole-body box pushing through unconfined sand on a large flat ground plane."""

from isaaclab_visualizers.newton import NewtonGLVisualizerCfg, NewtonRTXVisualizerCfg

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from isaaclab_tasks.contrib.velocity_mpm.g1_29dof_mpm.env_cfg.physics_cfg import configure_sparse_mpm_capacities

from .env_cfg.action_cfg import ActionsCfg
from .env_cfg.commands_cfg import CommandsCfg
from .env_cfg.curriculum_cfg import CurriculumCfg
from .env_cfg.event_cfg import EventsCfg
from .env_cfg.observation_cfg import ObservationsCfg
from .env_cfg.physics_cfg import G1PhysicsProxyCfg
from .env_cfg.reward_cfg import RewardsCfg
from .env_cfg.scene_cfg import SceneCfg
from .env_cfg.termination_cfg import TerminationsCfg


@configclass
class G1ObjectPushEnvCfg(ManagerBasedRLEnvCfg):
    scene = SceneCfg(num_envs=4, env_spacing=5.0)
    sim = SimulationCfg(dt=0.005, render_interval=4, use_newton_actuators=True, physics=G1PhysicsProxyCfg())
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

    push_object_name: str = "object"

    foot_body_expr: str = ".*_ankle_roll_link"
    foot_contact_force_threshold: float = 5.0
    reset_particle_jitter: float = 0.0
    particle_max_velocity: float = 20.0
    mpm_active_cell_count_per_world: int | None = None
    mpm_leaf_node_count_per_world: int = 8192
    mpm_lower_node_count_per_world: int = 1024
    mpm_upper_node_count_per_world: int = 16

    def validate_config(self):
        configure_sparse_mpm_capacities(self)

    def __post_init__(self):
        self.sim.visualizer_cfgs = [
            NewtonGLVisualizerCfg(
                eye=(-2.0, -0.5, 1.0),
                # show_particles=False,
                lookat=(0.0, 0.0, 0.0),
                origin_type="asset",
                origin_track_path="robot",
                streaming_view=False,
            ),
            NewtonRTXVisualizerCfg(
                eye=(-0.0, -2.0, 1.0),
                lookat=(0.0, 0.0, 0.0),
                origin_type="asset",
                origin_track_path="robot",
                streaming_view=False,
            ),
        ]
        self.video_recorders = [
            VideoRecorderCfg(source="visualizer:newton_gl", output_dir=None, video_length=200, video_interval=2000),
        ]


@configclass
class G1ObjectPushEnvCfg_PLAY(G1ObjectPushEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.observations.policy.enable_corruption = False

        self.commands.object_velocity.ranges.lin_vel_x = (0.4, 0.4)
        self.commands.object_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.object_velocity.rel_standing_envs = 0.0
        self.commands.object_velocity.debug_vis = True

        self.sim.visualizer_cfgs = [
            NewtonGLVisualizerCfg(
                eye=(-2.0, -0.5, 1.0),
                lookat=(0.0, 0.0, 0.0),
                origin_type="asset",
                origin_track_path="robot",
                streaming_view=False,
            ),
            NewtonRTXVisualizerCfg(
                eye=(-0.0, -2.0, 1.0),
                lookat=(0.0, 0.0, 0.0),
                origin_type="asset",
                origin_track_path="robot",
                streaming_view=False,
            ),
        ]
        self.video_recorders = [
            # VideoRecorderCfg(source="visualizer:newton_gl", output_dir=None, video_length=200, video_interval=2000),
        ]
