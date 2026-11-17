# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Single-policy whole-body box pushing on rigid flat ground."""

from isaaclab_visualizers.newton import NewtonGLVisualizerCfg, NewtonRTXVisualizerCfg

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from .env_cfg import (
    CommandsCfg,
    CurriculumCfg,
    EventsCfg,
    G1ActionsCfg,
    G1ObjectPushSceneCfg,
    G1PhysicsCfg,
    ObservationsCfg,
    RewardsCfg,
    TerminationsCfg,
)


@configclass
class G1ObjectPushEnvCfg(ManagerBasedRLEnvCfg):
    """Compose task-local scene, control, learning, and rigid-body physics settings."""

    scene: G1ObjectPushSceneCfg = G1ObjectPushSceneCfg(num_envs=16, env_spacing=5.0)
    sim: SimulationCfg = SimulationCfg(dt=1 / 200, render_interval=4, use_newton_actuators=True, physics=G1PhysicsCfg())
    decimation: int = 4
    is_finite_horizon: bool = False

    # mdp terms
    observations: ObservationsCfg = ObservationsCfg()
    actions: G1ActionsCfg = G1ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    seed: int = 42

    def __post_init__(self):
        self.episode_length_s = 12.0
        self.sim.render_interval = self.decimation
        self.sim.visualizer_cfgs = [
            NewtonGLVisualizerCfg(
                eye=(2.0, -3.0, 1.5),
                lookat=(0.5, 0.0, -0.2),
                origin_type="asset",
                origin_track_path="robot",
            ),
            NewtonRTXVisualizerCfg(
                eye=(2.0, -3.0, 1.5),
                lookat=(0.5, 0.0, -0.2),
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
        self.commands.base_velocity.ranges.lin_vel_x = (0.35, 0.35)
        self.commands.base_velocity.rel_standing_envs = 0.0

        self.sim.visualizer_cfgs = [
            NewtonGLVisualizerCfg(
                eye=(2.0, -3.0, 1.5),
                lookat=(0.5, 0.0, -0.2),
                origin_type="asset",
                origin_track_path="robot",
            ),
            NewtonRTXVisualizerCfg(
                eye=(2.0, -3.0, 1.5),
                lookat=(0.5, 0.0, -0.2),
                # origin_type="asset",
                # origin_track_path="robot",
                # streaming_view=False,
            ),
        ]
        self.video_recorders = []
