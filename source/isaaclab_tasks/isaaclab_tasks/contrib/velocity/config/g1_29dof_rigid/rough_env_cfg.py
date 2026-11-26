# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg
from isaaclab.sim import RigidBodyMaterialCfg, SimulationCfg
from isaaclab.utils import configclass
from isaaclab.visualizers import VisualizerCfg
from isaaclab.visualizers.visualizer_cfg import WindowCfg

from .env_cfg.action_cfg import G1ActionsCfg
from .env_cfg.commands_cfg import G1CommandsCfg
from .env_cfg.curriculum_cfg import G1RoughCurriculumCfg
from .env_cfg.event_cfg import G1RoughEventCfg
from .env_cfg.observation_cfg import G1RoughObservationsCfg
from .env_cfg.physics_cfg import G1PhysicsCfg
from .env_cfg.reward_cfg import G1RoughRewardsCfg
from .env_cfg.scene_cfg import G1RoughSceneCfg
from .env_cfg.termination_cfg import G1TerminationsCfg


@configclass
class G1RoughEnvCfg(ManagerBasedRLEnvCfg):
    scene = G1RoughSceneCfg(num_envs=4096, env_spacing=4.0)
    actions = G1ActionsCfg()
    observations = G1RoughObservationsCfg()
    commands = G1CommandsCfg()
    events = G1RoughEventCfg()
    rewards = G1RoughRewardsCfg()
    curriculum = G1RoughCurriculumCfg()
    terminations = G1TerminationsCfg()
    video_recorders = [VideoRecorderCfg(source="visualizer:newton_gl", video_length=200, video_interval=2000)]
    seed = 42
    decimation = 4
    episode_length_s = 20.0
    sim = SimulationCfg(
        dt=0.005,
        render_interval=4,
        physics=G1PhysicsCfg(),
        visualizer_cfgs=[
            # One shared view; the backend is chosen on the command line (``--viz newton_gl`` or ``newton_rtx``).
            VisualizerCfg(
                headless=True,
                window=WindowCfg(size=(1280, 720), fps=30.0),
                focal_length=12.0,
                streaming_view=False,
                eye=(4.0, -4.0, 3.0),
                lookat=(0.0, 0.0, 0.0),
            ),
        ],
        physics_material=RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
    )


@configclass
class G1RoughEnvCfg_PLAY(G1RoughEnvCfg):
    """Interactive playback retaining the training observation and terrain definitions."""

    video_recorders = []

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        for visualizer in self.sim.visualizer_cfgs:
            visualizer.headless = False
