# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Manager-based G1 locomotion on rigid rough terrain."""

from isaaclab_visualizers.newton import NewtonGLVisualizerCfg, NewtonRTXVisualizerCfg

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from .env_cfg.action_cfg import G1RoughActionsCfg
from .env_cfg.commands_cfg import G1RoughCommandsCfg
from .env_cfg.curriculum_cfg import G1RoughCurriculumCfg, G1RoughCurriculumCfg_PLAY
from .env_cfg.event_cfg import G1RoughEventCfg, G1RoughEventCfg_PLAY
from .env_cfg.observation_cfg import G1RoughObservationsCfg, G1RoughObservationsCfg_PLAY
from .env_cfg.physics_cfg import G1RoughPhysicsCfg
from .env_cfg.reward_cfg import G1RewardsRoughCfg
from .env_cfg.scene_cfg import G1RoughSceneCfg
from .env_cfg.termination_cfg import G1RoughTerminationsCfg


@configclass
class G1RoughEnvCfg(ManagerBasedRLEnvCfg):
    """Rigid-contact locomotion using task-local component configurations."""

    scene: G1RoughSceneCfg = G1RoughSceneCfg(num_envs=1280, env_spacing=4.0)
    sim: SimulationCfg = SimulationCfg(
        dt=0.005,
        render_interval=4,
        use_newton_actuators=True,
        physics=G1RoughPhysicsCfg().newton_mjwarp,
        visualizer_cfgs=[
            NewtonGLVisualizerCfg(
                eye=(3.0, 3.0, 2.0),
                origin_type="asset",
                origin_track_path="robot",
                headless=True,
            ),
            NewtonRTXVisualizerCfg(
                eye=(3.0, 3.0, 2.0),
                origin_type="asset",
                origin_track_path="robot",
                headless=True,
            ),
        ],
    )
    observations: G1RoughObservationsCfg = G1RoughObservationsCfg()
    actions: G1RoughActionsCfg = G1RoughActionsCfg()
    commands: G1RoughCommandsCfg = G1RoughCommandsCfg()
    rewards: G1RewardsRoughCfg = G1RewardsRoughCfg()
    terminations: G1RoughTerminationsCfg = G1RoughTerminationsCfg()
    events: G1RoughEventCfg = G1RoughEventCfg()
    curriculum: G1RoughCurriculumCfg = G1RoughCurriculumCfg()
    decimation: int = 4
    episode_length_s: float = 20.0
    is_finite_horizon: bool = False
    seed: int = 42
    foot_contact_force_threshold: float = 5.0
    """Force magnitude above which a foot counts as in contact [N]."""
    video_recorders = [
        VideoRecorderCfg(source="visualizer:newton_gl", output_dir=None, video_length=200, video_interval=2000),
    ]

    def __post_init__(self) -> None:
        self.scene.contact_forces.update_period = self.sim.dt

    def validate_config(self) -> None:
        """Enable terrain-row progression after command-line overrides."""
        self.scene.terrain.terrain_generator.curriculum = self.curriculum.terrain_levels is not None
        self.scene.terrain.use_terrain_origins = True


@configclass
class G1RoughEnvCfg_PLAY(G1RoughEnvCfg):
    """Playback with the same rough terrain and uncorrupted policy observations."""

    scene: G1RoughSceneCfg = G1RoughSceneCfg(num_envs=1, env_spacing=4.0)
    observations: G1RoughObservationsCfg_PLAY = G1RoughObservationsCfg_PLAY()
    events: G1RoughEventCfg_PLAY = G1RoughEventCfg_PLAY()
    curriculum: G1RoughCurriculumCfg_PLAY = G1RoughCurriculumCfg_PLAY()
    video_recorders = []

    def __post_init__(self) -> None:
        super().__post_init__()
        for visualizer in self.sim.visualizer_cfgs:
            visualizer.headless = False
