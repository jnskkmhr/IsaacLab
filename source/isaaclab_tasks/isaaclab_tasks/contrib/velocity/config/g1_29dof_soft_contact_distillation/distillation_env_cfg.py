# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_physx.sim.spawners.materials import PhysxRigidBodyMaterialCfg
from isaaclab_visualizers.kit import KitVisualizerCfg
from isaaclab_visualizers.newton import NewtonGLVisualizerCfg, NewtonRTXVisualizerCfg

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from .env_cfg.action_cfg import G1ActionsCfg
from .env_cfg.commands_cfg import G1CommandsCfg
from .env_cfg.curriculum_cfg import G1CurriculumCfg
from .env_cfg.event_cfg import G1EventCfg
from .env_cfg.observation_cfg import G1StudentObservationsCfg, G1TeacherObservationsCfg
from .env_cfg.physics_cfg import G1NewtonPhysicsCfg, G1PhysicsCfg
from .env_cfg.reward_cfg import G1RewardsCfg
from .env_cfg.scene_cfg import G1SceneCfg
from .env_cfg.terminations_cfg import G1TerminationsCfg


@configclass
class G1TeacherEnvCfg(ManagerBasedRLEnvCfg):
    scene = G1SceneCfg(num_envs=4096, env_spacing=2.5)
    actions = G1ActionsCfg()
    observations = G1TeacherObservationsCfg()
    commands = G1CommandsCfg()
    events = G1EventCfg()
    rewards = G1RewardsCfg()
    curriculum = G1CurriculumCfg()
    terminations = G1TerminationsCfg()
    video_recorders = [VideoRecorderCfg(source="visualizer:kit", video_length=200, video_interval=2000)]
    decimation = 4
    episode_length_s = 20.0
    sim = SimulationCfg(
        dt=0.005,
        render_interval=4,
        physics=G1PhysicsCfg(),
        visualizer_cfgs=[
            KitVisualizerCfg(
                headless=True,
                eye=(4.0, -4.0, 3.0),
                lookat=(0.0, 0.0, 0.0),
                origin_type="asset",
                origin_track_path="robot",
            )
        ],
        physics_material=PhysxRigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
        ),
    )


@configclass
class G1StudentEnvCfg(G1TeacherEnvCfg):
    observations = G1StudentObservationsCfg()


@configclass
class G1NewtonTeacherEnvCfg(G1TeacherEnvCfg):
    sim = SimulationCfg(
        dt=0.005,
        render_interval=4,
        physics=G1NewtonPhysicsCfg(),
        physics_material=PhysxRigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
        ),
        visualizer_cfgs=[
            NewtonGLVisualizerCfg(
                headless=True,
                eye=(4.0, -4.0, 3.0),
                lookat=(0.0, 0.0, 0.0),
                origin_type="asset",
                origin_track_path="robot",
                streaming_view=False,
            ),
            NewtonRTXVisualizerCfg(
                headless=True,
                eye=(4.0, -4.0, 3.0),
                lookat=(0.0, 0.0, 0.0),
                origin_type="asset",
                origin_track_path="robot",
                streaming_view=False,
            ),
        ],
    )
    video_recorders = [VideoRecorderCfg(source="visualizer:newton_gl", video_length=200, video_interval=2000)]


@configclass
class G1NewtonStudentEnvCfg(G1NewtonTeacherEnvCfg):
    observations = G1StudentObservationsCfg()
