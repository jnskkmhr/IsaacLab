# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for G1 29-DoF locomotion over a coupled MJWarp/MPM granular bed."""

from __future__ import annotations

import math

from isaaclab_visualizers.kit import KitVisualizerCfg
from isaaclab_visualizers.newton import NewtonGLVisualizerCfg, NewtonRTXVisualizerCfg

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass

from .env_cfg import (
    G1ActionsCfg,
    G1CommandsCfg,
    G1CurriculumCfg,
    G1EventCfg,
    G1MPMSceneCfg,
    G1ObservationsCfg,
    G1RewardsCfg,
    G1TerminationsCfg,
)
from .env_cfg.physics_cfg import DEFAULT_PROXY_MASS_SCALE, configure_sparse_mpm_capacities, g1_mpm_physics_cfg
from .env_cfg.scene_cfg import MPM_VISUAL_COLOR

# VISUALIZER = "newton_gl"
VISUALIZER = "newton_rtx"
# VISUALIZER = "kit"


@configclass
class G1MPMEnvCfg(ManagerBasedRLEnvCfg):
    """Velocity-tracking locomotion of the G1 on granular media.

    The task mirrors the flat soft-contact task; the analytical contact model is replaced by an
    implicit MPM bed coupled to MJWarp through lagged foot proxies.
    """

    # scene
    scene: G1MPMSceneCfg = G1MPMSceneCfg(num_envs=32, env_spacing=4.0, replicate_physics=True, clone_in_fabric=True)

    # basic settings
    observations: G1ObservationsCfg = G1ObservationsCfg()
    actions: G1ActionsCfg = G1ActionsCfg()
    commands: G1CommandsCfg = G1CommandsCfg()

    # mdp settings
    rewards: G1RewardsCfg = G1RewardsCfg()
    terminations: G1TerminationsCfg = G1TerminationsCfg()
    events: G1EventCfg = G1EventCfg()
    curriculum: G1CurriculumCfg = G1CurriculumCfg()

    seed: int = 42

    # -- granular terrain state
    foot_body_expr: str = ".*_ankle_roll_link"
    """Body-name expression selecting the feet tracked against the bed."""

    foot_contact_force_threshold: float = 5.0
    """Granular reaction magnitude above which a foot counts as in contact [N]."""

    reset_particle_jitter: float = 0.0
    """Half-width of the uniform position jitter applied when the bed is reset [m]."""

    particle_max_velocity: float = 20.0
    """Speed limit enforced on the particles by the Newton model [m/s]."""

    # -- solver capacities, scaled with the world count before the simulation is created
    proxy_mass_scale: float = DEFAULT_PROXY_MASS_SCALE
    mpm_active_cell_count_per_world: int = 1 << 16
    mpm_leaf_node_count_per_world: int = 1 << 13
    mpm_lower_node_count_per_world: int = 1 << 10
    mpm_upper_node_count_per_world: int = 16

    def __post_init__(self) -> None:
        # general settings
        self.decimation = 4  # 50 Hz control
        self.episode_length_s = 10.0
        self.is_finite_horizon = False

        # simulation settings
        self.sim = SimulationCfg(dt=1 / 200, render_interval=self.decimation)  # 200 Hz physics
        self.sim.physics = g1_mpm_physics_cfg(self.proxy_mass_scale)
        self.sim.use_newton_actuators = True

        # The bed is a few metres across, so the high-speed stages of the cloned command
        # curriculum would drive the robot off it within an episode.
        self.curriculum.command_vel = None  # type: ignore
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.resampling_time_range = (5.0, 5.0)

        if VISUALIZER == "newton_gl":
            self.sim.visualizer_cfgs = [
                NewtonGLVisualizerCfg(
                    eye=(0.0, -6.0, 1.5), headless=True, show_particles=True, particle_color=MPM_VISUAL_COLOR
                ),
            ]

            self.video_recorders = [
                VideoRecorderCfg(source="visualizer:newton_gl", output_dir="videos/"),
            ]

        elif VISUALIZER == "newton_rtx":
            self.sim.visualizer_cfgs = [
                NewtonRTXVisualizerCfg(
                    eye=(0.0, -6.0, 1.5), headless=True, show_particles=True, particle_color=MPM_VISUAL_COLOR
                ),
            ]

            # self.video_recorders = [
            #     VideoRecorderCfg(source="visualizer:newton_rtx", output_dir="videos/"),
            # ]

        elif VISUALIZER == "kit":
            self.sim.visualizer_cfgs = [
                KitVisualizerCfg(eye=(0.0, -6.0, 1.5), headless=True),
            ]

            self.video_recorders = [
                VideoRecorderCfg(source="visualizer:kit", output_dir="videos/"),
            ]

        configure_sparse_mpm_capacities(self)


@configclass
class G1MPMEnvCfg_PLAY(G1MPMEnvCfg):
    """Small playback scene with randomization disabled."""

    def __post_init__(self) -> None:
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 4

        # disable curriculum and observation corruption
        self.curriculum.track_lin_vel = None  # type: ignore
        self.curriculum.track_ang_vel = None  # type: ignore
        self.curriculum.track_heading = None  # type: ignore
        self.curriculum.track_lin_vel_weight = None  # type: ignore
        self.observations.policy.enable_corruption = False

        # remove random events
        self.events.add_base_mass = None  # type: ignore
        self.events.push_robot = None  # type: ignore
        self.events.physics_material = None  # type: ignore
        self.events.scale_actuator_gains = None  # type: ignore
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-math.pi, math.pi)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }
        self.events.reset_robot_joints.params["position_range"] = (0.0, 0.0)

        # commands
        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.resampling_time_range = (self.episode_length_s, self.episode_length_s)

        if VISUALIZER == "newton_gl":
            self.sim.visualizer_cfgs = [
                NewtonGLVisualizerCfg(eye=(0.0, -6.0, 1.5), show_particles=True, particle_color=MPM_VISUAL_COLOR),
            ]

            self.video_recorders = [
                VideoRecorderCfg(source="visualizer:newton_gl", output_dir="videos/"),
            ]

        elif VISUALIZER == "newton_rtx":
            self.sim.visualizer_cfgs = [
                NewtonRTXVisualizerCfg(eye=(0.0, -6.0, 1.5), show_particles=True, particle_color=MPM_VISUAL_COLOR),
            ]

            self.video_recorders = [
                VideoRecorderCfg(source="visualizer:newton_rtx", output_dir="videos/"),
            ]

        elif VISUALIZER == "kit":
            self.sim.visualizer_cfgs = [
                KitVisualizerCfg(eye=(0.0, -6.0, 1.5)),
            ]

            self.video_recorders = [
                VideoRecorderCfg(source="visualizer:kit", output_dir="videos/"),
            ]

        configure_sparse_mpm_capacities(self)
