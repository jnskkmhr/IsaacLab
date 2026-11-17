# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for G1 29-DoF locomotion over a coupled MJWarp/MPM granular bed."""

from __future__ import annotations

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass
from isaaclab.visualizers import VisualizerCfg

from isaaclab_tasks.utils.presets import MultiBackendVisualizerCfg

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
from .env_cfg.moving_patch_cfg import MovingPatchCfg
from .env_cfg.physics_cfg import DEFAULT_PROXY_MASS_SCALE, configure_sparse_mpm_capacities, g1_mpm_physics_cfg
from .env_cfg.scene_cfg import (
    MPM_VISUAL_COLOR,
    G1DescendingMPMSceneCfg,
    G1DuneMPMSceneCfg,
    G1GeneratedMPMSceneCfg,
    G1RoughMPMSceneCfg,
)
from .env_cfg.surface_cfg import SandSurfaceCfg
from .env_cfg.visualizer_cfg import SandGLBackendCfg, SandRTXBackendCfg
from .util.terrain import TerrainMPMGridCfg

# VISUALIZER = "newton_gl"
VISUALIZER = "newton_rtx"
# VISUALIZER = "kit"


@configclass
class G1MovingPatchEnvCfg(ManagerBasedRLEnvCfg):
    """Velocity-tracking locomotion of the G1 on granular media.

    The task mirrors the flat soft-contact task; the analytical contact model is replaced by an
    implicit MPM bed coupled to MJWarp through lagged foot proxies.
    """

    # scene
    scene: G1MPMSceneCfg = G1MPMSceneCfg(num_envs=32, env_spacing=8.0, replicate_physics=True)

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

    moving_patch: MovingPatchCfg = MovingPatchCfg()
    sand_surface: SandSurfaceCfg = SandSurfaceCfg()

    mpm_sleeping_enabled: bool = True
    """Enable native GPU activity selection; required by the moving-region task."""

    mpm_cache_initial_state: bool = True
    """Settle the actual bed once and restore its GPU snapshot on subsequent resets."""

    mpm_settle_time: float = 1.0
    """Initial sand-only simulation duration [s]; remaining moving particles stay awake.

    The initial material snapshot is approximate after material randomization. Geometry changes
    require a new environment/snapshot. Set ``mpm_cache_initial_state=False`` for raw spawn resets.
    """

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
    mpm_active_cell_count_per_world: int | None = None
    """Active-cell capacity per world; None derives it from the sand particle grid."""
    mpm_leaf_node_count_per_world: int = 1 << 13
    mpm_lower_node_count_per_world: int = 1 << 10
    mpm_upper_node_count_per_world: int = 16

    def __post_init__(self) -> None:
        # general settings
        self.decimation = 4  # 50 Hz control
        # The approach has to be crossed before the granular part of the episode even begins.
        self.episode_length_s = 13.0
        self.is_finite_horizon = False

        # simulation settings
        self.sim = SimulationCfg(dt=1 / 200, render_interval=self.decimation)  # 200 Hz physics
        self.sim.physics = g1_mpm_physics_cfg(
            self.proxy_mass_scale, rough_terrain=isinstance(self.scene.sand.spawn, TerrainMPMGridCfg)
        )

        self.sim.visualizer_cfgs = [
            VisualizerCfg(
                backend=MultiBackendVisualizerCfg(
                    newton_gl=SandGLBackendCfg(particle_color=MPM_VISUAL_COLOR), newton_rtx=SandRTXBackendCfg()
                ),
                eye=(3.0, -6.0, 3.0),
                lookat=(0.0, 0.0, 0.3),
                streaming_view=False,
            )
        ]

        self.video_recorders = [
            VideoRecorderCfg(source="visualizer:newton_gl", output_dir=None, video_length=200, video_interval=2000),
            # VideoRecorderCfg(source="visualizer:newton_rtx", output_dir=None, video_length=200, video_interval=2000),
            # VideoRecorderCfg(source="visualizer:kit", output_dir=None, video_length=200, video_interval=2000),
        ]

        configure_sparse_mpm_capacities(self)


@configclass
class G1MovingPatchEnvCfg_PLAY(G1MovingPatchEnvCfg):
    """Small playback scene with randomization disabled."""

    def __post_init__(self) -> None:
        super().__post_init__()

        # disable curriculum and observation corruption
        self.curriculum.track_lin_vel = None  # type: ignore
        self.curriculum.track_ang_vel = None  # type: ignore
        self.curriculum.track_heading = None  # type: ignore
        self.curriculum.track_lin_vel_weight = None  # type: ignore
        self.observations.policy.enable_corruption = False

        # remove random events
        self.events.mpm_material = None  # type: ignore
        self.events.mpm_material_log = None  # type: ignore
        self.events.add_base_mass = None  # type: ignore
        self.events.push_robot = None  # type: ignore
        self.events.physics_material = None  # type: ignore
        self.events.scale_actuator_gains = None  # type: ignore
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.25, 0.25), "y": (-0.25, 0.25), "yaw": (0.0, 0.0)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)

        # commands
        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.resampling_time_range = (self.episode_length_s, self.episode_length_s)

        self.sim.visualizer_cfgs = [
            VisualizerCfg(
                backend=MultiBackendVisualizerCfg(
                    newton_gl=SandGLBackendCfg(particle_color=MPM_VISUAL_COLOR), newton_rtx=SandRTXBackendCfg()
                ),
                eye=(3.0, -6.0, 3.0),
                lookat=(0.0, 0.0, 0.3),
                streaming_view=False,
            )
        ]

        self.video_recorders = [
            # VideoRecorderCfg(source="visualizer:newton_gl", output_dir=None, video_length=200, video_interval=2000),
            # VideoRecorderCfg(source="visualizer:newton_rtx", output_dir=None, video_length=200, video_interval=2000),
            # VideoRecorderCfg(source="visualizer:kit", output_dir=None, video_length=200, video_interval=2000),
        ]


@configclass
class G1RoughMPMEnvCfg(G1MovingPatchEnvCfg):
    """G1 walking from the approach platform onto a fixed generated sand surface.

    This geometry variant retains the base task's rewards. Height and touchdown
    rewards still reference flat ground; adapt them before rough-terrain training.
    """

    scene: G1RoughMPMSceneCfg = G1RoughMPMSceneCfg(num_envs=32, env_spacing=8.0)


@configclass
class G1RoughMPMEnvCfg_PLAY(G1MovingPatchEnvCfg_PLAY):
    """Playback with a fixed generated sand surface and no episode randomization."""

    scene: G1RoughMPMSceneCfg = G1RoughMPMSceneCfg(num_envs=1, env_spacing=8.0)

    def __post_init__(self) -> None:
        super().__post_init__()

        self.sim.visualizer_cfgs = [
            VisualizerCfg(
                backend=MultiBackendVisualizerCfg(
                    newton_gl=SandGLBackendCfg(particle_color=MPM_VISUAL_COLOR), newton_rtx=SandRTXBackendCfg()
                ),
                eye=(3.0, -6.0, 3.0),
                lookat=(0.0, 0.0, 0.3),
                streaming_view=False,
            )
        ]

        self.video_recorders = [
            # VideoRecorderCfg(source="visualizer:newton_gl", output_dir=None, video_length=200, video_interval=2000),
            VideoRecorderCfg(source="visualizer:newton_rtx", output_dir=None, video_length=200, video_interval=2000),
            # VideoRecorderCfg(source="visualizer:kit", output_dir=None, video_length=200, video_interval=2000),
        ]


@configclass
class G1DuneMovingPatchEnvCfg(G1MovingPatchEnvCfg):
    """Body-following MPM activity on a finite dune bed."""

    scene: G1DuneMPMSceneCfg = G1DuneMPMSceneCfg(num_envs=32, env_spacing=8.0)


@configclass
class G1DuneMovingPatchEnvCfg_PLAY(G1MovingPatchEnvCfg_PLAY):
    """Dune playback with cached initial material state."""

    scene: G1DuneMPMSceneCfg = G1DuneMPMSceneCfg(num_envs=1, env_spacing=8.0)


@configclass
class G1DescendingMovingPatchEnvCfg(G1MovingPatchEnvCfg):
    """Body-following MPM activity on a finite descending bed."""

    scene: G1DescendingMPMSceneCfg = G1DescendingMPMSceneCfg(num_envs=32, env_spacing=8.0)


@configclass
class G1DescendingMovingPatchEnvCfg_PLAY(G1MovingPatchEnvCfg_PLAY):
    """Descending playback with cached initial material state."""

    scene: G1DescendingMPMSceneCfg = G1DescendingMPMSceneCfg(num_envs=1, env_spacing=8.0)


@configclass
class G1GeneratedMPMEnvCfg(G1MovingPatchEnvCfg):
    """Generated terrain with cached particle paging and GPU body-following activity."""

    scene: G1GeneratedMPMSceneCfg = G1GeneratedMPMSceneCfg(num_envs=1, env_spacing=8.0)
    mpm_leaf_node_count_per_world: int = 32768


@configclass
class G1GeneratedMPMEnvCfg_PLAY(G1MovingPatchEnvCfg_PLAY):
    """Playback on generated terrain with the same policy observation interface."""

    scene: G1GeneratedMPMSceneCfg = G1GeneratedMPMSceneCfg(num_envs=1, env_spacing=8.0)
    mpm_leaf_node_count_per_world: int = 32768
