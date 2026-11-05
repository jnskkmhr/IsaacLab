# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Manager-based G1 task configurations for recycling moving terrain."""

import math

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.visualizers import VisualizerCfg

from .env_cfg import (
    DEFAULT_PROXY_MASS_SCALE,
    G1ActionsCfg,
    G1CommandsCfg,
    G1CurriculumCfg,
    G1EventCfg,
    G1MovingPatchSceneCfg,
    G1ObservationsCfg,
    G1ParticleNeRDPhysicsCfg,
    G1RewardsCfg,
    G1TerminationsCfg,
    terrain_cfg,
)
from .env_cfg.visualizer_cfg import MovingPatchGLVisualizerCfg, MovingPatchRTXVisualizerCfg


@configclass
class G1ParticleNeRDEnvCfg(ManagerBasedRLEnvCfg):
    """Configure G1 with Particle NeRD and task-local learning settings."""

    scene: G1MovingPatchSceneCfg = G1MovingPatchSceneCfg(num_envs=1, env_spacing=4.0)
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200, render_interval=4, use_newton_actuators=True, physics=G1ParticleNeRDPhysicsCfg()
    )
    decimation: int = 4
    is_finite_horizon: bool = False

    # mdp terms
    observations: G1ObservationsCfg = G1ObservationsCfg()
    actions: G1ActionsCfg = G1ActionsCfg()
    commands: G1CommandsCfg = G1CommandsCfg()
    rewards: G1RewardsCfg = G1RewardsCfg()
    terminations: G1TerminationsCfg = G1TerminationsCfg()
    events: G1EventCfg = G1EventCfg()
    curriculum: G1CurriculumCfg = G1CurriculumCfg()

    seed: int = 42

    # Terrain contact state
    foot_body_expr: str = ".*_ankle_roll_link"
    """Body-name expression selecting the feet tracked against the bed."""

    foot_contact_force_threshold: float = 5.0
    """Granular reaction magnitude above which a foot counts as in contact [N]."""

    foot_contact_margin_range: tuple[float, float] | None = None
    """Optional foot collider margins [m], evenly spaced and randomly assigned per environment.
    Applied before solver construction to both rigid and neural particle contact. None preserves
    asset margins.
    A single environment receives the lower bound. Margins remain fixed across resets.
    """

    reset_particle_jitter: float = 0.0
    """Half-width of the uniform position jitter applied when the bed is reset [m]."""

    particle_max_velocity: float = 20.0
    """Speed limit enforced on the particles by the Newton model [m/s]."""

    proxy_mass_scale: float = DEFAULT_PROXY_MASS_SCALE
    """Dimensionless effective-mass multiplier for rigid proxies."""

    def validate_config(self) -> None:
        """Apply final terrain overrides and capacities before simulation creation.

        IsaacLab calls this through ``cfg.validate()`` after CLI/Hydra overrides.
        Repeated validation recomputes derived settings without replacing the solvers.
        """

        use_terrain_curriculum = getattr(self.curriculum, "terrain_levels", None) is not None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.curriculum = use_terrain_curriculum
        if use_terrain_curriculum:
            self.scene.terrain.use_terrain_origins = True
        if self.sim.physics.use_cuda_graph:
            raise ValueError("Use capture_solver=True and use_cuda_graph=False for neural particle inference")
        self.scene.configure_terrain()
        # GL/RTX submit the filtered particle batch themselves.
        # An authored Points visual would also be rendered by RTX's borrowed stage.
        self.scene.sand.spawn.visible = any(v.visualizer_type == "kit" for v in self.sim.visualizer_cfgs)
        self.sim.physics.configure_terrain(self.scene.terrain.moving_patch_terrain, self.proxy_mass_scale)  # type: ignore

    def __post_init__(self):
        self.episode_length_s = 20.0
        self.sim.render_interval = self.decimation

        # self.sim.default_visualizer_cfg = VisualizerCfg(eye=(0.0, -15.0, 4.0), lookat=(0.0, 0.0, 0.0))
        self.sim.visualizer_cfgs = [
            MovingPatchGLVisualizerCfg(eye=(-20.0, 0.0, 6.0), lookat=(0.0, 0.0, 0.0), show_particles=False),
            MovingPatchRTXVisualizerCfg(
                eye=(-50.0, -15.0, 5.0),
                lookat=(-10.0, 0.0, 0.0),
                show_particles=False,
            ),
            # MovingPatchKitVisualizerCfg(eye=(0.0, -15.0, 4.0), show_particles=False),
        ]
        self.video_recorders = [
            VideoRecorderCfg(source="visualizer:newton_gl", output_dir=None, video_length=200, video_interval=2000),
            # VideoRecorderCfg(source="visualizer:newton_rtx", output_dir=None, video_length=200, video_interval=2000),
            # VideoRecorderCfg(source="visualizer:kit", output_dir="videos/"),
        ]

        # self.curriculum.command_vel = None  # type: ignore


@configclass
class G1ParticleNeRDEnvCfg_PLAY(G1ParticleNeRDEnvCfg):
    """Deterministic reset and forward command for inspecting the moving terrain."""

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.enable_corruption = False

        generator = terrain_cfg.ROUGH_TERRAINS_DIFFICULT_CFG
        # generator = terrain_cfg.SLOPE_TERRAINS_CFG
        # generator = terrain_cfg.WAVE_TERRAINS_CFG
        # generator = terrain_cfg.FLAT_TERRAINS_CFG

        self.scene.terrain.terrain_generator = generator

        for name in ("add_base_mass", "push_robot", "physics_material", "scale_actuator_gains"):
            setattr(self.events, name, None)
        self.events.reset_base.params = {
            "pose_range": {"x": (-5.0, 5.0), "y": (-5.0, 5.0), "yaw": (-math.pi, math.pi)},
            "velocity_range": {key: (0.0, 0.0) for key in ("x", "y", "z", "roll", "pitch", "yaw")},
        }
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)

        # override command ranges for the PLAY configuration
        self.curriculum.command_vel = None  # type: ignore
        for name in ("track_lin_vel", "track_ang_vel", "track_heading", "track_lin_vel_weight"):
            setattr(self.curriculum, name, None)

        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.resampling_time_range = (5.0, 5.0)
        # self.commands.base_velocity.debug_vis = False

        self.sim.default_visualizer_cfg = VisualizerCfg(eye=(0.0, -15.0, 2.0), lookat=(0.0, 0.0, 0.0))
        self.sim.visualizer_cfgs = [
            MovingPatchGLVisualizerCfg(
                show_particles=False,
                eye=(-2.0, -0.0, 0.5),
                lookat=(0.0, 0.0, 0.0),
                origin_type="asset",
                origin_track_path="robot",
                streaming_view=False,
            ),
            MovingPatchRTXVisualizerCfg(
                show_particles=False,
                # eye=(-2.0, -0.0, 0.5),
                # lookat=(0.0, 0.0, 0.0),
                eye=(-0.0, -25.0, 5.0),
                lookat=(0.0, 0.0, 0.0),
                # origin_type="asset",
                # origin_track_path="robot",
                # streaming_view=False,
                headless=True,
            ),
            # MovingPatchKitVisualizerCfg(eye=(0.0, -15.0, 4.0)),
        ]
        self.video_recorders = [
            # VideoRecorderCfg(source="visualizer:newton_gl", output_dir=None, video_length=200, video_interval=2000),
            VideoRecorderCfg(source="visualizer:newton_rtx", output_dir=None, video_length=1000, video_interval=2000),
            # VideoRecorderCfg(source="visualizer:kit", output_dir="videos/"),
        ]
