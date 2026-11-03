# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Manager-based G1 task configurations for moving-patch MPM terrain."""

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from isaaclab_tasks.contrib.g1_29dof_mpm_moving_patch.util.solver_setting import configure_sparse_mpm_capacities
from isaaclab_tasks.contrib.g1_29dof_mpm_moving_patch.util.visualizer import MovingPatchGLVisualizerCfg

from .env_cfg import (
    DEFAULT_PROXY_MASS_SCALE,
    G1ActionsCfg,
    G1CommandsCfg,
    G1CurriculumCfg,
    G1EventCfg,
    G1MovingPatchSceneCfg,
    G1ObservationsCfg,
    G1PhysicsProxyCfg,
    G1RewardsCfg,
    G1TerminationsCfg,
)


@configclass
class G1MovingPatchAgileEnvCfg(ManagerBasedRLEnvCfg):
    """Configure G1 directly on sand with two-way coupling and local learning settings."""

    scene: G1MovingPatchSceneCfg = G1MovingPatchSceneCfg(num_envs=64, env_spacing=4.0)
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200, render_interval=4, use_newton_actuators=False, physics=G1PhysicsProxyCfg()
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

    seed: int = 1
    compute_final_obs: bool = True

    # Terrain contact state
    foot_body_expr: str = ".*_ankle_roll_link"
    """Body-name expression selecting the feet tracked against the bed."""

    foot_contact_force_threshold: float = 5.0
    """Granular reaction magnitude above which a foot counts as in contact [N]."""

    foot_contact_margin_range: tuple[float, float] | None = None
    """Derived as (0.5 * voxel_size, 0.5 * voxel_size) before solver construction.

    Mixed terrain instead assigns its fixed per-terrain foot margins.
    """

    reset_particle_jitter: float = 0.0
    """Half-width of the uniform position jitter applied when the bed is reset [m]."""

    particle_max_velocity: float = 20.0
    """Speed limit enforced on the particles by the Newton model [m/s]."""

    # -- solver capacities, scaled with the world count before the simulation is created
    proxy_mass_scale: float = DEFAULT_PROXY_MASS_SCALE
    mpm_active_cell_count_per_world: int | None = None
    """Reserved cells per world; None follows particle count, with leaf capacity as a minimum."""
    mpm_leaf_node_count_per_world: int = 1 << 13
    mpm_lower_node_count_per_world: int = 1 << 10
    mpm_upper_node_count_per_world: int = 16

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
        self.scene.configure_terrain()
        self.sim.physics.configure_terrain(self.scene.terrain.moving_patch_terrain, self.proxy_mass_scale)  # type: ignore
        configure_sparse_mpm_capacities(self)
        margin = 0.5 * self.scene.terrain.moving_patch_terrain.voxel_size
        self.foot_contact_margin_range = (margin, margin)

    def __post_init__(self):
        self.episode_length_s = 30.0
        self.sim.render_interval = self.decimation
        self.scene.height_scanner.update_period = self.sim.dt * self.decimation
        self.scene.height_measurement_sensor.update_period = self.sim.dt

        self.sim.visualizer_cfgs = [
            MovingPatchGLVisualizerCfg(
                origin_type="asset",
                origin_track_path="robot",
                origin_env_index=0,
                eye=(2.5, -2.5, 1.5),
                lookat=(0.0, 0.0, -0.2),
                show_particles=True,
            ),
        ]
        self.video_recorders = [
            VideoRecorderCfg(source="visualizer:newton_gl", output_dir=None, video_length=200, video_interval=2000),
        ]


@configclass
class G1MovingPatchAgileEnvCfg_PLAY(G1MovingPatchAgileEnvCfg):
    """Play the same coupled task with one environment."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
