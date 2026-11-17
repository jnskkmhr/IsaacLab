# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Standalone whole-body box-pushing environments on flat moving-patch sand."""

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from isaaclab_tasks.contrib.velocity_mpm.g1_29dof_mpm_moving_patch.util.solver_setting import (
    configure_sparse_mpm_capacities,
)
from isaaclab_tasks.contrib.velocity_mpm.g1_29dof_mpm_moving_patch.util.visualizer import MovingPatchGLVisualizerCfg

from .env_cfg import (
    DEFAULT_PROXY_MASS_SCALE,
    CommandsCfg,
    CurriculumCfg,
    EventsCfg,
    G1ActionsCfg,
    G1ObjectPushSceneCfg,
    G1PhysicsProxyCfg,
    ObservationsCfg,
    RewardsCfg,
    TerminationsCfg,
)


@configclass
class G1ObjectPushEnvCfg(ManagerBasedRLEnvCfg):
    """Compose task-local scene, control, learning, and coupled-physics settings."""

    scene: G1ObjectPushSceneCfg = G1ObjectPushSceneCfg(num_envs=16, env_spacing=5.0)
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200, render_interval=4, use_newton_actuators=True, physics=G1PhysicsProxyCfg()
    )
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

    # Terrain contact state
    foot_body_expr: str = ".*_ankle_roll_link"
    """Body-name expression selecting the feet tracked against the bed."""

    foot_contact_force_threshold: float = 5.0
    """Granular reaction magnitude above which a foot counts as in contact [N]."""

    foot_contact_margin_range: tuple[float, float] | None = None
    """Optional foot collider margins [m], evenly spaced and randomly assigned per environment.
    Applied before solver construction to both rigid and MPM contact. None preserves
    asset margins.
    A single environment receives the lower bound. Margins remain fixed across resets.
    """

    reset_particle_jitter: float = 0.0
    """Half-width of the uniform position jitter applied when the bed is reset [m]."""

    particle_max_velocity: float = 20.0
    """Speed limit enforced on the particles by the Newton model [m/s]."""

    # -- solver capacities, scaled with the world count before the simulation is created
    proxy_mass_scale: float = DEFAULT_PROXY_MASS_SCALE
    mpm_active_cell_count_per_world: int | None = None
    """Reserved cells per world; None follows particle count, with leaf capacity as a minimum."""
    mpm_leaf_node_count_per_world: int = 1 << 15
    mpm_lower_node_count_per_world: int = 1 << 12
    mpm_upper_node_count_per_world: int = 64

    def validate_config(self) -> None:
        """Apply final terrain overrides and capacities before simulation creation.

        IsaacLab calls this through ``cfg.validate()`` after CLI/Hydra overrides.
        Repeated validation recomputes derived settings without replacing the solvers.
        """

        use_terrain_curriculum = self.curriculum.terrain_levels is not None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.curriculum = use_terrain_curriculum
        if use_terrain_curriculum:
            self.scene.terrain.use_terrain_origins = True
        self.scene.configure_terrain()
        self.sim.physics.configure_terrain(self.scene.terrain.moving_patch_terrain, self.proxy_mass_scale)  # type: ignore
        configure_sparse_mpm_capacities(self)

    def __post_init__(self):
        self.episode_length_s = 12.0
        self.sim.render_interval = self.decimation
        self.sim.visualizer_cfgs = [
            MovingPatchGLVisualizerCfg(
                eye=(2.0, -3.0, 1.5),
                lookat=(0.5, 0.0, -0.2),
                show_particles=True,
                origin_type="asset",
                origin_track_path="robot",
            ),
        ]
        # Let --video select the recorder's renderer.
        self.video_recorders = []


@configclass
class G1ObjectPushEnvCfg_PLAY(G1ObjectPushEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.observations.policy.enable_corruption = False
        self.events.reset_base.params["pose_range"] = {}
        self.events.reset_box.params = {"distance_range": (0.9, 0.9)}
        self.commands.base_velocity.ranges.lin_vel_x = (0.25, 0.25)
        self.commands.base_velocity.rel_standing_envs = 0.0
