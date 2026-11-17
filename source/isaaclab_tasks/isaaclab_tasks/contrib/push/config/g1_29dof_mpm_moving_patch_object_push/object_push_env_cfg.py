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
from isaaclab_tasks.contrib.velocity_mpm.g1_29dof_mpm_moving_patch.util.visualizer import (
    MovingPatchGLVisualizerCfg,
    MovingPatchRTXVisualizerCfg,
)

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

    scene: G1ObjectPushSceneCfg = G1ObjectPushSceneCfg(num_envs=4, env_spacing=5.0)
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
    push_object_name: str = "object"

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
    # Compact fixed-patch tree reservations; active cells still reserve one per particle.
    mpm_leaf_node_count_per_world: int = 1 << 12
    mpm_lower_node_count_per_world: int = 1 << 7
    mpm_upper_node_count_per_world: int = 4

    def validate_config(self) -> None:
        """Apply final terrain overrides and capacities before simulation creation.

        IsaacLab calls this through ``cfg.validate()`` after CLI/Hydra overrides.
        Repeated validation recomputes derived settings without replacing the solvers.
        """

        import math

        terrain = self.scene.terrain.moving_patch_terrain
        # The flat shared mesh must fit all clone-grid origins, including the full patch.
        extent = math.ceil(math.sqrt(self.scene.num_envs)) * self.scene.env_spacing + 2 * max(terrain.total_patch_size)
        generator = self.scene.terrain.terrain_generator
        generator.size = tuple(max(size, extent) for size in generator.size)
        self.scene.configure_terrain()
        self.commands.object_direction.height_reference_offset = terrain.particle_depth
        self.sim.physics.configure_terrain(terrain, self.proxy_mass_scale)
        configure_sparse_mpm_capacities(self)

    def __post_init__(self):
        self.episode_length_s = 12.0
        self.sim.render_interval = self.decimation
        self.sim.visualizer_cfgs = [
            visualizer(
                eye=(0.0, -2.0, 0.9),
                lookat=(0.4, 0.0, 0.0),
                streaming_view=False,
                show_particles=True,
            )
            for visualizer in (MovingPatchGLVisualizerCfg, MovingPatchRTXVisualizerCfg)
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
        self.events.reset_object.params["forward_range"] = (0.9, 0.9)
        self.commands.object_direction.angle_range = (0.0, 0.0)
        self.commands.object_direction.standing_probability = 0.0
