# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Random rigid-ground and MPM-ground assignments for one G1 policy."""

import math

import isaaclab.envs.mdp as mdp
from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg  # noqa: F401
from isaaclab.utils import configclass

from .env_cfg import (
    FLAT_TERRAINS_CFG,  # noqa: F401
    ROUGH_TERRAINS_CFG,  # noqa: F401
    ROUGH_TERRAINS_DIFFICULT_CFG,
    SLOPE_TERRAINS_CFG,  # noqa: F401
    WAVE_TERRAINS_CFG,  # noqa: F401
)
from .env_cfg.mixed_scene_cfg import G1MixedTerrainSceneCfg
from .env_cfg.physics_cfg import G1PhysicsProxyCfg
from .env_cfg.scene_cfg import FOOT_CONTACT_MARGIN
from .mdp.curriculums import terrain_levels_vel
from .mixed_env import outside_contact_region, randomize_mpm_material
from .mpm_env_cfg import G1MovingPatchEnvCfg
from .util.visualizer import (
    MovingPatchGLVisualizerCfg,
    MovingPatchRTXVisualizerCfg,
)


@configclass
class G1MixedTerrainEnvCfg(G1MovingPatchEnvCfg):
    """Sample rigid or MPM contact independently per environment, fixed across resets."""

    scene: G1MixedTerrainSceneCfg = G1MixedTerrainSceneCfg(num_envs=64, env_spacing=4.0)

    mpm_ratio: float = 0.5
    """Probability of MPM contact per environment, in [0, 1]. A single environment always uses MPM."""

    mpm_contact_margin: float = FOOT_CONTACT_MARGIN
    """Fixed robot foot collider margin in MPM environments [m]."""

    rigid_contact_margin: float = 0.0
    """Fixed robot foot collider margin in rigid-only environments [m]."""

    foot_contact_margin_range: tuple[float, float] | None = None
    """Disabled for mixed terrain; use the fixed per-terrain margins above."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 64
        self.sim.physics = G1PhysicsProxyCfg()
        self.curriculum.terrain_levels.func = terrain_levels_vel
        self.terminations.terrain_out_of_bounds.func = outside_contact_region
        for name in ("mpm_material", "mpm_material_log"):
            term = getattr(self.events, name, None)
            if term is not None:
                term.func = randomize_mpm_material

        self.curriculum.command_vel.params["velocity_stages"] = [
                {"step": 0, "lin_vel_x": (-1.0, 1.0), "ang_vel_z": (-0.5, 0.5)},
                {"step": 10000 * 24, "lin_vel_x": (-1.0, 1.5), "ang_vel_z": (-1.0, 1.0)},
            ]

    def validate_config(self) -> None:
        if not 0.0 <= self.mpm_ratio <= 1.0:
            raise ValueError("mpm_ratio must be between 0 and 1.")
        self.scene.clone_cfg.clone_combinations[0].weight = 1.0 - self.mpm_ratio
        self.scene.clone_cfg.clone_combinations[1].weight = self.mpm_ratio
        if self.foot_contact_margin_range is not None:
            raise ValueError(
                "Mixed terrain uses mpm_contact_margin and rigid_contact_margin, not a sampled margin range."
            )
        for name in ("mpm_contact_margin", "rigid_contact_margin"):
            margin = getattr(self, name)
            if not math.isfinite(margin) or margin < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative.")
        if self.scene.num_envs < 1:
            raise ValueError("Mixed terrain requires num_envs >= 1.")
        super().validate_config()


@configclass
class G1MixedTerrainEnvCfg_PLAY(G1MixedTerrainEnvCfg):
    """Play sampled rigid or MPM environments without pushes or observation noise."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 2
        self.observations.policy.enable_corruption = False
        self.observations.policy.projected_gravity.func = mdp.projected_gravity
        self.observations.policy.projected_gravity.params = {}
        self.curriculum.gravity_roll_bias = None
        self.curriculum.gravity_pitch_bias = None
        self.events.push_robot = None
        for visualizer_cfg in self.sim.visualizer_cfgs:
            visualizer_cfg.headless = False

        generator = ROUGH_TERRAINS_DIFFICULT_CFG
        # generator = SLOPE_TERRAINS_CFG
        # generator = WAVE_TERRAINS_CFG
        # generator = FLAT_TERRAINS_CFG

        self.scene.terrain.terrain_generator = generator
        self.scene.terrain.moving_patch_terrain.particle_depth = 0.35

        self.events.mpm_material_log = None
        self.events.mpm_material = None

        # Keep the play command ranges unchanged when environments reset.
        self.curriculum.command_vel = None
        self.commands.base_velocity.ranges.lin_vel_x = (1, 1)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.0, 0.0)
        self.commands.base_velocity.rel_standing_envs = 0.5
        self.commands.base_velocity.resampling_time_range = (5.0, 5.0)

        self.sim.visualizer_cfgs = [
            MovingPatchGLVisualizerCfg(
                eye=(0.0, -2.0, 0.5),
                lookat=(0.0, 0.0, 0.0),
                origin_type="asset",
                origin_track_path="robot",
                streaming_view=False,
            ),
            MovingPatchRTXVisualizerCfg(
                eye=(0.0, -2.0, 0.5),
                lookat=(0.0, 0.0, 0.0),
                origin_type="asset",
                origin_track_path="robot",
                streaming_view=False,
                origin_env_index=0,
            ),
            # MovingPatchKitVisualizerCfg(eye=(0.0, -15.0, 4.0)),
        ]
        self.video_recorders = [
            # VideoRecorderCfg(source="visualizer:newton_gl", output_dir=None, video_length=200, video_interval=2000),
            # VideoRecorderCfg(source="visualizer:newton_rtx", output_dir=None, video_length=1000, video_interval=2000),
            # VideoRecorderCfg(source="visualizer:kit", output_dir="videos/"),
        ]
