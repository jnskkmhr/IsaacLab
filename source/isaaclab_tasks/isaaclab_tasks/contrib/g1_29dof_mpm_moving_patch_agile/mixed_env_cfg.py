# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Fixed rigid-ground and MPM-ground assignments for one G1 policy."""

import math

from isaaclab.utils import configclass

from isaaclab_tasks.contrib.g1_29dof_mpm_moving_patch.mdp.curriculums import terrain_levels_vel
from isaaclab_tasks.contrib.g1_29dof_mpm_moving_patch.mixed_env import outside_contact_region, randomize_mpm_material

from .env_cfg.mixed_scene_cfg import G1MixedTerrainSceneCfg
from .env_cfg.physics_cfg import G1PhysicsProxyCfg
from .env_cfg.scene_cfg import FOOT_CONTACT_MARGIN
from .mpm_env_cfg import G1MovingPatchAgileEnvCfg


@configclass
class G1MixedTerrainAgileEnvCfg(G1MovingPatchAgileEnvCfg):
    """Assign contiguous MPM and rigid groups on the original terrain grid."""

    scene: G1MixedTerrainSceneCfg = G1MixedTerrainSceneCfg(num_envs=64, env_spacing=4.0)

    mpm_contact_margin: float = FOOT_CONTACT_MARGIN
    """Fixed robot foot collider margin in MPM environments [m]."""

    rigid_contact_margin: float = 0.016
    """Fixed robot foot collider margin in rigid-only environments [m]."""

    foot_contact_margin_range: tuple[float, float] | None = None
    """Disabled for mixed terrain; use the fixed per-terrain margins above."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 64
        self.sim.physics = G1PhysicsProxyCfg()
        self.curriculum.terrain_levels.func = terrain_levels_vel
        self.curriculum.terrain_levels.params = {}
        self.terminations.terrain_out_of_bounds.func = outside_contact_region
        for name in ("mpm_material", "mpm_material_log"):
            term = getattr(self.events, name, None)
            if term is not None:
                term.func = randomize_mpm_material

    def validate_config(self) -> None:
        if self.foot_contact_margin_range is not None:
            raise ValueError(
                "Mixed terrain uses mpm_contact_margin and rigid_contact_margin, not a sampled margin range."
            )
        for name in ("mpm_contact_margin", "rigid_contact_margin"):
            margin = getattr(self, name)
            if not math.isfinite(margin) or margin < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative.")
        if self.scene.num_envs < 2 or self.scene.num_envs % 2:
            raise ValueError("Mixed terrain requires an even num_envs >= 2 (half MPM, half rigid).")
        super().validate_config()
        self.foot_contact_margin_range = None
        self.mpm_contact_margin = 0.5 * self.scene.terrain.moving_patch_terrain.voxel_size


@configclass
class G1MixedTerrainAgileEnvCfg_PLAY(G1MixedTerrainAgileEnvCfg):
    """Play one MPM and one rigid environment."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 2
