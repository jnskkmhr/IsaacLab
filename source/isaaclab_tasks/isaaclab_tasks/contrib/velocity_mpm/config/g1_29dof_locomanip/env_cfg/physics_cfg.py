# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Coupled MJWarp/MPM physics presets with several bodies on the sand: the robot and three free objects.

Each body carries its own streaming pool, which needs the tile cache; a pool without a cache follows
exactly one body. Select a preset with ``physics=NAME``.
"""

from __future__ import annotations

from isaaclab_newton.physics import MPMSleepingCfg, ParticleCacheCfg, StrainActivationCfg, TrackedRegionActivationCfg

from isaaclab.utils import configclass

from isaaclab_tasks.utils import PresetCfg

from ...g1_29dof.env_cfg.physics_cfg import G1StreamPhysicsProxyCfgAllParticleActiveNoCache
from .streaming_scene_cfg import BOX_NAME, CYLINDER_NAME, SPHERE_NAME

OBJECT_NAMES = [BOX_NAME, SPHERE_NAME, CYLINDER_NAME]
"""Free objects resting on the sand."""

OBJECT_BODIES = [r"/World/envs/env_.*/" + name for name in OBJECT_NAMES]
"""The objects as rigid bodies of the MJWarp entry and as colliders of the MPM entry."""

ANCHOR_BODY_NAMES = ["pelvis", *OBJECT_NAMES]
"""Bodies that each carry a streaming pool, and around which the sleeping preset keeps the sand awake."""


@configclass
class G1LocomanipPhysicsCfgAllParticleActive(G1StreamPhysicsProxyCfgAllParticleActiveNoCache):
    """The robot and every object each carry a cached streaming pool; every loaded particle is simulated."""

    def __post_init__(self):
        solver_cfg = self.solver_cfg
        rigid, sand = solver_cfg.entries
        rigid.bodies.extend(OBJECT_BODIES)
        solver_cfg.proxies[0].bodies.extend(OBJECT_BODIES)
        streaming = sand.solver_cfg.streaming
        streaming.body_names = list(ANCHOR_BODY_NAMES)
        # Four pools that do not overlap load four times the tiles of one; the rest is room for displaced sand.
        streaming.cache = ParticleCacheCfg(simulation_capacity_multiplier=8.0)


@configclass
class G1LocomanipPhysicsCfgCache(G1LocomanipPhysicsCfgAllParticleActive):
    """Cached streaming pools in which settled sand sleeps.

    Sand stays awake only inside a box around each body, and settled sand inside those boxes sleeps too.
    """

    def __post_init__(self):
        super().__post_init__()
        mpm = self.solver_cfg.entries[1].solver_cfg
        mpm.reuse_grid_topology = True
        mpm.sleeping = MPMSleepingCfg(
            strain=StrainActivationCfg(stress_filter_time=0.1),
            region=TrackedRegionActivationCfg(body_names=list(ANCHOR_BODY_NAMES), size=(1.2, 1.2, 3.0)),
        )


@configclass
class G1LocomanipPhysicsCfg(PresetCfg):
    """Activation presets for the cached streaming pools, named as in the single-robot task."""

    solver_cfg_all_active_cache = G1LocomanipPhysicsCfgAllParticleActive()
    solver_cfg_cache = G1LocomanipPhysicsCfgCache()
    default = solver_cfg_cache
