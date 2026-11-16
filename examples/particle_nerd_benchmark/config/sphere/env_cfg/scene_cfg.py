# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Standalone sphere benchmark scene."""

from isaaclab_newton.sim.schemas import NewtonCollisionCfg

import isaaclab.sim as sim
from isaaclab.assets import RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from particle_nerd_benchmark.core.terrain import configure_terrain


@configclass
class SceneCfg(InteractiveSceneCfg):
    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=sim.SphereCfg(
            radius=0.10,
            rigid_props=sim.UsdPhysicsRigidBodyCfg(kinematic_enabled=True),
            mass_props=sim.MassCfg(mass=2.0),
            collision_props=NewtonCollisionCfg(contact_margin=0.0),
            visual_material=sim.PreviewSurfaceCfg(diffuse_color=(0.12, 0.32, 0.25)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.65, 0.0, 0.30)),
    )

    def __post_init__(self):
        configure_terrain(self, "flat")
