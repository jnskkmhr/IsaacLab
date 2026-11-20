# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The streaming sand scene of the G1 task with a free box, sphere and cylinder in front of the robot."""

from isaaclab_newton.sim.schemas import NewtonCollisionPropertiesCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.utils import configclass

from ...g1_29dof.env_cfg.streaming_scene_cfg import FOOT_CONTACT_MARGIN, G1MPMSceneCfg

BOX_NAME = "Box"
"""Prim and body name of the box; the streaming pool and the activity region find it by this name."""
BOX_SIZE = (0.3, 0.3, 0.3)
"""Edge lengths of the box [m]."""
BOX_MASS = 15.0
"""Mass of the box [kg]."""
SPHERE_NAME = "Sphere"
"""Prim and body name of the sphere."""
SPHERE_RADIUS = 0.15
"""Radius of the sphere [m]."""
SPHERE_MASS = 15.0
"""Mass of the sphere [kg]."""
CYLINDER_NAME = "Cylinder"
"""Prim and body name of the cylinder."""
CYLINDER_RADIUS = 0.15
"""Radius of the upright cylinder [m]."""
CYLINDER_HEIGHT = 0.3
"""Height of the upright cylinder [m]."""
CYLINDER_MASS = 15.0
"""Mass of the cylinder [kg]."""


@configclass
class G1LocomanipSceneCfg(G1MPMSceneCfg):
    """G1, shared terrain and sand layer, plus a free box, sphere and cylinder resting on the sand ahead."""

    box: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/" + BOX_NAME,
        spawn=sim_utils.CuboidCfg(
            size=BOX_SIZE,
            rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(),
            mass_props=sim_utils.MassCfg(mass=BOX_MASS),
            # The same contact margin as the feet, so the box meets the sand the same way.
            collision_props=NewtonCollisionPropertiesCfg(
                collision_enabled=True, contact_margin=FOOT_CONTACT_MARGIN, contact_gap=0.0
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.35, 0.75), roughness=0.6),
        ),
        # One metre ahead of the robot, just above the sand surface.
        init_state=RigidObjectCfg.InitialStateCfg(pos=(1.0, 0.0, 0.5 * BOX_SIZE[2] + 0.05)),
    )
    sphere: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/" + SPHERE_NAME,
        spawn=sim_utils.SphereCfg(
            radius=SPHERE_RADIUS,
            rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(),
            mass_props=sim_utils.MassCfg(mass=SPHERE_MASS),
            collision_props=NewtonCollisionPropertiesCfg(
                collision_enabled=True, contact_margin=FOOT_CONTACT_MARGIN, contact_gap=0.0
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.80, 0.20, 0.15), roughness=0.6),
        ),
        # To the left of the box.
        init_state=RigidObjectCfg.InitialStateCfg(pos=(1.0, 0.8, SPHERE_RADIUS + 0.05)),
    )
    cylinder: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/" + CYLINDER_NAME,
        spawn=sim_utils.CylinderCfg(
            radius=CYLINDER_RADIUS,
            height=CYLINDER_HEIGHT,
            axis="Z",
            rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(),
            mass_props=sim_utils.MassCfg(mass=CYLINDER_MASS),
            collision_props=NewtonCollisionPropertiesCfg(
                collision_enabled=True, contact_margin=FOOT_CONTACT_MARGIN, contact_gap=0.0
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.20, 0.65, 0.30), roughness=0.6),
        ),
        # To the right of the box.
        init_state=RigidObjectCfg.InitialStateCfg(pos=(1.0, -0.8, 0.5 * CYLINDER_HEIGHT + 0.05)),
    )
