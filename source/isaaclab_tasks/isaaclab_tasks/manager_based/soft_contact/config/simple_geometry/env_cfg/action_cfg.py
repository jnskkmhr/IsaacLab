# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.envs.mdp as mdp
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.soft_contact import (
    BoxColliderCfg,  # noqa: F401
    CylinderColliderCfg,
    PhysicsCallbackActionCfg,
    PlaneColliderCfg,  # noqa: F401
    SphereColliderCfg,  # noqa: F401
)

# SHAPE_TYPE = "plane"
# SHAPE_TYPE = "box"
SHAPE_TYPE = "sphere"
# SHAPE_TYPE = "cylinder"

if SHAPE_TYPE == "plane":
    collider_cfg = PlaneColliderCfg(
        contact_edge_x=(-0.1, 0.1),
        contact_edge_y=(-0.05, 0.05),
        contact_edge_z=(-0.04, 0.0),
        resolution=(5, 5),
    )

elif SHAPE_TYPE == "box":
    collider_cfg = BoxColliderCfg(
        contact_edge_x=(-0.1, 0.1),
        contact_edge_y=(-0.1, 0.1),
        contact_edge_z=(-0.05, 0.05),
        resolution=(5, 5),
    )
elif SHAPE_TYPE == "sphere":
    collider_cfg = SphereColliderCfg(
        radius=0.05,
        resolution=(8, 8),
    )

elif SHAPE_TYPE == "cylinder":
    collider_cfg = CylinderColliderCfg(
        radius=0.1,
        height=0.1,
        axis=(0.0, 0.0, 1.0),
        # resolution=(10, 50),
        resolution=(5, 5),
    )

else:
    raise ValueError(f"Unsupported shape type: {SHAPE_TYPE}")


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    """
    Contact solver.
    """
    physics_callback = PhysicsCallbackActionCfg(
        asset_name="robot",
        body_names=[".*"],
        # backend="2D-warp",
        backend="3D-warp",
        # backend="cone-drft",
        # backend="cone-drft-multipoint",
        intruder_geometry_cfg=collider_cfg,
        # enable_ema_filter=True,
        enable_ema_filter=False,
        contact_threshold=40.0,
        debug_vis=True,
        # debug_vis=False,
        contact_vis_scale=1000.0,
    )

    # # kinematics animator
    # velocity_setter = contact_mdp.VelocitySetActionCfg(
    #     asset_name="robot",
    #     velocity_stages=[
    #             {"step": 0, "lin_vel_x": (0.0, 0.0), "lin_vel_y": (0.1, 0.1), "lin_vel_z": (-0.1, -0.1), "ang_vel_z": (0.0, 0.0)},
    #             # {"step": 150, "lin_vel_y": (0.1, 0.1), "lin_vel_z": (0.1, 0.1), "ang_vel_z": (0.0, 0.0)},
    #             # {"step": 300, "lin_vel_y": (0.1, 0.1), "lin_vel_z": (-0.1, -0.1), "ang_vel_z": (0.0, 0.0)},
    #             # {"step": 450, "lin_vel_y": (0.1, 0.1), "lin_vel_z": (0.1, 0.1), "ang_vel_z": (0.0, 0.0)},
    #         ],
    # )

contact_model = "3D-warp"
# contact_model = "2D-warp"
# contact_model = "cone-drft"
# contact_model = "cone-drft-multipoint"

box_collider_cfg = BoxColliderCfg(
    contact_edge_x=(-0.1, 0.1),
    contact_edge_y=(-0.1, 0.1),
    contact_edge_z=(-0.05, 0.05),
    resolution=(5, 5),
)

sphere_collider_cfg = SphereColliderCfg(
    radius=0.05,
    resolution=(5, 5),
)

cylinder_collider_cfg = CylinderColliderCfg(
    radius=0.1,
    height=0.1,
    axis=(0.0, 0.0, 1.0),
    resolution=(5, 5),
)

@configclass
class ActionsBoxSceneCfg:
    """Action specifications for the MDP."""

    """
    Contact solver.
    """

    physics_callback = PhysicsCallbackActionCfg(
        asset_name="robot",
        body_names=[".*"],
        backend=contact_model,
        intruder_geometry_cfg=box_collider_cfg,
        # enable_ema_filter=True,
        enable_ema_filter=False,
        contact_threshold=40.0,
        debug_vis=True,
        # debug_vis=False,
        contact_vis_scale=1000.0,
    )

@configclass
class ActionsSphereSceneCfg:
    """Action specifications for the MDP."""

    """
    Contact solver.
    """

    physics_callback = PhysicsCallbackActionCfg(
        asset_name="robot",
        body_names=[".*"],
        backend=contact_model,
        intruder_geometry_cfg=sphere_collider_cfg,
        # enable_ema_filter=True,
        enable_ema_filter=False,
        contact_threshold=40.0,
        debug_vis=True,
        # debug_vis=False,
        contact_vis_scale=1000.0,
    )

@configclass
class ActionsCylinderSceneCfg:
    """Action specifications for the MDP."""

    """
    Contact solver.
    """

    physics_callback = PhysicsCallbackActionCfg(
        asset_name="robot",
        body_names=[".*"],
        backend=contact_model,
        intruder_geometry_cfg=cylinder_collider_cfg,
        # enable_ema_filter=True,
        enable_ema_filter=False,
        contact_threshold=40.0,
        debug_vis=True,
        # debug_vis=False,
        contact_vis_scale=1000.0,
    )