* When defining term config, avoid inheriting from tasks outside mimic 
* For example, when you inherit terms from velocity, it is bit hard to track where this term came from. 
```python 
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_tasks.contrib.mimic.config.g1_29dof.env_cfg.action_cfg import G1ActionsCfg as RigidActionsCfg
from isaaclab_tasks.contrib.velocity.config.g1_29dof_soft.env_cfg.action_cfg import (
    G1ActionsCfg as SoftVelocityActionsCfg,
)


@configclass
class G1ActionsCfg(RigidActionsCfg):
    """Preserve joint actions and add the zero-dimensional soft-contact callback."""

    physics_callback = SoftVelocityActionsCfg().physics_callback.copy()

```

I would just write everything. 
```python 
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils.configclass import configclass

from isaaclab_tasks.contrib.soft_contact import (
    BoxColliderCfg,
    PhysicsCallbackActionCfg,
    PlaneColliderCfg,
    SphereColliderCfg,
)
from isaaclab_tasks.contrib.velocity.config.g1_29dof_rigid.mdp import symmetry
from isaaclab_tasks.contrib.velocity.config.vel_mdp import (
    MirrorActionTermCfg,
    MirrorJointPositionActionCfg,
    mirror_identity,
)

"""
collider geometry
"""

COLLIDER_SHAPE = "box"  # "plane", "sphere"

if COLLIDER_SHAPE == "plane":
    collider_cfg = PlaneColliderCfg(
        contact_edge_x=(-0.065, 0.141),
        contact_edge_y=(-0.0368, 0.0368),
        contact_edge_z=(-0.03539, 0.0),
        resolution=(5, 5),
    )
elif COLLIDER_SHAPE == "box":
    collider_cfg = BoxColliderCfg(
        contact_edge_x=(-0.065, 0.141),
        contact_edge_y=(-0.0368, 0.0368),
        contact_edge_z=(-0.03539, 0.0),
        resolution=(5, 5),
    )
elif COLLIDER_SHAPE == "sphere":
    collider_cfg = SphereColliderCfg(
        radius=0.05,
        center=(0.0, 0.0, 0.0),
        resolution=(8, 8),
    )

SOFT_CONTACT_THRESHOLD = 40.0

contact_model = "3D-warp"
# contact_model = "2D-warp"
# contact_model = "cone-drft"
# contact_model = "cone-drft-multipoint"


ACTIVE_JOINT = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]


@configclass
class MirrorPhysicsCallbackActionCfg(PhysicsCallbackActionCfg, MirrorActionTermCfg):
    """Soft-contact callback with an identity mirror for its empty policy action slice."""

    mirror = mirror_identity


@configclass
class G1ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = MirrorJointPositionActionCfg(
        mirror=symmetry.mirror_g1_joints,
        mirror_params={"joint_names": ACTIVE_JOINT},
        asset_name="robot",
        joint_names=ACTIVE_JOINT,
        scale=0.25,
        use_default_offset=True,
        preserve_order=True,
    )

    """
    Contact solver.
    """
    physics_callback = MirrorPhysicsCallbackActionCfg(
        asset_name="robot",
        body_names=[".*_ankle_roll_link"],
        backend=contact_model,
        intruder_geometry_cfg=collider_cfg,
        enable_ema_filter=False,
        contact_threshold=SOFT_CONTACT_THRESHOLD,
        debug_vis=False,
        contact_data_history_length=10,  # logging interval = 0.005*10 = 0.05s, 10 history -> 0.5s
        history_logging_decimation=10,
        contact_vis_force_threshold=SOFT_CONTACT_THRESHOLD,
    )

```

same goes for curriculum and events. 
I think scene is ok. 