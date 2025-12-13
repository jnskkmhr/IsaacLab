# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Booster robots.

The following configurations are available:

* :obj:`T1_CFG`: T1 humanoid robot with 7DOF Arms
* :obj:`T1_REACH_CFG`: Fixed base version of T1_CFG

Reference: https://booster.feishu.cn/wiki/UvowwBes1iNvvUkoeeVc3p5wnUg
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ActuatorNetMLPCfg, DCMotorCfg, ImplicitActuatorCfg, DelayedPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab_assets import ISAACLAB_ASSETS_DATA_DIR

##
# Configuration
##


T1_29DOF_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/booster/t1_29dof_no_upper_collision_aug28.usd", # less visuals
        # usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/booster/t1_29dof_1108_add_ee.usd", 

        # Turns on contact sensors for collision/contact detection.
        activate_contact_sensors=True,

        # Physical properties for rigid bodies
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,  # Resistance to linear movement.
            angular_damping=0.0,  # Resistance to rotational movement.
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # pos=(0.0, 0.0, 0.72), 
        pos=(0.0, 0.0, 0.69),
        joint_pos={
            "AAHead_yaw": 0.0,
            "Head_pitch": 0.0, 
            ".*_Shoulder_Pitch": 0.2,
            "Left_Shoulder_Roll": -1.35,
            "Right_Shoulder_Roll": 1.35,
            ".*_Elbow_Pitch": 0.0,
            "Left_Elbow_Yaw": -0.5,
            "Right_Elbow_Yaw": 0.5,
            ".*_Wrist_Pitch": 0.0,
            ".*_Wrist_Yaw": 0.0,
            ".*_Hand_Roll": 0.0,
            # ".*_Link1": 0.0,
            # ".*_Link11": 0.0,
            # ".*_Link2": 0.0,
            # ".*_Link22": 0.0,
            "Waist": 0.0,
            ".*_Hip_Pitch": -0.2,
            ".*_Hip_Roll": 0.0,
            ".*_Hip_Yaw": 0.0,
            ".*_Knee_Pitch": 0.4,
            ".*_Ankle_Pitch": -0.25,
            ".*_Ankle_Roll": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "head": ImplicitActuatorCfg(
            effort_limit_sim=7,
            velocity_limit_sim=12.56,
            joint_names_expr=["Head_pitch", "AAHead_yaw"],
            stiffness=40.0,
            damping=5.0,
            armature=0.01,
        ),
        "legs": ImplicitActuatorCfg(
        # "legs": DelayedPDActuatorCfg(
            joint_names_expr=[
                ".*_Hip_Yaw",
                ".*_Hip_Roll",
                ".*_Hip_Pitch",
                ".*_Knee_Pitch",
                "Waist",
            ],
            effort_limit_sim={
                ".*_Hip_Yaw": 30.0,
                ".*_Hip_Roll": 25.0,
                ".*_Hip_Pitch": 60.0,
                ".*_Knee_Pitch": 60.0,
                "Waist": 30.0,
            },
            effort_limit={
                ".*_Hip_Yaw": 30.0,
                ".*_Hip_Roll": 25.0,
                ".*_Hip_Pitch": 60.0,
                ".*_Knee_Pitch": 60.0,
                "Waist": 30.0,
            },
            velocity_limit_sim= {
                ".*_Hip_Yaw": 10.9,
                ".*_Hip_Roll": 10.9,
                ".*_Hip_Pitch": 12.5,
                ".*_Knee_Pitch": 11.7,
                "Waist": 10.88,
            },
            velocity_limit= {
                ".*_Hip_Yaw": 10.9,
                ".*_Hip_Roll": 10.9,
                ".*_Hip_Pitch": 12.5,
                ".*_Knee_Pitch": 11.7,
                "Waist": 10.88,
            },
            stiffness=200.0,
            # stiffness=100.0,
            damping=5.0,
            # damping=2.5,
            armature={
                ".*_Hip_.*": 0.01,
                ".*_Knee_Pitch": 0.01,
                "Waist": 0.01,
            },
            # min_delay=10,
            # max_delay=20
        ),
        "feet": ImplicitActuatorCfg(
        # "feet": DelayedPDActuatorCfg(
            joint_names_expr=[".*_Ankle_Pitch", ".*_Ankle_Roll"],
            effort_limit_sim={
                ".*_Ankle_Pitch": 24.0,
                ".*_Ankle_Roll": 15.0,
            },
            effort_limit={
                ".*_Ankle_Pitch": 24.0,
                ".*_Ankle_Roll": 15.0,
            },
            velocity_limit_sim={
                ".*_Ankle_Pitch": 18.8,
                ".*_Ankle_Roll": 12.4,
            },
            velocity_limit={
                ".*_Ankle_Pitch": 18.8,
                ".*_Ankle_Roll": 12.4,
            },
            stiffness=50.0,
            # stiffness=25.0,
            damping=1.0,
            # damping=3.0,
            armature=0.01,
            # min_delay=10,
            # max_delay=20
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_Shoulder_Pitch",
                ".*_Shoulder_Roll",
                ".*_Elbow_Pitch",
                ".*_Elbow_Yaw",
            ],
            effort_limit_sim={
                ".*_Shoulder_.*": 10.0,
                ".*_Elbow_.*": 10.0,
            },
            velocity_limit_sim={
                ".*_Shoulder_.*": 18.84,
                ".*_Elbow_.*": 18.84,
            },
            stiffness=40.0,
            damping=5.0,
            armature={
                ".*_Shoulder_.*": 0.01,
                ".*_Elbow_.*": 0.01,
            },
        ),

        # wrist 
        "wrist": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_Wrist_Pitch",
                ".*_Wrist_Yaw",
                ".*_Hand_Roll",
            ],
            effort_limit_sim={
                ".*_Wrist_.*": 10.0,
                ".*_Hand_Roll": 10.0,
            },
            velocity_limit_sim={
                ".*_Wrist_.*": 18.84,
                ".*_Hand_Roll": 18.84,
            },
            stiffness=40.0,
            damping=5.0,
            armature={
                ".*_Wrist_.*": 0.001,
                ".*_Hand_Roll": 0.001,
            },
        ),
    },
)


T1_29DOF_MINIMAL_CFG = T1_29DOF_CFG.copy()
"""Configuration for the Booster T1 Humanoid robot with fewer collision meshes.

This configuration removes most collision meshes to speed up simulation.
"""



"""
23dof
"""

T1_23DOF_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/booster/t1_23dof/t1_23dof.usd",

        # Turns on contact sensors for collision/contact detection.
        activate_contact_sensors=True,

        # Physical properties for rigid bodies
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,  # Resistance to linear movement.
            angular_damping=0.0,  # Resistance to rotational movement.
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # pos=(0.0, 0.0, 0.72), 
        pos=(0.0, 0.0, 0.69),
        joint_pos={
            "AAHead_yaw": 0.0,
            "Head_pitch": 0.0, 

            ".*_Hip_Pitch": -0.2,
            ".*_Hip_Roll": 0.0,
            ".*_Hip_Yaw": 0.0,
            ".*_Knee_Pitch": 0.4,
            ".*_Ankle_Pitch": -0.25,
            ".*_Ankle_Roll": 0.0,

            "Waist": 0.0,

            ".*_Shoulder_Pitch": 0.2,
            "Left_Shoulder_Roll": -1.35,
            "Right_Shoulder_Roll": 1.35,
            ".*_Elbow_Pitch": 0.0,
            "Left_Elbow_Yaw": -0.5,
            "Right_Elbow_Yaw": 0.5,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "head": ImplicitActuatorCfg(
            effort_limit_sim=7,
            velocity_limit_sim=12.56,
            joint_names_expr=["Head_pitch", "AAHead_yaw"],
            stiffness=40.0,
            damping=5.0,
            armature=0.01,
        ),
        "legs": ImplicitActuatorCfg(
        # "legs": DelayedPDActuatorCfg(
            joint_names_expr=[
                ".*_Hip_Yaw",
                ".*_Hip_Roll",
                ".*_Hip_Pitch",
                ".*_Knee_Pitch",
                "Waist",
            ],
            effort_limit_sim={
                ".*_Hip_Yaw": 30.0,
                ".*_Hip_Roll": 25.0,
                ".*_Hip_Pitch": 60.0,
                ".*_Knee_Pitch": 60.0,
                "Waist": 30.0,
            },
            effort_limit={
                ".*_Hip_Yaw": 30.0,
                ".*_Hip_Roll": 25.0,
                ".*_Hip_Pitch": 60.0,
                ".*_Knee_Pitch": 60.0,
                "Waist": 30.0,
            },
            velocity_limit_sim= {
                ".*_Hip_Yaw": 10.9,
                ".*_Hip_Roll": 10.9,
                ".*_Hip_Pitch": 12.5,
                ".*_Knee_Pitch": 11.7,
                "Waist": 10.88,
            },
            velocity_limit= {
                ".*_Hip_Yaw": 10.9,
                ".*_Hip_Roll": 10.9,
                ".*_Hip_Pitch": 12.5,
                ".*_Knee_Pitch": 11.7,
                "Waist": 10.88,
            },
            stiffness=200.0,
            # stiffness=100.0,
            damping=5.0,
            # damping=2.5,
            armature={
                ".*_Hip_.*": 0.01,
                ".*_Knee_Pitch": 0.01,
                "Waist": 0.01,
            },
            # min_delay=10,
            # max_delay=20
        ),
        "feet": ImplicitActuatorCfg(
        # "feet": DelayedPDActuatorCfg(
            joint_names_expr=[".*_Ankle_Pitch", ".*_Ankle_Roll"],
            effort_limit_sim={
                ".*_Ankle_Pitch": 24.0,
                ".*_Ankle_Roll": 15.0,
            },
            effort_limit={
                ".*_Ankle_Pitch": 24.0,
                ".*_Ankle_Roll": 15.0,
            },
            velocity_limit_sim={
                ".*_Ankle_Pitch": 18.8,
                ".*_Ankle_Roll": 12.4,
            },
            velocity_limit={
                ".*_Ankle_Pitch": 18.8,
                ".*_Ankle_Roll": 12.4,
            },
            stiffness=50.0,
            # stiffness=25.0,
            damping=1.0,
            # damping=3.0,
            armature=0.01,
            # min_delay=10,
            # max_delay=20
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_Shoulder_Pitch",
                ".*_Shoulder_Roll",
                ".*_Elbow_Pitch",
                ".*_Elbow_Yaw",
            ],
            effort_limit_sim={
                ".*_Shoulder_.*": 10.0,
                ".*_Elbow_.*": 10.0,
            },
            velocity_limit_sim={
                ".*_Shoulder_.*": 18.84,
                ".*_Elbow_.*": 18.84,
            },
            stiffness=40.0,
            damping=5.0,
            armature={
                ".*_Shoulder_.*": 0.01,
                ".*_Elbow_.*": 0.01,
            },
        ),

        # # wrist 
        # "wrist": ImplicitActuatorCfg(
        #     joint_names_expr=[
        #         ".*_Wrist_Pitch",
        #         ".*_Wrist_Yaw",
        #         ".*_Hand_Roll",
        #     ],
        #     effort_limit_sim={
        #         ".*_Wrist_.*": 10.0,
        #         ".*_Hand_Roll": 10.0,
        #     },
        #     velocity_limit_sim={
        #         ".*_Wrist_.*": 18.84,
        #         ".*_Hand_Roll": 18.84,
        #     },
        #     stiffness=40.0,
        #     damping=5.0,
        #     armature={
        #         ".*_Wrist_.*": 0.001,
        #         ".*_Hand_Roll": 0.001,
        #     },
        # ),
    },
)

T1_23DOF_MINIMAL_CFG = T1_23DOF_CFG.copy()
"""Configuration for the Booster T1 Humanoid robot with fewer collision meshes.

This configuration removes most collision meshes to speed up simulation.
"""