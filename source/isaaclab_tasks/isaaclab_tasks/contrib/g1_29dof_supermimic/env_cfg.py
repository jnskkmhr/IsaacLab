# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""G1 controls and PR velocity rewards with a flight penalty on rigid and mixed MPM terrain."""

from isaaclab.managers import RewardTermCfg, SceneEntityCfg
from isaaclab.utils import configclass

from isaaclab_tasks.contrib.g1_29dof_mpm_moving_patch_agile.mixed_env_cfg import G1MixedTerrainAgileEnvCfg
from isaaclab_tasks.contrib.g1_29dof_supermimic.mdp import observations, rewards, terminations
from isaaclab_tasks.contrib.velocity.config.g1_29dof_rigid_agile.env_cfg.physics_cfg import G1PhysicsCfg
from isaaclab_tasks.contrib.velocity.config.g1_29dof_rigid_agile.env_cfg.scene_cfg import G1_29DOF_BeyondMimic_delay
from isaaclab_tasks.core.velocity.config.g1.flat_env_cfg import G1FlatEnvCfg
from isaaclab_tasks.core.velocity.velocity_env_cfg import ObservationsCfg


@configclass
class G1SuperMimicEnvCfg(G1FlatEnvCfg):
    """Use the PR rewards with our 29-joint convex-foot G1 and 16 mm rigid margin."""

    final_obs_every_step = True

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = G1_29DOF_BeyondMimic_delay.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            newton_collision_shape_margin=0.016,
        )
        self.sim.physics = G1PhysicsCfg()
        self.sim.physics.newton_mjwarp.default_shape_cfg.margin = 0.016
        self.sim.use_newton_actuators = False
        self.observations.state = ObservationsCfg.StateCfg()
        self.observations.goal = ObservationsCfg.GoalCfg()
        self.observations.context = ObservationsCfg.ContextCfg()
        self.observations.labels = ObservationsCfg.LabelsCfg()
        self.rewards.no_jump = RewardTermCfg(
            func=rewards.no_jump,
            weight=-2.0,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link")},
        )
        # The PR's G1 asset has fingers and one torso joint; this asset has wrists and a three-axis waist.
        self.rewards.joint_deviation_fingers = None
        self.rewards.joint_deviation_arms.params["asset_cfg"] = SceneEntityCfg(
            "robot",
            joint_names=[".*_shoulder_.*", ".*_elbow_joint", ".*_wrist_.*"],
        )
        self.rewards.joint_deviation_torso.params["asset_cfg"] = SceneEntityCfg("robot", joint_names="waist_.*")


@configclass
class G1MixedSuperMimicEnvCfg(G1MixedTerrainAgileEnvCfg):
    """Retain moving-patch coupling and terrain allocation with the rigid task's learning MDP."""

    final_obs_every_step = True

    def __post_init__(self):
        super().__post_init__()
        rigid = G1SuperMimicEnvCfg()
        self.scene.robot = rigid.scene.robot.replace(newton_collision_shape_margin=None)
        self.actions = rigid.actions
        self.observations = rigid.observations
        self.commands = rigid.commands
        self.rewards = rigid.rewards
        self.terminations = rigid.terminations
        self.curriculum = rigid.curriculum
        # Keep particle-material events and terrain-aware reset, with identical actuator gains.
        self.events.randomize_actuator_gains = None
        self.observations.state.state.func = observations.VelocityState
        self.observations.labels.contacts.func = observations.VelocityLabels
        self.rewards.feet_air_time.params.pop("sensor_cfg")
        self.rewards.feet_air_time.params["asset_cfg"] = SceneEntityCfg("robot", body_names=".*_ankle_roll_link")
        self.rewards.feet_slide.params.pop("sensor_cfg")
        self.rewards.no_jump.params = {}
        self.terminations.base_contact.params.pop("sensor_cfg")
        self.terminations.base_contact.params["asset_cfg"] = SceneEntityCfg("robot", body_names="torso_link")
        self.rewards.feet_air_time.func = rewards.feet_air_time_positive_biped
        self.rewards.feet_slide.func = rewards.feet_slide
        self.terminations.base_contact.func = terminations.illegal_contact
        self.foot_contact_force_threshold = 1.0
