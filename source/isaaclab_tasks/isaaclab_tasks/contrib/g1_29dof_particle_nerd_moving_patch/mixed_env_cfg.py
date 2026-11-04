# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Mixed neural/rigid terrain using task-local learning and solver settings."""

import math

from isaaclab.utils import configclass

from isaaclab_tasks.contrib.g1_29dof_mpm_moving_patch.mdp.curriculums import terrain_levels_vel
from isaaclab_tasks.contrib.g1_29dof_mpm_moving_patch.mixed_env import outside_contact_region

from .env_cfg import terrain_cfg
from .env_cfg.mixed_scene_cfg import G1MixedTerrainSceneCfg
from .env_cfg.scene_cfg import FOOT_CONTACT_MARGIN
from .env_cfg.visualizer_cfg import MovingPatchGLVisualizerCfg, MovingPatchRTXVisualizerCfg
from .particle_nerd_env_cfg import G1ParticleNeRDEnvCfg


@configclass
class G1MixedTerrainEnvCfg(G1ParticleNeRDEnvCfg):
    """Keep equal, fixed neural-sand and rigid-ground groups across resets."""

    scene: G1MixedTerrainSceneCfg = G1MixedTerrainSceneCfg(num_envs=64, env_spacing=4.0)
    particle_contact_margin: float = FOOT_CONTACT_MARGIN
    """Foot margin in neural sand environments [m]."""
    rigid_contact_margin: float = 0.0
    """Foot margin in rigid environments [m]."""

    def __post_init__(self):
        super().__post_init__()
        self.curriculum.terrain_levels.func = terrain_levels_vel
        self.terminations.terrain_out_of_bounds.func = outside_contact_region

    def validate_config(self):
        if self.scene.num_envs < 2 or self.scene.num_envs % 2:
            raise ValueError("Mixed terrain requires an even num_envs >= 2 (half neural sand, half rigid).")
        if self.foot_contact_margin_range is not None:
            raise ValueError("Mixed terrain uses particle_contact_margin and rigid_contact_margin.")
        for name in ("particle_contact_margin", "rigid_contact_margin"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative.")
        super().validate_config()


@configclass
class G1MixedTerrainEnvCfg_PLAY(G1MixedTerrainEnvCfg):
    """Evaluate the same forward command on neural sand and rigid ground."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 2
        self.scene.terrain.terrain_generator = terrain_cfg.ROUGH_TERRAINS_DIFFICULT_CFG.copy()
        self.observations.policy.enable_corruption = False

        for name in ("add_base_mass", "push_robot", "physics_material", "scale_actuator_gains"):
            setattr(self.events, name, None)
        # self.events.reset_base.params = {"pose_range": {}, "velocity_range": {}}
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        for name in ("command_vel", "track_lin_vel", "track_ang_vel", "track_heading", "track_lin_vel_weight"):
            setattr(self.curriculum, name, None)

        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.resampling_time_range = (1000.0, 1000.0)

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
                # origin_type="asset",
                # origin_track_path="robot",
                # streaming_view=False,
                # origin_env_index=0,
            ),
            # MovingPatchKitVisualizerCfg(eye=(0.0, -15.0, 4.0)),
        ]
        self.video_recorders = [
            # VideoRecorderCfg(source="visualizer:newton_gl", output_dir=None, video_length=200, video_interval=2000),
            # VideoRecorderCfg(source="visualizer:newton_rtx", output_dir=None, video_length=1000, video_interval=2000),
            # VideoRecorderCfg(source="visualizer:kit", output_dir="videos/"),
        ]
