# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.common import ViewerCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.soft_contact.mdp as contact_mdp

##
# Pre-defined configs
##
from .env_cfg import (
    ActionsCfg,
    CommandsCfg,
    CurriculumCfg,
    EventCfg,
    ObservationsCfg,
    RewardsCfg,
    SceneCfg,
    TerminationsCfg,
)

from .env_cfg import (
    BoxSceneCfg,
    SphereSceneCfg,
    CylinderSceneCfg,
    ActionsBoxSceneCfg,
    ActionsSphereSceneCfg,
    ActionsCylinderSceneCfg,
)


@configclass
class SoftcontactEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the motion tracking environment."""

    # Scene settings
    scene: SceneCfg = SceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        # self.decimation = 10
        self.episode_length_s = 3.0
        # simulation settings
        # self.sim.dt = 1/500
        # self.decimation = 1
        self.sim.dt = 1 / 500
        self.decimation = 10
        # self.sim.dt = 1/200
        # self.decimation = 4
        self.sim.render_interval = self.decimation
        # self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        # change terrain
        self.scene.terrain = contact_mdp.SoftTerrain

        # disable rigid contact
        self.scene.terrain.disable_collider = True
        # self.scene.terrain = None

        # # disable soft contact
        # self.actions.physics_callback.disable = True # disable soft contact
        # self.scene.terrain.disable_collider = False

        # # viewer settings
        # self.viewer = ViewerCfg(
        #     eye=(-12.0, -12.0, 2.0),
        #     lookat=(0.0, 0.0, 0.0),
        #     resolution=(1920, 1080),
        # )
        self.viewer = ViewerCfg(
            eye=(-0.0, -1.2, 0.1),
            lookat=(0.0, -0.0, 0.0),
            # eye=(1.5, 0.0, 0.1),
            # lookat=(0.0, 0.0, 0.0),
            resolution=(1920, 1080),
            # origin_type="asset_root",
            # asset_name="robot",
        )

        # high quality rendering
        # self.sim.render.enable_translucency = True
        # self.sim.render.enable_reflections = True
        # self.sim.render.enable_global_illumination = True  # bounce lighting, colour bleed
        # self.sim.render.enable_ambient_occlusion = True  # contact shadows / depth cues
        # self.sim.render.enable_shadows = True
        # self.sim.render.enable_dl_denoiser = True
        # self.sim.render.antialiasing_mode = "DLAA"
        # self.sim.render.rendering_mode = "quality"
        # self.sim.render.samples_per_pixel = 32

        # self.sim.render.carb_settings = {
        #     "rtx.pathtracing.spp": 32,  # samples per pixel
        #     "rtx.pathtracing.totalSpp": 1024,  # total accumulated samples
        #     "rtx.pathtracing.maxBounces": 8,  # light bounces for realism
        #     "rtx.domeLight.upperLowerStrategy": 3,  # Dome/sky light strategy
        #     "rtx.post.aa.op": 3,  # DLAA op
        #     # Needed to properly render transparent textures (it1's visor clips weirdly without this)
        #     "rtx.raytracing.fractionalCutoutOpacity": True,
        #     "rtx.ambientOcclusion.enabled": True,
        # }

@configclass
class SoftcontactBoxEnvCfg(SoftcontactEnvCfg):
    """Configuration for the motion tracking environment."""

    # Scene settings
    scene: BoxSceneCfg = BoxSceneCfg(num_envs=4096, env_spacing=2.5)
    actions: ActionsBoxSceneCfg = ActionsBoxSceneCfg()

@configclass
class SoftcontactSphereEnvCfg(SoftcontactEnvCfg):
    """Configuration for the motion tracking environment."""

    # Scene settings
    scene: SphereSceneCfg = SphereSceneCfg(num_envs=4096, env_spacing=2.5)
    actions: ActionsSphereSceneCfg = ActionsSphereSceneCfg()

@configclass
class SoftcontactCylinderEnvCfg(SoftcontactEnvCfg):
    """Configuration for the motion tracking environment."""

    # Scene settings
    scene: CylinderSceneCfg = CylinderSceneCfg(num_envs=4096, env_spacing=2.5)
    actions: ActionsCylinderSceneCfg = ActionsCylinderSceneCfg()