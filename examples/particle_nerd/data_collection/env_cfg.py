# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Manager-based collection environment using the current G1 MPM solver settings."""

from isaaclab_newton.physics import MJWarpSolverCfg, MPMSolverCfg, NewtonCfg, NewtonCollisionPipelineCfg

from isaaclab.envs import ManagerBasedEnvCfg
from isaaclab.envs.mdp import reset_root_state_uniform, reset_scene_to_default
from isaaclab.managers import ActionTermCfg, EventTermCfg, SceneEntityCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from isaaclab_contrib.coupling import CouplerEntryCfg, CouplerProxyCfg, CouplerProxyMappingCfg

from particle_nerd.data_collection.scene_cfg import VOXEL_SIZE, ParticleSceneCfg
from particle_nerd.data_collection.terms.events import reset_material
from particle_nerd.data_collection.terms.recordings import ParticleRecorderCfg


@configclass
class ActionsCfg:
    """Optional external wrench; absent for legacy passive trajectories."""

    wrench: ActionTermCfg | None = None


@configclass
class EventsCfg:
    reset_scene = EventTermCfg(func=reset_scene_to_default, mode="reset")
    randomize_box = EventTermCfg(
        func=reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("body5"),
            "pose_range": {
                "x": (-0.025, 0.025),
                "y": (-0.025, 0.025),
                "z": (0.05, 0.12),
                "roll": (-0.35, 0.35),
                "pitch": (-0.35, 0.35),
                "yaw": (-3.14159, 3.14159),
            },
            "velocity_range": {
                "x": (-0.1, 0.1),
                "y": (-0.1, 0.1),
                "z": (-0.2, 0.0),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )
    reset_material = EventTermCfg(func=reset_material, mode="reset")


@configclass
class ObservationsCfg:
    """No policy observations are needed for unactuated trajectories."""


@configclass
class ParticleEnvCfg(ManagerBasedEnvCfg):
    scene = ParticleSceneCfg(num_envs=4, env_spacing=2.0, replicate_physics=True)
    actions = ActionsCfg()
    observations = ObservationsCfg()
    events = EventsCfg()
    recorders = ParticleRecorderCfg()
    decimation = 1
    seed = 42
    body_shape: str = "box"
    sim = SimulationCfg(
        dt=0.005,
        physics=NewtonCfg(
            solver_cfg=CouplerProxyCfg(
                entries=[
                    CouplerEntryCfg(
                        name="rigid",
                        solver_cfg=MJWarpSolverCfg(use_mujoco_contacts=False, njmax=128),
                        bodies=[r"/World/envs/env_.*/Body[0-5]"],
                        substeps=1,
                    ),
                    CouplerEntryCfg(
                        name="sand",
                        solver_cfg=MPMSolverCfg(
                            voxel_size=VOXEL_SIZE,
                            grid_type="sparse",
                            grid_padding=0,
                            strain_basis="P0",
                            transfer_scheme="apic",
                            max_iterations=25,
                            tolerance=1.0e-4,
                            warmstart_mode="auto",
                            velocity_basis="Q1",
                            collider_basis="pic27",
                            collider_velocity_mode="forward",
                            solver="auto",
                            critical_fraction=0.0,
                            separate_worlds=True,
                            project_outside_colliders=False,
                        ),
                        all_particles=True,
                        in_place=True,
                    ),
                ],
                proxies=[
                    CouplerProxyMappingCfg(
                        source="rigid",
                        destination="sand",
                        bodies=[r"/World/envs/env_.*/Body[0-5]"],
                        mode="lagged",
                        collision_pipeline=None,
                    )
                ],
                iterations=1,
            ),
            collision_cfg=NewtonCollisionPipelineCfg(soft_contact_max=0),
            num_substeps=1,
            use_cuda_graph=False,
        ),
    )
