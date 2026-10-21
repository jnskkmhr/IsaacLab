# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Physics backend presets for the G1 29-DoF motion-tracking (mimic) task.

Ported from the manager-based PhysX-only config (``isaaclab_tasks.manager_based.mimic``), which set
``sim.physx.gpu_max_rigid_patch_count`` directly. That task also drives an ``AssistiveWrench`` event
every physics step (see ``env_cfg/event_cfg.py``), so -- same as the soft-terrain locomotion task in
``contrib/velocity`` -- the PhysX backends need to re-apply external forces on every solver iteration.
"""

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_ov.physics import OvPhysxCfg
from isaaclab_physx.physics import PhysxCfg

from isaaclab.physics import PhysxAutoCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.utils import PresetCfg


@configclass
class G1PhysicsCfg(PresetCfg):
    """Backend presets for the G1 29-DoF motion-tracking (mimic) task."""

    isaacsim_physx = PhysxCfg(gpu_max_rigid_patch_count=10 * 2**15, enable_external_forces_every_iteration=True)
    ovphysx = OvPhysxCfg(gpu_max_rigid_patch_count=10 * 2**15, enable_external_forces_every_iteration=True)
    physx = PhysxAutoCfg(isaacsim_physx=isaacsim_physx, ovphysx=ovphysx)
    newton_mjwarp = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            njmax=1000,
            nconmax=300,
            cone="pyramidal",
            impratio=1.0,
            integrator="implicitfast",
            use_mujoco_contacts=False,
        ),
        num_substeps=2,
        debug_mode=False,
    )
    default = newton_mjwarp
