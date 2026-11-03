# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""One whole-body actor with additional manipulation observations."""

from isaaclab.utils import configclass

from isaaclab_tasks.contrib.g1_29dof_mpm_moving_patch.agents.compositional_ppo_cfg import G1MovingPatchPPORunnerCfg


@configclass
class G1ObjectPushPPORunnerCfg(G1MovingPatchPPORunnerCfg):
    experiment_name = "g1_29dof_mpm_object_push"
    wandb_project = "g1_29dof_mpm_object_push"
    env_interface = None  # Flat terrain has no terrain-level curriculum to checkpoint.

    def __post_init__(self):
        self.networks.actor.input = ["policy", "command", "manipulation"]
        self.networks.critic.input = ["critic", "command", "privileged", "manipulation", "object_privileged"]
        # Hand/object mirror transforms must be defined before augmentation can be enabled.
        self.algorithm.symmetry_cfg = None
