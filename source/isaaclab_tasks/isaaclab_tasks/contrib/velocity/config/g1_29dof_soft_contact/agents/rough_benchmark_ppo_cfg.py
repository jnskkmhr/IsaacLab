# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Moving-patch PPO settings with symmetry augmentation and mirror loss."""

from dataclasses import replace

from isaaclab.utils import configclass

from isaaclab_contrib.mdp import compute_mirrored_states

from isaaclab_rl.rsl_rl import RslRlSymmetryCfg

from isaaclab_tasks.contrib.velocity.agents.compositional_ppo_cfg import G1ContactPPORunnerCfg


@configclass
class G1RoughPPORunnerCfg(G1ContactPPORunnerCfg):
    """Fresh soft_contact rough-contact benchmark."""

    experiment_name = "g1_29dof_soft_contact_rough_benchmark"
    wandb_project = "g1_29dof_contact_benchmark"

    algorithm = replace(
        G1ContactPPORunnerCfg().algorithm,
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=True,
            use_mirror_loss=True,
            mirror_loss_coeff=0.1,
            data_augmentation_func=compute_mirrored_states,
        ),
    )
