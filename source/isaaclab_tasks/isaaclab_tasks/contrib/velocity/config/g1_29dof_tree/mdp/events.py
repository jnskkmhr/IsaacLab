# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Event terms of the G1 29-DoF locomotion task among trees."""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    import torch

    from isaaclab.assets import CableObject
    from isaaclab.envs import ManagerBasedEnv


def reset_cables(env: ManagerBasedEnv, env_ids: torch.Tensor, asset_cfgs: list[SceneEntityCfg]) -> None:
    """Restore cables to their authored shape and pose.

    A cable is not reset by the asset itself, so without this term an obstacle that the robot
    kicked apart in one episode stays apart for every episode after it, and the task the policy is
    trained on drifts away from the authored one.

    Args:
        env: The environment.
        env_ids: Environments to reset.
        asset_cfgs: The cables to restore.
    """
    for asset_cfg in asset_cfgs:
        asset: CableObject = env.scene[asset_cfg.name]
        segment_pose = asset.data.default_segment_pose_w.torch[env_ids].clone()
        segment_velocity = asset.data.default_segment_velocity_w.torch[env_ids].clone()
        asset.write_segment_pose_to_sim_index(segment_pose=segment_pose, env_ids=env_ids)
        asset.write_segment_velocity_to_sim_index(segment_velocity=segment_velocity, env_ids=env_ids)
