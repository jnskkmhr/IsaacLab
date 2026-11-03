# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Contact rewards for rigid and coupled terrain."""

import torch

from isaaclab.managers import SceneEntityCfg


def feet_air_time_positive_biped(env, command_name: str, threshold: float, asset_cfg: SceneEntityCfg):
    contact = env.foot_contact_time > 0
    in_mode = torch.where(contact, env.foot_contact_time, env.foot_air_time)
    value = in_mode.amin(-1).clamp(max=threshold) * (contact.sum(-1) == 1)
    return value * (env.command_manager.get_command(command_name)[:, :2].norm(dim=-1) > 0.1)


def feet_slide(env, asset_cfg: SceneEntityCfg):
    contact = env.body_contact_force_history[:, :, asset_cfg.body_ids].norm(dim=-1).amax(dim=1) > 1.0
    velocity = env.scene[asset_cfg.name].data.body_lin_vel_w.torch[:, asset_cfg.body_ids, :2]
    return (velocity.norm(dim=-1) * contact).sum(-1)


def no_jump(env, sensor_cfg: SceneEntityCfg | None = None) -> torch.Tensor:
    """Penalize simultaneous flight of both feet using the task's current contact clocks.

    Rigid terrain uses the selected contact sensor; coupled terrain uses foot clocks
    derived from combined rigid and particle reaction forces. This matches the current
    foot-contact labels used by the differentiable reward, without a force-history delay.
    """
    if sensor_cfg is None:
        contact_time = env.foot_contact_time
    else:
        contact_time = env.scene.sensors[sensor_cfg.name].data.current_contact_time.torch[:, sensor_cfg.body_ids]
    return (contact_time <= 0).all(dim=-1).float()
