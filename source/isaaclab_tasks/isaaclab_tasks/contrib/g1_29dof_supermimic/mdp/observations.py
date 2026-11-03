# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Physical state and labels from the coupled solver, in articulation body order."""

import torch

from isaaclab.managers import ManagerTermBase


class VelocityState(ManagerTermBase):
    """Use the PR physical state layout with coupled foot clocks."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        feet = env.cfg.rewards.feet_air_time.params["asset_cfg"]
        self._feet, _ = env.scene["robot"].find_bodies(feet.body_names, preserve_order=True)
        if env.cfg.rewards.feet_air_time.func.__name__ != "feet_air_time_positive_biped":
            raise ValueError("Coupled velocity state currently supports the positive-biped air-time reward.")

    def __call__(self, env):
        data = env.scene["robot"].data
        return torch.cat(
            (
                data.root_lin_vel_b.torch,
                data.root_ang_vel_b.torch,
                data.root_quat_w.torch,
                data.root_pos_w.torch[:, 2:3] - env.scene.env_origins[:, 2:3],
                data.joint_pos.torch,
                data.joint_vel.torch,
                data.body_lin_vel_w.torch[:, self._feet].flatten(1),
                env.foot_air_time,
                env.foot_contact_time,
                # The positive-biped reward uses current mode clocks, never last_air_time.
                torch.zeros_like(env.foot_air_time),
            ),
            -1,
        )


class VelocityLabels(ManagerTermBase):
    """Label current foot contact, body force history, and foot touchdown."""

    def __call__(self, env):
        env.update_contact_state()
        return torch.cat(
            (
                env.foot_contact_time > 0,
                env.body_contact_force_history.norm(dim=-1).amax(dim=1) > 1.0,
                env.foot_first_contact,
            ),
            -1,
        ).float()
