# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The mixed tow environment for towed objects that differ between environments."""

from __future__ import annotations

import numpy as np
import torch

from isaaclab.cloner.cloner_strategies import sequential

from .tow_mixed_env import G1TowMixedEnv
from .tow_objects_env_cfg import OBJECT_NAMES, TETHER_OFFSETS, G1TowObjectsEnvCfg


class G1TowObjectsEnv(G1TowMixedEnv):
    """The mixed tow environment with a box, a sphere or a wheel behind each robot.

    The tether is tied to each object at its own attachment point, and the towed object's speed is logged per
    object.
    """

    cfg: G1TowObjectsEnvCfg

    def load_managers(self) -> None:
        super().load_managers()
        # The clone plan lists the sand environments first, then the rigid ones, each split evenly over the
        # object variants in the spawner's order.
        fraction = self.scene.terrain.cfg.sand_env_fraction
        count = len(OBJECT_NAMES)
        weights = np.repeat([fraction, 1.0 - fraction], count) / count
        names = [OBJECT_NAMES[index % count] for index in sequential(weights, self.num_envs)]
        self.tow_object_names: list[str] = names
        """Name of the towed object of each environment."""
        self.tether.set_object_offsets(torch.tensor([TETHER_OFFSETS[name] for name in names], device=self.device))
        self._object_masks = {
            name: torch.tensor([n == name for n in names], device=self.device) for name in TETHER_OFFSETS
        }

    def step(self, action: torch.Tensor):
        obs, reward, terminated, truncated, extras = super().step(action)
        speed = torch.linalg.norm(self._tow_object.data.root_lin_vel_w.torch[:, :2], dim=-1)
        sand = self.scene.terrain.mpm_env_mask
        log = extras["log"]
        for name, mask in self._object_masks.items():
            for ground, ground_mask in (("sand", sand), ("rigid", ~sand)):
                both = mask & ground_mask
                if both.any():
                    log[f"Metrics/tow/sled_speed_{ground}_{name}"] = speed[both].mean()
        return obs, reward, terminated, truncated, extras
