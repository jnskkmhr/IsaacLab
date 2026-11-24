# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Curriculum terms of the tow task."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import CurriculumTermCfg, ManagerTermBase, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject

    from ..tow_env import G1TowEnv


class sled_mass_levels(ManagerTermBase):
    """Raise the sled mass of a sand environment when the sled keeps up with the command, lower it when it stalls.

    Each sand environment has its own sled mass. At the end of an episode the path the sled travelled is compared
    with the path the velocity command asked for: above ``promote_ratio`` the mass goes up one step, below
    ``demote_ratio`` it goes down one step. Episodes that asked for less than ``min_commanded_distance`` leave the
    mass unchanged. Environments on rigid ground keep the configured sled mass.
    """

    def __init__(self, cfg: CurriculumTermCfg, env: G1TowEnv):
        super().__init__(cfg, env)
        self._sled: RigidObject = env.scene[cfg.params["asset_cfg"].name]
        self._default_mass = self._sled.data.body_mass.torch.clone()
        self._default_inertia = self._sled.data.body_inertia.torch.clone()
        # The environment may name the environments that follow the curriculum; otherwise the sand ones do.
        sand = getattr(env, "sled_mass_curriculum_mask", None)
        if sand is None:
            sand = getattr(env.scene.terrain, "mpm_env_mask", None)
        self.sand = torch.ones(env.num_envs, dtype=torch.bool, device=env.device) if sand is None else sand.clone()
        """Environments whose sled mass follows the curriculum, shape ``(num_envs,)``."""
        low, _ = cfg.params["mass_range"]
        self.mass = self._default_mass[:, 0].clone()
        """Current sled mass [kg], shape ``(num_envs,)``."""
        self.mass[self.sand] = low
        self._write(torch.arange(env.num_envs, device=env.device))

    def _write(self, env_ids: torch.Tensor) -> None:
        scale = (self.mass[env_ids] / self._default_mass[env_ids, 0]).view(-1, 1)
        self._sled.set_masses_index(masses=self._default_mass[env_ids] * scale, env_ids=env_ids)
        self._sled.set_inertias_index(inertias=self._default_inertia[env_ids] * scale.unsqueeze(-1), env_ids=env_ids)

    def __call__(
        self,
        env: G1TowEnv,
        env_ids: Sequence[int] | torch.Tensor | slice,
        asset_cfg: SceneEntityCfg,
        mass_range: tuple[float, float],
        mass_step: float = 1.0,
        promote_ratio: float = 0.6,
        demote_ratio: float = 0.3,
        min_commanded_distance: float = 2.0,
    ) -> dict[str, float]:
        """Update the sled mass of the sand environments that are being reset.

        Args:
            env: Environment instance.
            env_ids: Environments being reset.
            asset_cfg: The sled.
            mass_range: Lowest and highest sled mass of the curriculum [kg].
            mass_step: Mass change per episode [kg].
            promote_ratio: Sled path over commanded path above which the mass goes up.
            demote_ratio: Sled path over commanded path below which the mass goes down.
            min_commanded_distance: Commanded path below which the episode is not judged [m].

        Returns:
            Mean and highest sled mass over the sand environments [kg].
        """
        if isinstance(env_ids, slice):
            env_ids = torch.arange(env.num_envs, device=env.device)[env_ids]
        env_ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
        ids = env_ids[self.sand[env_ids]]
        if ids.numel():
            commanded = env.commanded_path[ids]
            ratio = env.sled_path[ids] / commanded.clamp_min(1.0e-6)
            judged = commanded > min_commanded_distance
            change = torch.zeros_like(ratio)
            change[judged & (ratio > promote_ratio)] = mass_step
            change[judged & (ratio < demote_ratio)] = -mass_step
            self.mass[ids] = (self.mass[ids] + change).clamp(mass_range[0], mass_range[1])
            self._write(ids)
        sand_mass = self.mass[self.sand]
        return {"mean": sand_mass.mean().item(), "max": sand_mass.max().item()}
