# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Curriculum logging; reward tolerances stay at the rigid-ground policy's final values."""

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.utils import configclass

from .. import mdp as push_mdp


@configclass
class CurriculumCfg:
    """Logs the box mass curriculum, which adapts itself inside its reset event."""

    box_mass = CurrTerm(func=push_mdp.box_mass_curriculum_state)


@configclass
class AssistCurriculumCfg(CurriculumCfg):
    """Logs the assistive push force; the force adapts itself inside its event term."""

    assistive_push_force = CurrTerm(func=push_mdp.assistive_push_force_state)
