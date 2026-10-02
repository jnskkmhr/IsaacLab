# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg
from isaaclab.utils import configclass

from isaaclab_tasks.contrib.wbc.mdp import observations


@configclass
class G1ObservationsCfg:
    @configclass
    class TeacherCfg(ObservationGroupCfg):
        state = ObservationTermCfg(func=observations.teacher_observation)

    @configclass
    class StudentCfg(ObservationGroupCfg):
        state = ObservationTermCfg(func=observations.student_observation)

    @configclass
    class CriticCfg(ObservationGroupCfg):
        state = ObservationTermCfg(func=observations.critic_observation)

    teacher: TeacherCfg = TeacherCfg(enable_corruption=False, concatenate_terms=True)
    student: StudentCfg = StudentCfg(enable_corruption=False, concatenate_terms=True)
    critic: CriticCfg = CriticCfg(enable_corruption=False, concatenate_terms=True)
