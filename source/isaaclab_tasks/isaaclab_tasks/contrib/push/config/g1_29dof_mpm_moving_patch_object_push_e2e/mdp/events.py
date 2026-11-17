# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Box reset on the sand surface, the box mass curriculum, and the training-only assistive push force."""

import torch
import warp as wp

from isaaclab.managers import ManagerTermBase
from isaaclab.utils.math import quat_apply, quat_from_euler_xyz, quat_mul, yaw_quat

from isaaclab_tasks.contrib.push.config.g1_29dof_object_push_e2e.mdp import progress_ratio


def reset_box_on_sand(
    env,
    env_ids,
    distance_range: tuple[float, float],
    lateral_range: tuple[float, float],
    yaw_range: tuple[float, float],
    object_name: str = "box",
):
    """Sample an upright box pose relative to the newly reset robot, resting on the sand.

    Same sampling as the rigid-ground ``reset_box``; only the height differs: the box bottom
    starts 1 cm above the undisturbed sand surface under its center.

    Args:
        env: The environment.
        env_ids: Environments to reset.
        distance_range: Box center distance ahead of the pelvis along the robot heading [m].
        lateral_range: Box center offset to the left of the robot heading [m].
        yaw_range: Box yaw relative to the robot heading [rad].
        object_name: Scene name of the box.
    """
    box, robot = env.scene[object_name], env.scene["robot"]
    position = robot.data.root_pos_w.torch[env_ids]
    orientation = yaw_quat(robot.data.root_quat_w.torch[env_ids])
    offset = torch.zeros_like(position)
    offset[:, 0].uniform_(*distance_range)
    offset[:, 1].uniform_(*lateral_range)
    yaw = torch.empty(len(position), device=position.device, dtype=position.dtype).uniform_(*yaw_range)
    zeros = torch.zeros_like(yaw)
    box_position = (position + quat_apply(orientation, offset)).contiguous()
    # The terrain chart returns the undisturbed sand surface, not the floor underneath.
    points = wp.from_torch(box_position, dtype=wp.vec3)
    env.scene.terrain.background_mesh.sample_surface_heights(points)
    box_position = wp.to_torch(points).clone()
    box_position[:, 2] += getattr(env.cfg.scene, object_name).spawn.size[2] / 2 + 0.01
    pose = torch.cat((box_position, quat_mul(orientation, quat_from_euler_xyz(zeros, zeros, yaw))), dim=-1)
    box.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
    box.write_root_velocity_to_sim_index(root_velocity=torch.zeros((len(pose), 6), device=env.device), env_ids=env_ids)


class AssistivePushForce(ManagerTermBase):
    r"""Training-only horizontal force on the robot base that helps it push, retired as the box follows the command.

    The force follows the assistive wrench of ``mimic.mdp.AssistiveWrench`` (ZEST,
    https://arxiv.org/abs/2602.00401), reduced to the one quantity this task commands: the planar base
    velocity. Along the commanded direction :math:`\hat{d}` in the world frame,

    .. math::

        F = \beta \, \min(m \, k_d \, \max(v_{cmd} - v \cdot \hat{d}, 0), F_{max}) \, \hat{d}

    with :math:`m` the robot mass and :math:`v` the base velocity. The force never brakes, never acts
    sideways or vertically, and is zero for standing commands.

    The gain :math:`\beta = \mathrm{clip}(1 - s / \eta, 0, \beta_{max})` is frozen per environment at
    reset. :math:`s` is a running mean, over finished episodes of this process, of the episode-mean box
    speed ratio (see :func:`progress_ratio`), so the force fades as the box tracks the command and is
    gone once the ratio reaches ``eta``. The ratio is measured with the force applied.

    Register as an ``interval`` event with ``interval_range_s=(0.0, 0.0)`` and ``is_global_time=True``
    so it runs every environment step on all environments.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._robot = env.scene["robot"]
        self._box = env.scene[cfg.params.get("object_name", "box")]
        self._body_id = self._robot.body_names.index(cfg.params.get("body_name", "pelvis"))
        mass = self._robot.data.body_mass
        mass = mass.torch if hasattr(mass, "torch") else mass
        self._mass = mass.sum(dim=1).to(self.device)
        self._ratio_sum = torch.zeros(self.num_envs, device=self.device)
        self._moving_steps = torch.zeros(self.num_envs, device=self.device)
        self._forces = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._torques = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self.similarity = 0.0
        """Running mean of the episode-mean box speed ratio over finished episodes."""
        self.beta = torch.full((self.num_envs,), self._gain(), device=self.device)
        """Per-environment gain, frozen at reset."""
        self.force = torch.zeros(self.num_envs, device=self.device)
        """Force magnitude applied in the last step [N]."""

    def _gain(self) -> float:
        params = self.cfg.params
        return min(max(1.0 - self.similarity / params["eta"], 0.0), params["beta_max"])

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        steps = self._moving_steps[ids]
        finished = steps > 0
        count = int(finished.sum())
        if count:
            episode_ratio = (self._ratio_sum[ids][finished] / steps[finished]).mean().item()
            # One update per finished episode, so the time constant does not depend on batch size.
            weight = 1.0 - (1.0 - self.cfg.params.get("similarity_rate", 0.01)) ** count
            self.similarity += weight * (episode_ratio - self.similarity)
        self.beta[ids] = self._gain()
        self._ratio_sum[ids] = 0.0
        self._moving_steps[ids] = 0.0

    def __call__(
        self,
        env,
        env_ids,
        command_name: str,
        eta: float,
        beta_max: float,
        k_d: float,
        max_force: float,
        command_threshold: float,
        similarity_rate: float = 0.01,
        object_name: str = "box",
        body_name: str = "pelvis",
    ):
        """Apply the force for this step and accumulate the box speed ratio.

        Args:
            env: The environment.
            env_ids: Unused; the term acts on all environments.
            command_name: Velocity command the force assists.
            eta: Box speed ratio at which the force is fully retired.
            beta_max: Upper bound of the gain.
            k_d: Velocity-error gain [1/s]; the force is the robot mass times this times the error.
            max_force: Bound on the force before the gain is applied [N].
            command_threshold: Planar command speed below which no force is applied [m/s].
            similarity_rate: Weight of one finished episode in the running box speed ratio.
            object_name: Scene name of the box.
            body_name: Robot body the force acts on.
        """
        command = env.command_manager.get_command(command_name)[:, :2]
        speed = torch.linalg.vector_norm(command, dim=-1)
        moving = speed > command_threshold
        robot_quat = self._robot.data.root_quat_w.torch
        command_w = quat_apply(yaw_quat(robot_quat), torch.cat((command, torch.zeros_like(command[:, :1])), dim=-1))
        direction = command_w / speed.clamp(min=command_threshold).unsqueeze(-1)
        error = speed - (self._robot.data.root_lin_vel_w.torch * direction).sum(dim=-1)
        self.force = self.beta * (self._mass * k_d * error).clamp(0.0, max_force) * moving
        self._forces[:, 0] = self.force.unsqueeze(-1) * direction
        self._robot.permanent_wrench_composer.set_forces_and_torques_index(
            forces=self._forces, torques=self._torques, body_ids=[self._body_id], is_global=True
        )

        self._ratio_sum += progress_ratio(command, robot_quat, self._box.data.root_lin_vel_w.torch, command_threshold)
        self._moving_steps += moving


class BoxMassCurriculum(ManagerTermBase):
    """Resample the box mass at reset from a range that widens as the box follows the command.

    ``level`` in [0, 1] interpolates the mass range from ``start_range`` to ``final_range``. Each
    finished episode with a moving command raises the level by ``level_step`` when its mean box
    speed ratio exceeds ``promote_ratio`` and lowers it by the same amount below ``demote_ratio``.
    The level belongs to this process; distributed workers each keep their own.

    Register as a ``reset`` event. Reset events run before the command term is reset, which is when
    the finished episode's ratio is still available.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._box = env.scene[cfg.params.get("object_name", "box")]
        self._default_mass = None
        self._default_inertia = None
        self.level = 0.0
        """Curriculum progress in [0, 1]."""
        self.mass_range = tuple(cfg.params["start_range"])
        """Mass range currently sampled [kg]."""

    def __call__(
        self,
        env,
        env_ids,
        command_name: str,
        start_range: tuple[float, float],
        final_range: tuple[float, float],
        promote_ratio: float,
        demote_ratio: float,
        level_step: float,
        object_name: str = "box",
    ):
        """Update the level from the finished episodes and assign new masses.

        Args:
            env: The environment.
            env_ids: Environments being reset.
            command_name: Velocity command whose box speed ratio drives the level.
            start_range: Mass range at level 0 [kg].
            final_range: Mass range at level 1 [kg].
            promote_ratio: Episode box speed ratio above which the level rises.
            demote_ratio: Episode box speed ratio below which the level falls.
            level_step: Level change per finished episode.
            object_name: Scene name of the box.
        """
        if self._default_mass is None:
            self._default_mass = self._box.data.body_mass.torch.clone()
            self._default_inertia = self._box.data.body_inertia.torch.clone()
        ratio, valid = env.command_manager.get_term(command_name).episode_box_speed_ratio(env_ids)
        change = ((ratio > promote_ratio) & valid).sum() - ((ratio < demote_ratio) & valid).sum()
        self.level = min(max(self.level + level_step * int(change), 0.0), 1.0)
        self.mass_range = tuple(a + self.level * (b - a) for a, b in zip(start_range, final_range))

        default_mass = self._default_mass[env_ids]
        masses = torch.empty_like(default_mass).uniform_(*self.mass_range)
        self._box.set_masses_index(masses=masses, env_ids=env_ids)
        # Uniform-density box: inertia scales with mass.
        inertias = self._default_inertia[env_ids] * (masses / default_mass).unsqueeze(-1)
        self._box.set_inertias_index(inertias=inertias, env_ids=env_ids)


def box_mass_curriculum_state(env, env_ids, term_name: str = "box_mass") -> dict[str, float]:
    """Report the box mass curriculum for logging; changes nothing.

    Args:
        env: The environment.
        env_ids: Unused.
        term_name: Name of the :class:`BoxMassCurriculum` event term.

    Returns:
        The level, the upper bound of the sampled mass range [kg], and the mean box mass [kg].
    """
    term = env.event_manager.get_term_cfg(term_name).func
    box = env.scene[term.cfg.params.get("object_name", "box")]
    return {"level": term.level, "max_mass": term.mass_range[1], "mean_mass": box.data.body_mass.torch.mean().item()}


def assistive_push_force_state(env, env_ids, term_name: str = "assistive_push_force") -> dict[str, float]:
    """Report the assistive force state for logging; changes nothing.

    Args:
        env: The environment.
        env_ids: Unused.
        term_name: Name of the :class:`AssistivePushForce` event term.

    Returns:
        The mean gain, the running box speed ratio that sets it, and the mean applied force [N].
    """
    term = env.event_manager.get_term_cfg(term_name).func
    return {"beta": term.beta.mean().item(), "similarity": term.similarity, "force": term.force.mean().item()}
