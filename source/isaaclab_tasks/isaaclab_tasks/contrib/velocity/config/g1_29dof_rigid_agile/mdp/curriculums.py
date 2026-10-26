# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from typing import TYPE_CHECKING, TypedDict

import numpy as np
import torch

from isaaclab.assets import Articulation
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def terrain_levels_vel(
    env: ManagerBasedRLEnv, env_ids: Sequence[int], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Curriculum based on the distance the robot walked when commanded to move at a desired velocity.

    This term is used to increase the difficulty of the terrain when the robot walks far enough and decrease the
    difficulty when the robot walks less than half of the distance required by the commanded velocity.

    .. note::
        It is only possible to use this term with the terrain type ``generator``. For further information
        on different terrain types, check the :class:`isaaclab.terrains.TerrainImporter` class.

    Returns:
        The mean terrain level for the given environment ids.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain  # type: ignore
    command = env.command_manager.get_command("base_velocity")
    # compute the distance the robot walked
    distance = torch.norm(asset.data.root_pos_w.torch[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    # robots that walked far enough progress to harder terrains
    move_up = distance > terrain.cfg.terrain_generator.size[0] / 2
    # robots that walked less than half of their required distance go to simpler terrains
    move_down = distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
    move_down *= ~move_up
    # update terrain levels
    terrain.update_env_origins(env_ids, move_up, move_down)  # type: ignore
    # return the mean terrain level
    return torch.mean(terrain.terrain_levels.float())


"""
command curriculums
"""


class VelocityStage(TypedDict):
    step: int
    lin_vel_x: tuple[float, float] | None
    lin_vel_y: tuple[float, float] | None
    ang_vel_z: tuple[float, float] | None


def commands_vel(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    command_name: str,
    velocity_stages: list[VelocityStage],
) -> dict[str, torch.Tensor]:
    """
    Curriculum that updates the command velocity ranges based on predefined learning iterations.
    Example:
        "velocity_stages": [
          {"step": 0, "lin_vel_x": (-1.0, 1.0), "ang_vel_z": (-0.5, 0.5)},
          {"step": 5000 * 24, "lin_vel_x": (-1.5, 2.0), "ang_vel_z": (-0.7, 0.7)},
          {"step": 10000 * 24, "lin_vel_x": (-2.0, 3.0)},
        ],
    """
    del env_ids  # Unused.
    command_term = env.command_manager.get_term(command_name)
    assert command_term is not None
    cfg = command_term.cfg
    for stage in velocity_stages:
        if env.common_step_counter > stage["step"]:
            if "lin_vel_x" in stage and stage["lin_vel_x"] is not None:
                cfg.ranges.lin_vel_x = stage["lin_vel_x"]  # type: ignore
            if "lin_vel_y" in stage and stage["lin_vel_y"] is not None:
                cfg.ranges.lin_vel_y = stage["lin_vel_y"]  # type: ignore
            if "ang_vel_z" in stage and stage["ang_vel_z"] is not None:
                cfg.ranges.ang_vel_z = stage["ang_vel_z"]  # type: ignore
    return {
        "lin_vel_x_min": torch.tensor(cfg.ranges.lin_vel_x[0]),  # type: ignore
        "lin_vel_x_max": torch.tensor(cfg.ranges.lin_vel_x[1]),  # type: ignore
        "lin_vel_y_min": torch.tensor(cfg.ranges.lin_vel_y[0]),  # type: ignore
        "lin_vel_y_max": torch.tensor(cfg.ranges.lin_vel_y[1]),  # type: ignore
        "ang_vel_z_min": torch.tensor(cfg.ranges.ang_vel_z[0]),  # type: ignore
        "ang_vel_z_max": torch.tensor(cfg.ranges.ang_vel_z[1]),  # type: ignore
    }


"""
reward weight curriculums
"""


def modify_reward_std(env: ManagerBasedRLEnv, env_ids: Sequence[int], term_name: str, std: float, num_steps: int):
    """Curriculum that modifies a exponential reward std a given number of steps.

    Args:
        env: The learning environment.
        env_ids: Not used since all environments are affected.
        term_name: The name of the reward term.
        std: The std of the exponential reward term.
        num_steps: The number of steps after which the change should be applied.
    """
    if env.common_step_counter > num_steps:
        # obtain term settings
        term_cfg = env.reward_manager.get_term_cfg(term_name)
        # update term settings
        term_cfg.params["std"] = std
        env.reward_manager.set_term_cfg(term_name, term_cfg)


def ramp_reward_weight(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    term_name: str,
    weight_0: float,
    weight_1: float,
    step_0: int,
    step_1: int,
) -> float:
    """Curriculum that continuously sets the reward weight. The weight is `weight_0` before
    `step_0`, then linearly ramps to `weight_1` at `step_1`, and is `weight_1 after `step_1`.

    This overwrites the initial weight of the term and will set the absolute value of the
    weight at every environment step so it will not play nice with other curriculum terms
    on the same reward.

    Args:
        env: The learning environment.
        env_ids: Not used since all environments are affected.
        term_name: The name of the reward term.
        weight_0: The weight of the reward term below `step_0`.
        weight_1: The weight of the reward term above `step_1`.
        step_0: The step at which the weight will start to ramp.
        step_1: The step at which the weight will reach `weight_1`.
    """
    if step_0 == step_1:
        alpha = 0.0 if env.common_step_counter < step_0 else 1.0
    else:
        alpha = (env.common_step_counter - step_0) / (step_1 - step_0)
        alpha = np.clip(alpha, a_min=0.0, a_max=1.0)

    term_cfg = env.reward_manager.get_term_cfg(term_name)
    term_cfg.weight = (1.0 - alpha) * weight_0 + alpha * weight_1
    env.reward_manager.set_term_cfg(term_name, term_cfg)

    return env.reward_manager.get_term_cfg(term_name).weight


def modify_reward_param(
    env: ManagerBasedRLEnv, env_ids: Sequence[int], term_name: str, param_name: str, param_val: float, num_steps: int
):
    """Curriculum that modifies a parameter value after a given number of steps.

    Args:
        env: The learning environment.
        env_ids: Not used since all environments are affected.
        term_name: The name of the reward term.
        param_name: The name of the parameter.
        param_val: The value of the parameter.
        num_steps: The number of steps after which the change should be applied.
    """
    if env.common_step_counter > num_steps:
        # obtain term settings
        term_cfg = env.reward_manager.get_term_cfg(term_name)
        # update term settings
        term_cfg.params[param_name] = param_val
        env.reward_manager.set_term_cfg(term_name, term_cfg)

    return env.reward_manager.get_term_cfg(term_name).params[param_name]


def ramp_reward_param(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    term_name: str,
    param_name: str,
    val_0: float,
    val_1: float,
    step_0: int,
    step_1: int,
) -> float:
    """Curriculum that continuously sets the reward parameter. The value is `val_0` before
    `step_0`, then linearly ramps to `val_1` at `step_1`, and is `val_1` after `step_1`.

    This overwrites the initial value of the term parameter and will set the absolute value of the
    parameter at every environment step so it will not play nice with other curriculum terms
    on the same reward.

    Args:
        env: The learning environment.
        env_ids: Not used since all environments are affected.
        term_name: The name of the reward term.
        val_0: The value of the reward term below `step_0`.
        val_1: The value of the reward term above `step_1`.
        step_0: The step at which the value will start to ramp.
        step_1: The step at which the value will reach `val_1`.
    """

    if step_0 == step_1:
        alpha = 0.0 if env.common_step_counter < step_0 else 1.0
    else:
        alpha = (env.common_step_counter - step_0) / (step_1 - step_0)
        alpha = np.clip(alpha, a_min=0.0, a_max=1.0)

    term_cfg = env.reward_manager.get_term_cfg(term_name)
    term_cfg.params[param_name] = (1.0 - alpha) * val_0 + alpha * val_1
    env.reward_manager.set_term_cfg(term_name, term_cfg)

    return env.reward_manager.get_term_cfg(term_name).params[param_name]  # type: ignore


class terrain_levels_vel_curriculum(ManagerTermBase):
    """Curriculum based on the distance the robot walked when commanded to move at a desired velocity."""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.num_failures = torch.zeros(env.num_envs, device=env.device, dtype=torch.int32)
        self.num_successes = torch.zeros(env.num_envs, device=env.device, dtype=torch.int32)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor | slice,
        command_name: str = "base_velocity",
        move_up_distance: float = 6.0,
        move_down_distance: float = 3.0,
        n_failures: int = 3,
        n_successes: int = 3,
        p_random_move_up: float = 0.0,
        p_random_move_down: float = 0.0,
    ) -> torch.Tensor:
        """Curriculum based on the distance the robot walked when commanded to move at a desired velocity.

        This term is used to increase the difficulty of the terrain when the robot walks far enough and decrease the
        difficulty when the robot walks less than half of the distance required by the commanded velocity.

        .. note::
            It is only possible to use this term with the terrain type ``generator``. For further information
            on different terrain types, check the :class:`isaaclab.terrains.TerrainImporter` class.

        Returns:
            The mean terrain level for the given environment ids.
        """
        env_ids = torch.arange(env.num_envs, device=env.device)[env_ids] if isinstance(env_ids, slice) else env_ids
        # extract the used quantities (to enable type-hinting)
        terrain: TerrainImporter = env.scene.terrain

        traveled_distance = env.command_manager._terms[command_name].metrics["traveled_distance"][env_ids]

        # move up if the robot has traveled far enou  gh
        succeeded = traveled_distance > move_up_distance
        self.num_successes[env_ids] += succeeded

        # move down if the robot has failed too many times
        failed = traveled_distance < move_down_distance
        self.num_failures[env_ids] += failed

        move_up = self.num_successes[env_ids] >= n_successes
        move_down = self.num_failures[env_ids] >= n_failures

        # reset the number of successes and failures when the robot moves up or down
        self.num_failures[env_ids[move_up | move_down]] = 0.0
        self.num_successes[env_ids[move_up | move_down]] = 0.0

        # add random move up and down
        if p_random_move_up > 0.0:
            random_move_up = (torch.rand(env_ids.shape, device=env.device) < p_random_move_up) & ~move_down
            move_up = move_up | random_move_up
        if p_random_move_down > 0.0:
            random_move_down = (torch.rand(env_ids.shape, device=env.device) < p_random_move_down) & ~move_up
            move_down = move_down | random_move_down

        terrain.update_env_origins(env_ids, move_up, move_down)
        return torch.mean(terrain.terrain_levels.float())


class update_reward_weight_step(ManagerTermBase):
    """Curriculum to update reward weights given the iteration."""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        reward_name = cfg.params["reward_name"]
        if not isinstance(reward_name, str):
            raise ValueError(f"reward_name must be a string, got {type(reward_name)}")
        self.start_weight: float = env.reward_manager.get_term_cfg(reward_name).weight

        # Validate log space parameters if use_log_space is enabled
        use_log_space = cfg.params["use_log_space"]
        if use_log_space:
            terminal_weight = cfg.params["terminal_weight"]

            # Check that start and terminal weights have the same sign
            if (self.start_weight > 0) != (terminal_weight > 0):
                raise ValueError(
                    f"For log space scaling, start_weight ({self.start_weight}) and "
                    f"terminal_weight ({terminal_weight}) must have the same sign"
                )

            if self.start_weight == 0 or terminal_weight == 0:
                raise ValueError(
                    f"For log space scaling, weights cannot be zero. "
                    f"start_weight={self.start_weight}, terminal_weight={terminal_weight}"
                )

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],  # noqa: ARG002
        reward_name: str,
        start_step: int,
        num_steps: int,
        terminal_weight: float,
        use_log_space: bool = False,
    ) -> float:
        """Curriculum that changes the reward weight linearly or logarithmically given number of steps.

        Args:
            env: The learning environment.
            env_ids: Not used since all environments are affected.
            reward_name: reward to update
            start_step: when to start updating
            num_steps: how long to update
            terminal_weight: reward weight after curriculum is finished
            use_log_space: if True, change weight magnitude logarithmically instead of linearly.
                          Both start_weight and terminal_weight must have the same sign.
        """
        if env.common_step_counter <= start_step:
            return self.start_weight
        elif env.common_step_counter > start_step + num_steps:
            env.reward_manager.get_term_cfg(reward_name).weight = terminal_weight
            return terminal_weight
        else:
            scale = (env.common_step_counter - start_step) / num_steps

            if use_log_space:
                # Work with absolute values for log space interpolation
                abs_start = abs(self.start_weight)
                abs_terminal = abs(terminal_weight)

                # Interpolate in log space
                log_start = math.log(abs_start)
                log_terminal = math.log(abs_terminal)
                log_weight = log_start + scale * (log_terminal - log_start)
                abs_new_weight = math.exp(log_weight)

                # Apply the original sign
                new_weight = abs_new_weight if self.start_weight > 0 else -abs_new_weight
            else:
                # Linear interpolation (original behavior)
                new_weight = self.start_weight + (terminal_weight - self.start_weight) * scale

            env.reward_manager.get_term_cfg(reward_name).weight = new_weight
            return new_weight


class ContactRewardStage:
    def __init__(
        self,
        reward_manager,
        *,
        term_name,
        final_weight,
        initial_fraction=0.1,
        start_seconds,
        ramp_seconds,
    ):
        if (
            not isinstance(term_name, str)
            or not term_name.startswith("smooth_")
            or type(final_weight) not in (int, float)
            or not math.isfinite(final_weight)
            or final_weight >= 0
            or type(initial_fraction) not in (int, float)
            or not math.isfinite(initial_fraction)
            or not 0 < initial_fraction <= 1
        ):
            raise ValueError("Invalid contact reward stage")
        if (
            type(start_seconds) not in (int, float)
            or not math.isfinite(start_seconds)
            or start_seconds < 0
            or type(ramp_seconds) not in (int, float)
            or not math.isfinite(ramp_seconds)
            or ramp_seconds <= 0
        ):
            raise ValueError("Invalid contact reward time schedule")
        self.manager = reward_manager
        self.term_name = term_name
        self.final_weight = float(final_weight)
        self.initial_fraction = float(initial_fraction)
        self.start_seconds, self.ramp_seconds = start_seconds, ramp_seconds
        self.clock_origin = self.elapsed_seconds = None
        self.progress = 0.0
        self.completed_steps = None
        self.weight = self.final_weight * self.initial_fraction
        if self.manager.get_term_cfg(term_name).weight != self.weight:
            raise ValueError("Native contact reward must start at declared initial weight")

    def start_clock(self, origin):
        """Bind once at the runner's training-loop start, excluding setup/queue time."""
        if (
            type(origin) not in (int, float)
            or not math.isfinite(origin)
            or self.clock_origin is not None
            or self.completed_steps is not None
        ):
            raise ValueError("Contact curriculum clock must be bound once before collection")
        self.clock_origin = origin

    def before_collection(self, completed_steps):
        if (
            type(completed_steps) is not int
            or completed_steps < 0
            or self.completed_steps is not None
            and completed_steps <= self.completed_steps
        ):
            raise ValueError("Contact stage requires strictly increasing completed collection steps")
        if self.clock_origin is None:
            raise ValueError("Bind contact curriculum clock before collection")
        elapsed = time.perf_counter() - self.clock_origin
        if (
            not math.isfinite(elapsed)
            or elapsed < 0
            or self.elapsed_seconds is not None
            and elapsed < self.elapsed_seconds
        ):
            raise ValueError("Contact curriculum requires a monotonic elapsed clock")
        self.elapsed_seconds = elapsed
        fraction = min(max((elapsed - self.start_seconds) / self.ramp_seconds, 0.0), 1.0)
        scale = self.initial_fraction ** (1 - fraction)
        weight = self.final_weight * scale
        cfg = self.manager.get_term_cfg(self.term_name)
        if cfg.weight != self.weight:
            raise ValueError("Contact weight was changed outside its declared stage")
        if weight != self.weight:
            cfg.weight = weight
            self.manager.set_term_cfg(self.term_name, cfg)
        self.weight, self.completed_steps = weight, completed_steps
        self.progress = fraction
        return weight

    def state(self):
        return {
            "term_name": self.term_name,
            "weight": self.weight,
            "final_weight": self.final_weight,
            "fraction": self.weight / self.final_weight,
            "completed_steps_before_collection": self.completed_steps,
            "clock": "training_wall_seconds",
            "elapsed_seconds": self.elapsed_seconds,
            "start_seconds": self.start_seconds,
            "ramp_seconds": self.ramp_seconds,
            "interpolation": "logarithmic",
            "ramp_progress": self.progress,
            "initial_weight": self.final_weight * self.initial_fraction,
            "multiplier": self.weight / (self.final_weight * self.initial_fraction),
            "semantics": "Weight frozen before collection and its updates; stored n-step costs relabeled",
        }


class ForwardCommandStage:
    def __init__(
        self,
        term,
        *,
        initial_max=0.5,
        final_max=1.0,
        start_fraction=1 / 3,
        ramp_fraction=1 / 3,
    ):
        values = (initial_max, final_max, start_fraction, ramp_fraction)
        if (
            any(type(v) not in (int, float) or not math.isfinite(v) for v in values)
            or not 0 < initial_max < final_max
            or not 0 <= start_fraction < 1
            or not 0 < ramp_fraction <= 1 - start_fraction
        ):
            raise ValueError("Invalid forward command schedule")
        if type(term).__name__ != "UniformNullVelocityCommand":
            raise ValueError("Expected native UniformNullVelocityCommand sampling")
        self.term = term
        self.initial_max, self.final_max = initial_max, final_max
        self.start_fraction, self.ramp_fraction = start_fraction, ramp_fraction
        self.maximum = initial_max
        self.origin = self.duration = self.elapsed = self.completed_steps = None
        self.progress = 0.0
        self._validate_ranges()

    def _validate_ranges(self):
        ranges = self.term.cfg.ranges
        if (
            tuple(ranges.lin_vel_x) != (-0.5, self.maximum)
            or tuple(ranges.lin_vel_y) != (-0.5, 0.5)
            or tuple(ranges.ang_vel_z) != (-1.0, 1.0)
            or self.term.cfg.heading_command
        ):
            raise ValueError("Unexpected command bounds or external command-range mutation")

    def start_clock(self, origin, duration_seconds):
        if (
            self.origin is not None
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in (origin, duration_seconds))
            or duration_seconds <= 0
        ):
            raise ValueError("Bind command clock once to a positive remaining training budget")
        self.origin, self.duration = origin, duration_seconds

    def before_collection(self, completed_steps):
        if (
            self.origin is None
            or type(completed_steps) is not int
            or completed_steps < 0
            or self.completed_steps is not None
            and completed_steps <= self.completed_steps
        ):
            raise ValueError("Bind clock before strictly increasing command-stage collection steps")
        elapsed = time.perf_counter() - self.origin
        if not math.isfinite(elapsed) or elapsed < 0 or self.elapsed is not None and elapsed < self.elapsed:
            raise ValueError("Command curriculum requires monotonic elapsed time")
        self._validate_ranges()
        progress = min(max((elapsed / self.duration - self.start_fraction) / self.ramp_fraction, 0.0), 1.0)
        maximum = self.initial_max + progress * (self.final_max - self.initial_max)
        # The native term reads these bounds on its next reset or timed resample.
        # Never overwrite the active command, observation history, or stored replay.
        self.term.cfg.ranges.lin_vel_x = (-0.5, maximum)
        self.maximum, self.progress = maximum, progress
        self.elapsed, self.completed_steps = elapsed, completed_steps
        return maximum

    def state(self):
        result = {
            "clock": "training_wall_seconds",
            "elapsed_seconds": self.elapsed,
            "training_budget_seconds": self.duration,
            "start_fraction": self.start_fraction,
            "ramp_fraction": self.ramp_fraction,
            "initial_max": self.initial_max,
            "final_max": self.final_max,
            "lin_vel_x_min": -0.5,
            "lin_vel_x_max": self.maximum,
            "lin_vel_y_min": -0.5,
            "lin_vel_y_max": 0.5,
            "ang_vel_z_min": -1.0,
            "ang_vel_z_max": 1.0,
            "ramp_progress": self.progress,
            "completed_steps_before_collection": self.completed_steps,
            "semantics": (
                "Linear range expansion at native reset/timed resampling; existing commands and replay unchanged"
            ),
        }
        return result


class ReplayRewardCurriculum:
    """Agile's collection-time curricula and sample-time n-step reward relabeling.

    Captures reward-manager components without reevaluating stateful rewards.
    Weights are captured before stepping because reset curricula can change them
    after the reward has been computed. Replay samples use the current weights.
    """

    sum_keys = ("rewards", "stationary_reward", "smoothing_terms", "contact_terms")

    def __init__(self, env, command_curriculum_seconds: float | None = None):
        self.env = env.unwrapped
        self.manager = self.env.reward_manager
        self.command_curriculum_seconds = command_curriculum_seconds
        self.smoothing_names = ("action_rate", "action_rate_rate")
        self.smoothing_indices = [self.manager.active_terms.index(name) for name in self.smoothing_names]
        self.contact_index = self.manager.active_terms.index("smooth_interval_loading")
        self.contact = ContactRewardStage(
            self.manager,
            term_name="smooth_interval_loading",
            final_weight=-5.0,
            initial_fraction=0.1,
            start_seconds=1200.0,
            ramp_seconds=1200.0,
        )
        self.command = ForwardCommandStage(self.env.command_manager.get_term("base_velocity"))

    def start(self, remaining_seconds: float | None) -> None:
        duration = self.command_curriculum_seconds or remaining_seconds
        if duration is None or duration <= 0:
            raise ValueError("Agile command curriculum requires a positive training time budget.")
        origin = time.perf_counter()
        self.contact.start_clock(origin)
        self.command.start_clock(origin, duration)

    def before_collection(self, collection_step: int) -> None:
        self.command.before_collection(collection_step)
        self.contact.before_collection(collection_step)
        self.collection_weights = tuple(self.manager.get_term_cfg(name).weight for name in self.smoothing_names)

    def process_transition(self, transition):
        weighted = self.manager._step_reward[:, self.smoothing_indices].detach() * self.env.step_dt
        contact = self.manager._step_reward[:, self.contact_index].detach() * self.env.step_dt
        transition["stationary_reward"] = transition["rewards"] - weighted.sum(-1) - contact
        transition["smoothing_terms"] = weighted / weighted.new_tensor(self.collection_weights)
        transition["contact_terms"] = contact / self.contact.weight
        return transition

    def process_batch(self, batch):
        weights = tuple(self.manager.get_term_cfg(name).weight for name in self.smoothing_names)
        terms = batch["smoothing_terms"]
        batch["rewards"] = (
            batch["stationary_reward"]
            + terms[:, 0] * weights[0]
            + terms[:, 1] * weights[1]
            + batch["contact_terms"] * self.contact.weight
        )
        return batch
