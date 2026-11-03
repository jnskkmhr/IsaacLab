# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for Factory sampling and curriculum scheduling."""

from types import SimpleNamespace

import torch

from isaaclab.utils import instantiate

from isaaclab_tasks.contrib.nist.mdp.curriculums import DifficultyScheduler
from isaaclab_tasks.contrib.nist.utils import (
    BetaSamplingStrategyCfg,
    Sampler,
    SamplerCfg,
    UniformSamplingStrategyCfg,
)


def test_uniform_only_sampler_is_uniform():
    """A uniform-only sampler with no floor returns exact uniform probabilities."""
    rates = torch.rand(100)
    cfg = SamplerCfg(strategies=[UniformSamplingStrategyCfg(weight=1.0)], eps=0.0)
    probs = instantiate(cfg, rates).probabilities()
    torch.testing.assert_close(probs, torch.full_like(probs, 1.0 / 100))


def test_sampler_probabilities_are_finite_nonnegative_and_normalized():
    rates = torch.rand(50)
    cfg = SamplerCfg(
        strategies=[
            BetaSamplingStrategyCfg(target=0.66, kappa=1.0, weight=1.0),
            UniformSamplingStrategyCfg(weight=0.1),
        ],
        eps=1e-3,
    )
    sampler = instantiate(cfg, rates)
    assert sampler.names == ["beta", "uniform"]
    probs = sampler.probabilities()
    assert torch.isfinite(probs).all()
    assert (probs >= 0).all()
    torch.testing.assert_close(probs.sum(), torch.tensor(1.0))


def test_sampler_clamps_negative_strategy_weight():
    """A negative strategy weight contributes no probability mass."""
    rates = torch.rand(50)
    beta = BetaSamplingStrategyCfg(target=0.66, kappa=1.0, weight=1.0)
    beta_only = Sampler(SamplerCfg(strategies=[beta], eps=1e-6), rates).probabilities()
    with_negative = Sampler(
        SamplerCfg(strategies=[beta, UniformSamplingStrategyCfg(weight=-1.0)], eps=1e-6), rates
    ).probabilities()
    torch.testing.assert_close(with_negative, beta_only)


def test_difficulty_scheduler_waits_for_accumulator_rates():
    """Initial curriculum updates preserve difficulty until reset rates exist."""
    scheduler = DifficultyScheduler.__new__(DifficultyScheduler)
    scheduler.current_adr_difficulties = torch.ones(3) * 2
    scheduler.difficulty_frac = torch.tensor(0.2)
    reset_accumulator = SimpleNamespace(monitor_success_rate=None)
    env = SimpleNamespace(
        device=torch.device("cpu"),
        event_manager=SimpleNamespace(get_term_cfg=lambda _name: SimpleNamespace(func=reset_accumulator)),
    )

    result = DifficultyScheduler.__call__(
        scheduler,
        env,
        torch.arange(3),
        "env.event_manager.get_term_cfg('reset_strategies').func.monitor_success_rate",
        max_difficulty=10,
    )

    assert result is scheduler.difficulty_frac
    torch.testing.assert_close(scheduler.current_adr_difficulties, torch.ones(3) * 2)


def test_difficulty_scheduler_averages_ready_success_rates():
    """Curriculum updates average the reset bank's per-state rates."""
    scheduler = DifficultyScheduler.__new__(DifficultyScheduler)
    scheduler.current_adr_difficulties = torch.ones(3) * 2
    scheduler.difficulty_frac = torch.tensor(0.2)
    reset_accumulator = SimpleNamespace(monitor_success_rate=torch.ones(4))
    env = SimpleNamespace(
        device=torch.device("cpu"),
        event_manager=SimpleNamespace(get_term_cfg=lambda _name: SimpleNamespace(func=reset_accumulator)),
    )

    result = DifficultyScheduler.__call__(
        scheduler,
        env,
        torch.arange(3),
        "env.event_manager.get_term_cfg('reset_strategies').func.monitor_success_rate",
        max_difficulty=10,
    )

    torch.testing.assert_close(scheduler.current_adr_difficulties, torch.ones(3) * 3)
    torch.testing.assert_close(result, torch.tensor(0.3))


def test_flash_sac_environment_curricula_and_replay_relabeling():
    """Environment curricula own schedules; replay preserves pre-reset reward components."""
    import math

    from isaaclab.managers import CurriculumManager

    from isaaclab_tasks.contrib.velocity.config.g1_29dof_rigid_agile.env_cfg.curriculum_cfg import G1CurriculumCfg
    from isaaclab_tasks.contrib.velocity.config.g1_29dof_rigid_agile.mdp.curriculums import ReplayRewardRelabeling

    command = SimpleNamespace(cfg=SimpleNamespace(ranges=SimpleNamespace(lin_vel_x=(-0.5, 0.5))))
    terms = {
        "action_rate": SimpleNamespace(weight=-1.0),
        "action_rate_rate": SimpleNamespace(weight=-0.5),
        "smooth_interval_loading": SimpleNamespace(weight=-0.5),
    }
    manager = SimpleNamespace(
        active_terms=list(terms),
        get_term_cfg=terms.__getitem__,
        _step_reward=torch.tensor([[-2.0, -1.5, -2.0]]),
    )
    env = SimpleNamespace(
        reward_manager=manager,
        command_manager=SimpleNamespace(get_term=lambda name: command),
        step_dt=0.02,
        common_step_counter=0,
        sim=SimpleNamespace(is_playing=lambda: True),
    )
    cfg = G1CurriculumCfg()
    # Keep schedule mechanics independent of tuned training durations.
    cfg.command_vel.params.update(step_0=5000, step_1=10000)
    cfg.increase_contact_regularization.params.update(start_step=5000, num_steps=5000)
    curriculum = CurriculumManager(
        {"command_vel": cfg.command_vel, "contact": cfg.increase_contact_regularization}, env
    )
    replay = ReplayRewardRelabeling(SimpleNamespace(unwrapped=env))
    replay.before_collection(0)
    # Collection callbacks must never advance the environment's schedules.
    replay.before_collection(15000)
    assert command.cfg.ranges.lin_vel_x == (-0.5, 0.5)
    assert terms["smooth_interval_loading"].weight == -0.5
    for step, velocity, weight in (
        (4999, 0.5, -0.5),
        (5000, 0.5, -0.5),
        (7500, 0.75, -math.sqrt(2.5)),
        (10000, 1.0, -5.0),
        (15000, 1.0, -5.0),
    ):
        env.common_step_counter = step
        curriculum.compute()
        assert command.cfg.ranges.lin_vel_x == (-0.5, velocity)
        assert math.isclose(terms["smooth_interval_loading"].weight, weight)
    # Reset curricula ran after reward computation, before the runner received the transition.
    batch = replay.process_transition({"rewards": torch.tensor([1.0])})
    torch.testing.assert_close(batch["contact_terms"], torch.tensor([0.08]))
    torch.testing.assert_close(replay.process_batch(batch)["rewards"], torch.tensor([0.64]))
