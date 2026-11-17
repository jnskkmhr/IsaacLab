# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Differentiable Agile task with measured force/actuator dynamics and unchanged physical rewards.

The world model predicts physical quantities needed by Agile's rewards, including
applied/computed torques and substep foot forces. It does not predict scalar rewards.
Contact thresholds and terminations use supervised event probabilities. Action-rate
memory and loading validity advance analytically. Random arm targets are privileged
model state; the policy sees only Agile's original proprioceptive observations.
"""

from __future__ import annotations

import torch
from rsl_rl.algorithms.supermimic_task import DifferentiableTask, State

from isaaclab.managers import ManagerTermBase
from isaaclab.utils import math as math_utils

from . import rewards

# Each component is flat; quaternion components use SO(3) increments.
_FIELDS = (
    ("lin_vel", 3),
    ("ang_vel", 3),
    ("root_quat", 4),
    ("joint_pos", 29),
    ("joint_vel", 29),
    ("feet_pos", 6),
    ("feet_vel", 6),
    ("left_quat", 4),
    ("right_quat", 4),
    ("torso_quat", 4),
    ("height", 1),
    ("height_scan", 9),
    ("applied_torque", 29),
    ("computed_torque", 29),
    ("loading_forces", 10),
    ("arm_targets", 14),
    ("arm_destinations", 14),
    ("previous_world_velocity", 6),
    ("previous_action_rate", 15),
    ("loading_valid", 1),
    ("acc_valid", 1),
)
_QUATERNIONS = (2, 7, 8, 9)
_LEARNED = 17
_INDEX = {name: i for i, (name, _) in enumerate(_FIELDS)}


class AgileStateRepresentation:
    """Physical coordinates with manifold rotations and analytically carried reward memory."""

    def __init__(self):
        self.widths = tuple(width for _, width in _FIELDS)
        self.increment_widths = tuple(3 if i in _QUATERNIONS else w for i, w in enumerate(self.widths[:_LEARNED]))
        self.increment_dim = sum(self.increment_widths)
        self.error_floor = torch.full((self.increment_dim,), 0.01)

    def unflatten(self, flat: torch.Tensor) -> State:
        return flat.split(self.widths, dim=-1)

    def features(self, state: State) -> torch.Tensor:
        return torch.cat(state, -1)

    def increments(self, state: State, next_state: State) -> torch.Tensor:
        parts = []
        for i in range(_LEARNED):
            if i in _QUATERNIONS:
                rotation = math_utils.quat_mul(next_state[i], math_utils.quat_conjugate(state[i]))
                rotation = torch.where(rotation[..., 3:] < 0, -rotation, rotation)
                length = torch.linalg.vector_norm(rotation[..., :3], dim=-1, keepdim=True)
                scale = 2 * torch.atan2(length, rotation[..., 3:]) / length.clamp_min(1e-8)
                parts.append(rotation[..., :3] * torch.where(length > 1e-6, scale, 2.0))
            else:
                parts.append(next_state[i] - state[i])
        return torch.cat(parts, -1)

    def integrate(self, state: State, increment: torch.Tensor) -> State:
        parts = increment.split(self.increment_widths, -1)
        successor = []
        for i, delta in enumerate(parts):
            if i in _QUATERNIONS:
                angle = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
                rotation = torch.cat((0.5 * torch.sinc(angle / (2 * torch.pi)) * delta, torch.cos(angle / 2)), -1)
                successor.append(math_utils.normalize(math_utils.quat_mul(rotation, state[i])))
            else:
                successor.append(state[i] + delta)
        return tuple(successor) + state[_LEARNED:]

    def error(self, prediction: State, target: State) -> torch.Tensor:
        return self.increments(target, prediction)


class AgileState(ManagerTermBase):
    """Read pre-reset physical state without advancing any stateful reward observer."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        robot = env.scene["robot"]
        self.feet, _ = robot.find_bodies(["left_ankle_roll_link", "right_ankle_roll_link"], preserve_order=True)
        self.torso = robot.find_bodies("torso_link")[0][0]
        self.force_feet = env.scene["smooth_interval_forces"].find_sensors(
            ["left_ankle_roll_link", "right_ankle_roll_link"], preserve_order=True
        )[0]

    def __call__(self, env):
        # ObservationManager probes shapes before RewardManager constructs its stateful terms.
        if not hasattr(env, "reward_manager"):
            return torch.zeros(env.num_envs, sum(w for _, w in _FIELDS), device=env.device)
        robot = env.scene["robot"]
        d = robot.data
        scan = env.scene["height_scanner"].data
        height = d.root_pos_w.torch[:, 2:3] - env.scene["height_measurement_sensor"].data.ray_hits_w.torch[..., 2].mean(
            -1, keepdim=True
        )
        feet_quat = d.body_quat_w.torch[:, self.feet]
        root_acc = env.reward_manager.get_term_cfg("root_acc").func
        action_rate = env.reward_manager.get_term_cfg("action_rate_rate").func
        loading = env._smooth_contact_observer
        arms = env.action_manager.get_term("random_pos")
        parts = (
            d.root_lin_vel_b.torch,
            d.root_ang_vel_b.torch,
            d.root_quat_w.torch,
            d.joint_pos.torch,
            d.joint_vel.torch,
            (d.body_pos_w.torch[:, self.feet] - d.root_pos_w.torch[:, None]).flatten(1),
            d.body_vel_w.torch[:, self.feet, :3].flatten(1),
            feet_quat[:, 0],
            feet_quat[:, 1],
            d.body_link_quat_w.torch[:, self.torso],
            height,
            (scan.pos_w.torch[:, 2:3] - scan.ray_hits_w.torch[..., 2] - 0.5).clamp(-1, 1),
            robot.actuators.applied_effort.torch / 100.0,
            robot.actuators.computed_effort.torch / 100.0,
            env.scene["smooth_interval_forces"].data.net_forces_w_history.torch[:, :, self.force_feet, 2].flatten(1)
            / 100.0,
            arms.processed_actions,
            arms._target_processed_actions,
            root_acc.prev_body_vel,
            action_rate.prev_action_rate,
            loading.valid[:, None].float(),
            torch.full_like(height, float(not root_acc.first_call)),
        )
        return torch.cat(parts, -1)


class AgileLabels(ManagerTermBase):
    """Two slip contacts, two jump contacts, and the three native termination events."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.feet = env.scene["contact_forces"].find_sensors(
            ["left_ankle_roll_link", "right_ankle_roll_link"], preserve_order=True
        )[0]

    def __call__(self, env):
        if not hasattr(env, "termination_manager"):
            return torch.zeros(env.num_envs, 7, device=env.device)
        sensor = env.scene["contact_forces"].data
        slip = sensor.net_forces_w_history.torch[:, :, self.feet].norm(dim=-1).amax(dim=1) > 1.0
        jump = sensor.net_forces_w.torch[:, self.feet].norm(dim=-1) >= 10.0
        flags = []
        for name in ("base_orientation", "illegal_contacts", "illegal_base_height"):
            cfg = env.termination_manager.get_term_cfg(name)
            flags.append(cfg.func(env, **cfg.params)[:, None])
        return torch.cat((slip, jump, *flags), -1).float()


def previous_action(env):
    """Action before the most recently applied action, including environment reset semantics."""
    return env.action_manager.prev_action


class AgileTask(DifferentiableTask):
    """Use the Agile actor/critic signals and all active native reward terms in imagination."""

    num_labels = 7

    def __init__(self, env, device="cpu"):
        owner = env.unwrapped
        robot = owner.scene["robot"]
        self.device, self.dt = device, owner.step_dt
        self.num_actions = owner.action_manager.total_action_dim
        if robot.num_joints != 29 or self.num_actions != 15:
            raise ValueError("Agile SuperMimic requires the 29-joint robot and 15 controlled leg/waist joints.")
        self.representation = AgileStateRepresentation()
        self.controlled = owner.action_manager.get_term("joint_pos")._joint_ids
        self.default_pos = robot.data.default_joint_pos.torch[0].clone()
        self.default_vel = robot.data.default_joint_vel.torch[0].clone()
        self.limits = robot.data.soft_joint_pos_limits.torch[0].clone()
        self.velocity_limits = robot.data.soft_joint_vel_limits.torch[0].clone()
        self.gravity = torch.tensor([0.0, 0.0, -1.0], device=device)
        self.rewards = {n: owner.reward_manager.get_term_cfg(n) for n in owner.reward_manager.active_terms}
        expected = {
            "termination_penalty",
            "track_lin_vel_xy_exp",
            "track_ang_vel",
            "base_height",
            "orientation",
            "torques",
            "ankle_torques",
            "ankle_roll_torques",
            "lin_vel_z",
            "ang_vel_xy",
            "dof_vel",
            "action_rate",
            "action_rate_rate",
            "dof_pos_limits",
            "dof_vel_limits",
            "torque_limits",
            "feet_slip",
            "feet_roll",
            "feet_yaw_diff",
            "feet_yaw_mean",
            "root_acc",
            "feet_distance",
            "jumping",
            "feet_signed_clearance",
            "smooth_observer",
            "smooth_interval_loading",
        }
        if set(self.rewards) != expected:
            raise ValueError(f"Unmatched Agile reward terms: {set(self.rewards) ^ expected}")
        self.policy_cfg = owner.cfg.observations.policy
        self.frame_terms = tuple(
            getattr(self.policy_cfg, n)
            for n in (
                "base_ang_vel",
                "projected_gravity",
                "velocity_commands",
                "controlled_joint_pos",
                "controlled_joint_vel",
                "actions",
            )
        )
        self.frame_widths = (3, 3, 3, 15, 15, 15)
        if owner.observation_manager.group_obs_dim["height_scan"] != (9,):
            raise ValueError("Agile SuperMimic expects the current 3-by-3 critic height scan.")

    def _gravity(self, quaternion):
        return math_utils.quat_apply_inverse(quaternion, self.gravity.expand(*quaternion.shape[:-1], 3))

    def actor_obs(self, state, goal, context, prev_actions):
        parts = (
            state[1],
            self._gravity(state[2]),
            goal,
            (state[3] - self.default_pos)[..., self.controlled],
            (state[4] - self.default_vel)[..., self.controlled],
            prev_actions[..., -1, :],
        )
        return torch.cat(
            tuple(p * (c.scale if c.scale is not None else 1.0) for p, c in zip(parts, self.frame_terms)), -1
        )

    def actor_goal(self, state, goal):
        # Commands are already included in every actor frame, exactly as in Agile.
        return goal[..., :0]

    def add_noise(self, observation):
        if not self.policy_cfg.enable_corruption:
            return observation
        parts = observation.split(self.frame_widths, -1)
        # ObservationManager adds noise before scaling.
        return torch.cat(
            tuple(
                term.noise.func(part / (term.scale if term.scale is not None else 1.0), term.noise)
                * (term.scale if term.scale is not None else 1.0)
                if term.noise is not None
                else part
                for part, term in zip(parts, self.frame_terms)
            ),
            -1,
        )

    def dynamics_context(self, state, context, prev_actions):
        return torch.cat((torch.log1p(context), prev_actions.flatten(-2)), -1)

    def critic_obs(self, state, goal, context, prev_actions):
        return torch.cat(
            (
                goal,
                state[0],
                state[1],
                self._gravity(state[2]),
                state[3] - self.default_pos,
                0.1 * (state[4] - self.default_vel),
                prev_actions[..., -1, :],
                state[11].clamp(-1, 1),
            ),
            -1,
        )

    def critic_obs_with_history(self, state, goal, context, prev_actions, history):
        # Agile concatenates each term's five frames before the next term.
        policy = torch.cat(tuple(p.flatten(-2) for p in history.split(self.frame_widths, -1)), -1)
        return torch.cat((policy, self.critic_obs(state, goal, context, prev_actions)), -1)

    def advance(self, next_state, goal, context):
        return goal, context

    def post_transition(self, state, next_state, actions, prev_actions, labels):
        velocity = torch.cat(
            (math_utils.quat_apply(next_state[2], next_state[0]), math_utils.quat_apply(next_state[2], next_state[1])),
            -1,
        )
        return next_state[:_LEARNED] + (
            velocity,
            actions - prev_actions[..., -1, :],
            torch.ones_like(state[-2]),
            torch.ones_like(state[-1]),
        )

    def termination_prob(self, next_state, goal, labels=None):
        if labels is None:
            raise ValueError("Agile termination requires observed or predicted event labels.")
        return 1 - (1 - labels[..., 4:]).prod(-1)

    def reward_terms(self, state, actions, next_state, goal, prev_actions, labels=None):
        if labels is None:
            raise ValueError("Agile rewards require contact/termination event labels.")
        s = dict(zip(_INDEX, next_state))
        shape = actions.shape[:-1]
        feet_pos = s["feet_pos"].reshape(*shape, 2, 3)
        feet_vel = s["feet_vel"].reshape(*shape, 2, 3)
        feet_quat = torch.stack((s["left_quat"], s["right_quat"]), -2)
        root_quat = s["root_quat"]
        rate = actions - prev_actions[..., -1, :]
        # Existing quaternion helpers take a single batch dimension.
        flat_feet = feet_quat.reshape(-1, 2, 4)
        values = {}
        for name, cfg in self.rewards.items():
            p = cfg.params
            ids = p["asset_cfg"].joint_ids if "asset_cfg" in p else slice(None)
            if name == "termination_penalty":
                value = self.termination_prob(next_state, goal, labels)
            elif name == "track_lin_vel_xy_exp":
                value = rewards.tracking_reward(
                    goal[..., :2], s["lin_vel"][..., :2], p["std"], p["max_std"], p["threshold"], p["full_speed"]
                )
            elif name == "track_ang_vel":
                value = torch.exp(-(goal[..., 2] - s["ang_vel"][..., 2]).square() / p["std"] ** 2)
            elif name == "base_height":
                value = torch.exp(-(s["height"][..., 0] - p["target_height"]).square() / p["std"] ** 2)
            elif name == "orientation":
                error = self._gravity(root_quat)[..., :2].square().sum(-1) + self._gravity(s["torso_quat"])[
                    ..., :2
                ].square().sum(-1)
                value = torch.exp(-error / p["std"] ** 2)
            elif name in ("torques", "ankle_torques", "ankle_roll_torques"):
                value = (100.0 * s["applied_torque"][..., ids]).square().sum(-1)
            elif name == "lin_vel_z":
                value = s["lin_vel"][..., 2].square()
            elif name == "ang_vel_xy":
                value = s["ang_vel"][..., :2].square().sum(-1)
            elif name == "dof_vel":
                value = s["joint_vel"][..., ids].square().sum(-1)
            elif name == "action_rate":
                value = rate.square().sum(-1)
            elif name == "action_rate_rate":
                value = (rate - state[_INDEX["previous_action_rate"]]).square().sum(-1)
            elif name == "dof_pos_limits":
                q = s["joint_pos"][..., ids]
                value = ((self.limits[ids, 0] - q).clamp_min(0) + (q - self.limits[ids, 1]).clamp_min(0)).sum(-1)
            elif name == "dof_vel_limits":
                value = (
                    (s["joint_vel"][..., ids].abs() - self.velocity_limits[ids] * p["soft_ratio"]).clamp(0, 1).sum(-1)
                )
            elif name == "torque_limits":
                value = (100.0 * (s["applied_torque"][..., ids] - s["computed_torque"][..., ids])).abs().sum(-1)
            elif name == "feet_slip":
                value = (feet_vel[..., :2].norm(dim=-1) * labels[..., :2]).sum(-1)
            elif name == "feet_roll":
                value = rewards._feet_roll_l2_from_quat(flat_feet).reshape(shape)
            elif name == "feet_yaw_diff":
                value = rewards._feet_yaw_diff_l2_from_quat(flat_feet).reshape(shape)
            elif name == "feet_yaw_mean":
                value = rewards._feet_yaw_mean_vs_base_from_quat(flat_feet, root_quat.reshape(-1, 4)).reshape(shape)
            elif name == "root_acc":
                world_velocity = torch.cat(
                    (math_utils.quat_apply(root_quat, s["lin_vel"]), math_utils.quat_apply(root_quat, s["ang_vel"])), -1
                )
                value = ((world_velocity - state[_INDEX["previous_world_velocity"]]) / self.dt).square().sum(-1).clamp(
                    max=1e6
                ) * state[-1][..., 0]
            elif name == "feet_distance":
                body_pos = math_utils.quat_apply_inverse(root_quat[..., None, :].expand_as(feet_quat), feet_pos)
                value = ((body_pos[..., 0, 1] - body_pos[..., 1, 1]).abs() - p["ref_distance"]).abs()
            elif name == "jumping":
                value = (1 - labels[..., 2:4]).prod(-1)
            elif name == "feet_signed_clearance":
                value = rewards.signed_clearance_cost(
                    feet_pos.reshape(-1, 2, 3), rewards.heading(root_quat).reshape(-1), p["minimum"]
                ).reshape(shape)
            elif name == "smooth_observer":
                value = torch.zeros_like(goal[..., 0])
            elif name == "smooth_interval_loading":
                force = s["loading_forces"].reshape(*shape, 5, 2).clamp_min(0)
                value = (force[..., :4, :] - force[..., 1:5, :]).clamp_min(0).square().sum(-1).mean(-1) * state[-2][
                    ..., 0
                ]
            else:
                raise ValueError(f"Unsupported Agile reward {name}")
            values[name] = value * cfg.weight * self.dt
        return values

    def reward(self, state, actions, next_state, goal, prev_actions, labels=None):
        return torch.stack(
            tuple(self.reward_terms(state, actions, next_state, goal, prev_actions, labels).values()), -1
        ).sum(-1)
