# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compare frozen-teacher rear pushing and an attempted side squeeze without training."""

import argparse
import json
from pathlib import Path

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import torch

from isaaclab.app import launch_simulation
from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_conjugate, quat_from_euler_xyz, quat_mul, yaw_quat

from isaaclab_tasks.utils import resolve_task_config

from ..mdp.commands import PushDirectionCommand
from ..mdp.terminations import object_outside_patch
from .rollout_teachers import load_actor


class ComparisonCommand(PushDirectionCommand):
    """Timed reach, hold, and forward-motion probe; no force feedback or welds."""

    def _update_command(self):
        super()._update_command()
        t = self._env.common_step_counter * self._env.step_dt
        stationary = self.cfg.probe_mode == "standing"
        blend = 0.0 if stationary else min(max((t - 2.0) / 2.0, 0.0), 1.0)
        speed = 0.0 if stationary else self.cfg.probe_speed * min(max(t - 6.0, 0.0), 1.0)
        self.direction[:] = 0 if stationary else torch.tensor([1.0, 0.0], device=self.device)
        self.velocity[:] = 0
        self.velocity[:, 0] = speed
        # Identical posture timing in both contact conditions. Height is above initial sand surface.
        self.height[:] = self.cfg.neutral_height + blend * (self.cfg.probe_height - self.cfg.neutral_height)
        self.pitch[:] = blend * self.cfg.contact_pitch
        self.engagement[:] = blend
        obj_q = yaw_quat(self.object.data.root_quat_w.torch)[:, None].expand(-1, 2, -1)
        points = torch.tensor(self.cfg.probe_points, device=self.device)[None].expand(self.num_envs, -1, -1)
        target_w = self.object.data.root_pos_w.torch[:, None] + quat_apply(obj_q, points)
        torso_p = self.robot.data.body_pos_w.torch[:, self.torso_id, None]
        torso_q = self.robot.data.body_quat_w.torch[:, self.torso_id, None].expand(-1, 2, -1)
        target_b = quat_apply_inverse(torso_q, target_w - torso_p)
        neutral = self._neutral[None].expand(self.num_envs, -1, -1)
        self.hand_pose[:, :, :3] = torch.lerp(neutral[:, :, :3], target_b, blend)
        angle = torch.tensor(self.cfg.probe_hand_yaw, device=self.device)[None].expand(self.num_envs, -1)
        rotation = quat_from_euler_xyz(torch.zeros_like(angle), torch.zeros_like(angle), angle)
        goal_q = quat_mul(quat_conjugate(torso_q), quat_mul(obj_q, rotation))
        goal_q = torch.where((goal_q * neutral[:, :, 3:]).sum(-1, keepdim=True) < 0, -goal_q, goal_q)
        q = torch.lerp(neutral[:, :, 3:], goal_q, blend)
        self.hand_pose[:, :, 3:] = q / q.norm(dim=-1, keepdim=True)
        # Send the actual requested targets: do not silently project a side grip into a rear push.
        errors = [torch.cdist(self.hand_pose[:, i, :3], w[:, :3]).min(-1).values for i, w in enumerate(self.workspace)]
        self.metrics["workspace_projection_error"][:] = torch.stack(errors, -1).mean(-1)


def euler_xyz(q):
    """XYZW quaternion to roll/pitch/yaw, radians, for recorded measurements."""
    x, y, z, w = q
    return [
        np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)),
        np.arcsin(np.clip(2 * (w * y - z * x), -1, 1)),
        np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)),
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("standing", "rear", "side"), required=True)
    parser.add_argument("--speed", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seconds", type=float, default=12.0)
    parser.add_argument("--height", type=float, default=0.60)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    task = "IsaacContrib-Push-G1-29dof-ParticleNeRD-MovingPatch-ObjectPush-Play"
    cfg, agent = resolve_task_config(task, "rsl_rl_cfg_entry_point", overrides=[])
    cfg.scene.num_envs = 1
    cfg.seed = args.seed
    cfg.episode_length_s = 1000.0
    for key in vars(cfg.terminations):
        if not key.startswith("_"):
            setattr(cfg.terminations, key, None)
    cfg.events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
    cfg.events.reset_object.params.update(forward_range=(0.70, 0.70), lateral_range=(0.0, 0.0))
    # Control keeps the same object in the scene but beyond reach, isolating robot-object contact.
    if args.mode == "standing":
        cfg.events.reset_object.params["forward_range"] = (1.25, 1.25)
    c = cfg.commands.object_direction
    c.class_type = ComparisonCommand
    c.resampling_time_range = (1000.0, 1000.0)
    c.probe_mode, c.probe_speed, c.probe_height = args.mode, args.speed, args.height
    # Side wrists start outside the side faces; 2 cm inward offset requests a modest squeeze.
    # This is a position offset, not a prescribed force; measured hand force is logged below.
    c.probe_points = (
        [(-0.20, 0.435, 0.05), (-0.20, -0.435, 0.05)]
        if args.mode == "side"
        else [(-0.455, 0.20, 0.30), (-0.455, -0.20, 0.30)]
    )
    c.probe_hand_yaw = [-np.pi / 2, np.pi / 2] if args.mode == "side" else [0.0, 0.0]
    cfg.sim.physics.solver_cfg.entries[1].solver_cfg.compile_model = False
    cfg.video_recorders = []
    cfg.sim.visualizer_cfgs = cfg.sim.visualizer_cfgs[:1]
    cfg.sim.visualizer_cfgs[0].headless = True
    cfg.sim.visualizer_cfgs[0].window.size = (960, 640)
    args.output.mkdir(parents=True, exist_ok=True)
    rows, reason = [], "completed"
    with launch_simulation(cfg, {"visualizer": ["newton_gl"]}):
        env = gym.make(task, cfg=cfg).unwrapped
        writer = imageio.get_writer(args.output / "rollout.mp4", fps=25, codec="libx264", quality=7)
        try:
            obs, _ = env.reset(seed=args.seed)
            command = env.command_manager.get_term("object_direction")
            robot, obj = env.scene["robot"], env.scene["object"]
            lower = load_actor(agent.networks.locomotion_teacher.pretrained_cfg["path"], env.device, 799, 15)
            upper = load_actor(agent.networks.manipulation_teacher.pretrained_cfg["path"], env.device, 76, 14)
            hand_ids = [robot.find_bodies(f"{side}_(wrist_yaw|hand_.*)_link")[0] for side in ("left", "right")]
            knee_ids = robot.find_bodies(".*knee.*")[0]
            with torch.inference_mode():
                for step in range(int(args.seconds / env.step_dt)):
                    command._update_command()
                    obs = env.observation_manager.compute(update_history=False)
                    a = lower(torch.cat((obs["locomotion_teacher_obs"], obs["locomotion_teacher_command"]), dim=-1))
                    b = upper(obs["manipulation_teacher_obs"])
                    obs, _, done, trunc, _ = env.step(torch.cat((a, b), -1))
                    if (done | trunc).any():
                        reason = "unexpected_auto_reset"
                        break
                    force = env.object_contact_force[0]
                    hand_force = torch.stack([force[ids].sum(0) for ids in hand_ids])
                    hand_mag = torch.stack([force[ids].norm(dim=-1).amax() for ids in hand_ids])
                    object_q = obj.data.root_quat_w.torch[0]
                    # Force of hand on object, in object yaw coordinates (reaction reverses sign).
                    local_force = quat_apply_inverse(yaw_quat(object_q).expand(2, -1), -hand_force)
                    wrist_w = robot.data.body_pos_w.torch[:, command.hand_ids]
                    local_wrist = quat_apply_inverse(
                        object_q[None, None].expand(1, 2, -1), wrist_w - obj.data.root_pos_w.torch[:, None]
                    )
                    torso_q = robot.data.body_quat_w.torch[0, command.torso_id].cpu().tolist()
                    row = dict(
                        time=(step + 1) * env.step_dt,
                        speed=float(command.velocity[0, 0]),
                        robot=robot.data.root_pos_w.torch[0].cpu().tolist(),
                        object=obj.data.root_pos_w.torch[0].cpu().tolist(),
                        torso_rpy=euler_xyz(torso_q),
                        object_rpy=euler_xyz(object_q.cpu().tolist()),
                        hand_contact=hand_mag.cpu().tolist(),
                        hand_force_object_yaw=local_force.cpu().tolist(),
                        knee_contact=float(force[knee_ids].norm(dim=-1).sum()),
                        all_object_contact=float(force.norm(dim=-1).sum()),
                        wrist_object_frame=local_wrist[0].cpu().tolist(),
                        hand_error=float(command.metrics["hand_target_error"][0]),
                        workspace_distance=float(command.metrics["workspace_projection_error"][0]),
                    )
                    rows.append(row)
                    if step % 2 == 0:
                        writer.append_data(env.sim._visualizers[0].render_rgb_array())
                    if not all(torch.isfinite(v).all() for v in obs.values()):
                        reason = "nonfinite_state"
                        break
                    if max(abs(row["torso_rpy"][0]), abs(row["torso_rpy"][1])) > np.pi / 3:
                        reason = "torso_tilt_over_60deg"
                        break
                    if args.mode == "standing" and row["all_object_contact"] > 1:
                        reason = "first_robot_object_contact"
                        break
                    if object_outside_patch(env, "object", (0.325, 0.325, 0.325), 0.05).any():
                        reason = "object_outside_simulated_patch"
                        break
                    if step % 100 == 0:
                        print("PROBE", json.dumps(row), flush=True)
        finally:
            writer.close()
            env.close()
    result = dict(
        mode=args.mode,
        speed=args.speed,
        seed=args.seed,
        stop_reason=reason,
        duration=rows[-1]["time"] if rows else 0,
        specification=dict(
            mass_kg=5.0,
            cube_side_m=0.65,
            target_height_m=args.height,
            target_pitch_deg=25,
            wrist_points=c.probe_points,
            hand_yaw=c.probe_hand_yaw,
            timeline="0-2s neutral, 2-4s reach, 4-6s hold, 6-7s velocity ramp, then push",
        ),
    )
    if rows:
        movement = [r for r in rows if r["time"] >= 7]
        holding = [r for r in rows if 4 <= r["time"] < 6]
        for name, segment in (("whole", rows), ("holding", holding), ("walking", movement)):
            if not segment:
                continue
            f = np.array([r["hand_contact"] for r in segment])
            force = np.array([r["hand_force_object_yaw"] for r in segment])
            squeeze = np.minimum(np.maximum(-force[:, 0, 1], 0), np.maximum(force[:, 1, 1], 0))
            result[name] = dict(
                mean_bilateral_squeeze_N=float(squeeze.mean()),
                knee_contact_fraction=float(np.mean([r["knee_contact"] > 1 for r in segment])),
                object_dx=segment[-1]["object"][0] - segment[0]["object"][0],
                robot_xy_drift=float(np.linalg.norm(np.array(segment[-1]["robot"][:2]) - segment[0]["robot"][:2])),
                bilateral_contact_fraction=float((f > 1).all(-1).mean()),
                any_contact_fraction=float((f > 1).any(-1).mean()),
                hand_error_m=float(np.mean([r["hand_error"] for r in segment])),
                workspace_distance_m=float(np.mean([r["workspace_distance"] for r in segment])),
                max_object_pitch_deg=float(np.rad2deg(max(abs(r["object_rpy"][1]) for r in segment))),
                max_torso_roll_deg=float(np.rad2deg(max(abs(r["torso_rpy"][0]) for r in segment))),
                max_torso_pitch_deg=float(np.rad2deg(max(abs(r["torso_rpy"][1]) for r in segment))),
            )
    (args.output / "measurements.json").write_text(json.dumps(rows, indent=2))
    (args.output / "summary.json").write_text(json.dumps(result, indent=2))
    print("RESULT", json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
