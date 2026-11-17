# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Generate paired, collision-checked wrist targets and measure held-out spatial coverage.

Run this file from the IsaacLab checkout. The optional MJCF input must have been exported
from the task's Newton/MJWarp model; otherwise this script exports that model first.
Collision checking preserves the asset's collision masks and adjacent-link exclusions.
The neutral waist and configured fixed finger posture are used.
Valid endpoints do not guarantee collision-free transitions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import re
import tempfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial import cKDTree


def _find_name(model, kind, suffix):
    count = model.njnt if kind == "joint" else model.nbody
    matches = [i for i in range(count) if getattr(model, kind)(i).name.endswith("_" + suffix)]
    if len(matches) != 1:
        raise ValueError(f"Expected one {kind} matching {suffix!r}, got {matches}.")
    return matches[0]


def _sample_batch(arguments):
    model_path, settings, seed, count = arguments
    model = mujoco.MjModel.from_xml_path(model_path)
    # Include near contacts so rejection enforces a small positive clearance.
    model.geom_margin[:] = np.maximum(model.geom_margin, settings["collision_clearance"])
    data = mujoco.MjData(model)
    for pattern, position in settings["fixed_joint_positions"].items():
        indices = [i for i in range(model.njnt) if re.search("_" + pattern + "$", model.joint(i).name)]
        if not indices:
            raise ValueError(f"No joints match fixed posture pattern {pattern!r}.")
        if any(not model.jnt_range[i, 0] <= position <= model.jnt_range[i, 1] for i in indices):
            raise ValueError(f"Fixed posture {position} exceeds limits for {pattern!r}.")
        data.qpos[model.jnt_qposadr[indices]] = position
    left_ids = [_find_name(model, "joint", name) for name in settings["left_joint_names"]]
    right_ids = [_find_name(model, "joint", name) for name in settings["right_joint_names"]]
    signs = np.asarray(settings["mirror_joint_signs"])
    left_limits = model.jnt_range[left_ids].copy()
    right_limits = model.jnt_range[right_ids].copy()
    # Intersect left bounds with mirrored right bounds before sampling.
    mirrored_right_limits = np.sort(right_limits * signs[:, None], axis=1)
    lower = np.maximum(left_limits[:, 0], mirrored_right_limits[:, 0]) + settings["joint_limit_margin"]
    upper = np.minimum(left_limits[:, 1], mirrored_right_limits[:, 1]) - settings["joint_limit_margin"]
    if np.any(lower >= upper):
        raise ValueError("Joint bounds and margin leave an empty sampling interval.")
    bodies = [_find_name(model, "body", settings[name]) for name in ("left_body_name", "right_body_name")]
    elbows = [_find_name(model, "body", settings[name]) for name in ("left_elbow_name", "right_elbow_name")]
    reference = _find_name(model, "body", settings["reference_body_name"])
    rng = np.random.default_rng(seed)
    candidates = rng.uniform(lower, upper, (count, len(left_ids)))
    poses, joints = [], []
    rejected = {"collision": 0, "elbow": 0, "asymmetry": 0}
    max_position_asymmetry = max_rotation_asymmetry = 0.0
    for q in candidates:
        data.qpos[model.jnt_qposadr[left_ids]] = q
        data.qpos[model.jnt_qposadr[right_ids]] = q * signs
        mujoco.mj_fwdPosition(model, data)
        if any(
            contact.dist < settings["collision_clearance"]
            and model.geom_bodyid[contact.geom1] != 0
            and model.geom_bodyid[contact.geom2] != 0
            for contact in data.contact
        ):
            rejected["collision"] += 1
            continue
        rotation = data.xmat[reference].reshape(3, 3)
        local_positions = (data.xpos[bodies] - data.xpos[reference]) @ rotation
        lateral = ((data.xpos[bodies] - data.xpos[elbows]) @ rotation)[:, 1] * (1, -1)
        if settings["require_elbows_outward"] and np.any(lateral > 0):
            rejected["elbow"] += 1
            continue
        quaternions = []
        for body in bodies:
            quat = np.empty(4)
            local_rotation = rotation.T @ data.xmat[body].reshape(3, 3)
            mujoco.mju_mat2Quat(quat, local_rotation.ravel())
            if quat[0] < 0:
                quat *= -1
            quaternions.append(quat[[1, 2, 3, 0]])
        left_pose = np.r_[local_positions[0], quaternions[0]]
        # Reflect the pose exactly; independently check the mirrored joint configuration by FK.
        right_pose = left_pose * (1, -1, 1, -1, 1, -1, 1)
        position_error = np.linalg.norm(right_pose[:3] - local_positions[1])
        rotation_error = 2 * np.arccos(np.clip(abs(np.dot(right_pose[3:], quaternions[1])), 0, 1))
        max_position_asymmetry = max(max_position_asymmetry, position_error)
        max_rotation_asymmetry = max(max_rotation_asymmetry, rotation_error)
        if position_error > 0.001 or rotation_error > 0.005:
            rejected["asymmetry"] += 1
            continue
        poses.append(np.stack((left_pose, right_pose)))
        joints.append(np.stack((q, q * signs)))
    return (
        np.asarray(poses, dtype=np.float32).reshape(-1, 2, 7),
        np.asarray(joints, dtype=np.float32).reshape(-1, 2, len(left_ids)),
        rejected,
        [max_position_asymmetry, max_rotation_asymmetry],
    )


def _export_model(path):
    import gymnasium as gym

    from isaaclab.app import launch_simulation
    from isaaclab.envs.mdp import UniformPoseCommandCfg

    from isaaclab_tasks.contrib.push.config.g1_29dof_manipulation.env_cfg.commands_cfg import LinkPoseCommandCfg
    from isaaclab_tasks.contrib.push.config.g1_29dof_manipulation.manipulation_env_cfg import G1ManipulationEnvCfg

    cfg = G1ManipulationEnvCfg()
    cfg.scene.num_envs = 1
    cfg.sim.visualizer_cfgs = []
    cfg.video_recorders = []
    cfg.sim.physics.solver_cfg.save_to_mjcf = str(path)
    # Exporting must also work before the dataset exists.
    for name in ("left_ee_pose", "right_ee_pose"):
        old = getattr(cfg.commands, name)
        setattr(
            cfg.commands,
            name,
            LinkPoseCommandCfg(
                asset_name=old.asset_name,
                body_name=old.body_name,
                reference_body_name=old.reference_body_name,
                resampling_time_range=(3.0, 5.0),
                debug_vis=False,
                ranges=UniformPoseCommandCfg.Ranges(
                    pos_x=(0.0, 0.0),
                    pos_y=(0.0, 0.0),
                    pos_z=(0.0, 0.0),
                    roll=(0.0, 0.0),
                    pitch=(0.0, 0.0),
                    yaw=(0.0, 0.0),
                ),
            ),
        )
    with launch_simulation(cfg):
        env = gym.make("IsaacContrib-Push-G1-29dof-Manipulation", cfg=cfg).unwrapped
        try:
            env.reset()
        finally:
            env.close()


def main():
    from isaaclab_tasks.contrib.push.config.g1_29dof_manipulation.env_cfg.dataset_cfg import HandPoseDatasetCfg

    defaults = HandPoseDatasetCfg()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, help="Optional previously exported task MJCF.")
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "hand_poses_closed.npz"
    )
    parser.add_argument("--candidates", type=int, default=defaults.candidates)
    parser.add_argument("--validation-candidates", type=int, default=defaults.validation_candidates)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    settings = asdict(defaults)
    settings.update(candidates=args.candidates, validation_candidates=args.validation_candidates)
    if args.candidates <= 0 or args.validation_candidates <= 0 or args.workers <= 0:
        parser.error("Sample counts and worker count must be positive.")
    with tempfile.TemporaryDirectory(prefix="g1-hand-poses-") as directory:
        model_path = args.model or Path(directory) / "robot.xml"
        if args.model is None:
            _export_model(model_path)
        batch_size = 25_000
        batches = [
            (str(model_path), settings, defaults.seed + i, min(batch_size, args.candidates - i * batch_size))
            for i in range((args.candidates + batch_size - 1) // batch_size)
        ]
        accepted_poses, accepted_joints = [], []
        counts = {}
        rejected = {"collision": 0, "elbow": 0, "asymmetry": 0}
        asymmetry = np.zeros(2)
        valid_count = 0
        # Spawn avoids inheriting CUDA/renderer state from model export.
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            for i, (poses, joints, failures, errors) in enumerate(pool.map(_sample_batch, batches)):
                valid_count += len(poses)
                asymmetry = np.maximum(asymmetry, errors)
                for key in rejected:
                    rejected[key] += failures[key]
                voxels = np.floor(poses[:, 0, :3] / defaults.voxel_size).astype(np.int32)
                keep = []
                for j, voxel in enumerate(voxels):
                    key = tuple(voxel)
                    if counts.get(key, 0) < defaults.samples_per_voxel:
                        keep.append(j)
                        counts[key] = counts.get(key, 0) + 1
                if keep:
                    accepted_poses.append(poses[keep])
                    accepted_joints.append(joints[keep])
                if (i + 1) % 8 == 0 or i + 1 == len(batches):
                    print(
                        f"Candidates: {min((i + 1) * batch_size, args.candidates):,}; "
                        f"valid: {valid_count:,}; cells: {len(counts):,}",
                        flush=True,
                    )
        if not accepted_poses:
            raise RuntimeError("No valid poses were sampled; inspect collision masks, joint limits, and filters.")
        poses = np.concatenate(accepted_poses)
        joints = np.concatenate(accepted_joints)
        validation, _, validation_failures, _ = _sample_batch(
            (str(model_path), settings, defaults.seed + 1_000_000, args.validation_candidates)
        )
        if not len(validation):
            raise RuntimeError("No valid held-out poses were sampled.")
        distances = cKDTree(poses[:, 0, :3]).query(validation[:, 0, :3])[0]
        metadata = {
            "format_version": 1,
            "settings": settings,
            "candidate_count": args.candidates,
            "valid_candidates": valid_count,
            "saved_pairs": len(poses),
            "occupied_1cm_cells": len(counts),
            "rejected": rejected,
            "validation_valid": len(validation),
            "validation_rejected": validation_failures,
            "held_out_nearest_position_distance_m": dict(
                zip(("p50", "p95", "p99", "max"), np.quantile(distances, (0.5, 0.95, 0.99, 1)).tolist())
            ),
            "held_out_fraction_within_1cm": float(np.mean(distances <= defaults.voxel_size)),
            "left_position_min_m": poses[:, 0, :3].min(axis=0).tolist(),
            "left_position_max_m": poses[:, 0, :3].max(axis=0).tolist(),
            "max_mirror_fk_error": {"position_m": float(asymmetry[0]), "orientation_rad": float(asymmetry[1])},
            "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
            "mujoco_version": mujoco.__version__,
            "collision_model": (
                "Task Newton/MJWarp MJCF export; asset collision masks and exclusions preserved; "
                "neutral waist; fixed finger posture from settings"
            ),
            "quaternion_order": "xyzw",
            "limitations": (
                "Endpoint feasibility only. Spatial coverage does not measure orientation coverage "
                "or prove complete workspace coverage."
            ),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.output,
            left_pose=poses[:, 0],
            right_pose=poses[:, 1],
            left_joint_pos=joints[:, 0],
            right_joint_pos=joints[:, 1],
            left_joint_names=np.asarray(defaults.left_joint_names),
            right_joint_names=np.asarray(defaults.right_joint_names),
            reference_body_name=np.asarray(defaults.reference_body_name),
            metadata_json=np.asarray(json.dumps(metadata)),
        )
        args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(json.dumps(metadata, indent=2), flush=True)
        print(f"Saved {args.output} ({args.output.stat().st_size / 1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
