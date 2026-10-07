# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sample G1 static postures from the central 90% of joint limits and project with Mink.

Run with ``uv run --with mink --with 'qpsolvers[quadprog]' python <this-file> --help``.
Use ``--stance-groups`` to share foot anchors across poses and add nearby posture variants.
Geometric gates do not certify collision-free transitions or dynamic tracking.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import mink
import mujoco
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

BODY_NAMES = (
    "pelvis",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
    "torso_link",
)


def load_robot(urdf_path: Path) -> mujoco.MjModel:
    """Load collision geometry and retain named fixed links for FK comparisons."""
    root = ET.parse(urdf_path).getroot()
    for mesh in root.findall(".//mesh"):
        relative_path = Path(mesh.attrib["filename"])
        candidates = (urdf_path.parent / relative_path, urdf_path.parent.parent / relative_path)
        mesh.set("filename", str(next(path for path in candidates if path.is_file())))
    for extension in root.findall("mujoco"):
        root.remove(extension)
    extension = ET.SubElement(root, "mujoco")
    ET.SubElement(extension, "compiler", fusestatic="false", discardvisual="true", strippath="false")
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    with tempfile.TemporaryDirectory() as directory:
        model_path = str(Path(directory) / "robot.xml")
        mujoco.mj_saveLastXML(model_path, model)
        root = ET.parse(model_path).getroot()
    world = root.find("worldbody")
    ET.SubElement(world.find("body"), "freejoint", name="floating_base")
    ET.SubElement(world, "geom", name="ground", type="plane", size="0 0 0.1")
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))


def nominal_configuration(model: mujoco.MjModel) -> np.ndarray:
    """Return a bent-knee, bent-elbow starting configuration above the ground."""
    position = model.qpos0.copy()
    position[2] = 0.757
    for joint_index in range(1, model.njnt):
        name = model.joint(joint_index).name
        value = 0.0
        if "hip_pitch" in name:
            value = -0.312
        elif "knee" in name:
            value = 0.669
        elif "ankle_pitch" in name:
            value = -0.363
        elif "elbow" in name:
            value = 0.6
        elif "shoulder_pitch" in name:
            value = 0.2
        elif "shoulder_roll" in name:
            value = 0.2 if name.startswith("left_") else -0.2
        position[model.jnt_qposadr[joint_index]] = value
    return position


class PoseRejected(ValueError):
    """A sampled posture failed an explicitly checked geometric acceptance criterion."""


class PoseSampler:
    """Sample and project joint postures into grounded, collision-separated G1 configurations."""

    def __init__(self, model: mujoco.MjModel):
        self.model = model
        self.joint_names = [model.joint(index).name for index in range(1, model.njnt)]
        self.original_limits = model.jnt_range[1:].copy()
        midpoint = self.original_limits.mean(axis=1)
        half_range = 0.45 * np.diff(self.original_limits, axis=1).ravel()
        self.lower = midpoint - half_range
        self.upper = midpoint + half_range
        # Enforce the same 90% interval in the IK solution, not just in the sampled objective.
        model.jnt_range[1:, 0] = self.lower
        model.jnt_range[1:, 1] = self.upper
        self.configuration = mink.Configuration(model, q=nominal_configuration(model))
        self.body_ids = [model.body(name).id for name in BODY_NAMES]
        self.feet = [
            mink.FrameTask(name, "body", position_cost=100.0, orientation_cost=100.0, gain=0.5)
            for name in BODY_NAMES[1:3]
        ]
        self.pelvis = mink.FrameTask(
            "pelvis", "body", position_cost=[0.0, 0.0, 3.0], orientation_cost=[0.5, 0.5, 0.5], lm_damping=0.1
        )
        self.center_of_mass = mink.ComTask(cost=[10.0, 10.0, 0.0])
        weights = np.ones(model.nv)
        weights[:6] = 0.0
        weights[6:18] = 0.05
        self.posture = mink.PostureTask(model, cost=weights, lm_damping=0.1)
        robot_geoms = [index for index in range(model.ngeom) if model.geom_bodyid[index] != 0]
        foot_bodies = set(self.body_ids[1:3])
        nonfoot_geoms = [index for index in robot_geoms if model.geom_bodyid[index] not in foot_bodies]
        self.collisions = mink.CollisionAvoidanceLimit(
            model,
            geom_pairs=[(robot_geoms, robot_geoms), (nonfoot_geoms, [model.geom("ground").id])],
            minimum_distance_from_collisions=0.003,
            collision_detection_distance=0.05,
        )
        self.limits = [mink.ConfigurationLimit(model, gain=0.8), self.collisions]
        self.limb_columns = []
        for side, chain in (
            ("left", ("hip", "knee", "ankle")),
            ("right", ("hip", "knee", "ankle")),
            ("left", ("shoulder", "elbow", "wrist")),
            ("right", ("shoulder", "elbow", "wrist")),
        ):
            self.limb_columns.append(
                [
                    model.jnt_dofadr[index + 1]
                    for index, name in enumerate(self.joint_names)
                    if name.startswith(side) and any(part in name for part in chain)
                ]
            )

    def sample(
        self,
        rng: np.random.Generator,
        category: str,
        stance: tuple[float, float] | None = None,
        near_pose: dict[str, np.ndarray] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, float]]:
        """Draw a full-range posture objective, add a category constraint, then solve and validate."""
        target = nominal_configuration(self.model)
        target[7:] = rng.uniform(self.lower, self.upper)
        # Category stratification deliberately covers waist extremes and low/wide configurations.
        for axis in ("roll", "pitch"):
            if category == f"waist_{axis}":
                index = self.joint_names.index(f"waist_{axis}_joint")
                target[7 + index] = rng.choice([-1, 1]) * rng.uniform(0.40, self.upper[index])
        width = rng.uniform(0.22, 0.36)
        height = rng.uniform(0.58, 0.74)
        pelvis_y = 0.0
        if category == "squat":
            width = rng.uniform(0.28, 0.44)
            height = rng.uniform(0.36, 0.49)
        elif category == "side_stretch":
            width = rng.uniform(0.48, 0.66)
            height = rng.uniform(0.42, 0.56)
            pelvis_y = rng.choice([-1, 1]) * width * rng.uniform(0.12, 0.32)
        toe_out = rng.uniform(0.5, 1.0) if category == "side_stretch" else 0.0
        if stance is not None:
            width, toe_out = stance
        initial = nominal_configuration(self.model)
        if near_pose is not None:
            target[7:] = np.clip(near_pose["joint_pos"] + rng.uniform(-0.18, 0.18, 29), self.lower, self.upper)
            height = near_pose["body_pos_w"][0, 2] + rng.uniform(-0.02, 0.02)
            pelvis_y = near_pose["body_pos_w"][0, 1]
            initial[:3] = near_pose["body_pos_w"][0]
            initial[3:7] = near_pose["body_quat_w"][0, [3, 0, 1, 2]]
            initial[7:] = near_pose["joint_pos"]
        foot_positions = np.array([[0.0, width / 2, 0.0354], [0.0, -width / 2, 0.0354]])
        foot_orientations = Rotation.from_rotvec([[0.0, 0.0, toe_out], [0.0, 0.0, -toe_out]])
        for task, position, rotation in zip(self.feet, foot_positions, foot_orientations.as_matrix()):
            task.set_target(mink.SE3.from_rotation_and_translation(mink.SO3.from_matrix(rotation), position))
        self.pelvis.set_target(
            mink.SE3.from_rotation_and_translation(mink.SO3.identity(), np.array([0.0, pelvis_y, height]))
        )
        self.center_of_mass.set_target(np.array([0.025, pelvis_y, 0.6]))
        self.posture.set_target(target)
        self.configuration.update(initial)
        for _ in range(100):
            velocity = mink.solve_ik(
                self.configuration,
                [*self.feet, self.posture, self.pelvis, self.center_of_mass],
                0.05,
                solver="quadprog",
                damping=1e-3,
                limits=self.limits,
            )
            self.configuration.integrate_inplace(velocity, 0.05)
            if np.linalg.norm(velocity) < 0.01:
                break
        result, metrics = self.validate(foot_positions, foot_orientations)
        result["sampled_joint_pos"] = target[7:].copy()
        for axis in ("roll", "pitch"):
            if category == f"waist_{axis}":
                index = self.joint_names.index(f"waist_{axis}_joint")
                if abs(result["joint_pos"][index]) < 0.35:
                    raise PoseRejected(f"{category}: projected waist angle below 0.35 rad")
        if category == "squat" and result["body_pos_w"][0, 2] > 0.51:
            raise PoseRejected("squat: projected pelvis height above 0.51 m")
        if near_pose is not None and np.sqrt(np.mean((result["joint_pos"] - near_pose["joint_pos"]) ** 2)) > 0.20:
            raise PoseRejected("Nearby posture exceeds 0.20 rad joint RMS change")
        return result, metrics

    def validate(
        self, foot_positions: np.ndarray, foot_orientations: Rotation
    ) -> tuple[dict[str, np.ndarray], dict[str, float]]:
        """Check the final configuration independently of the IK stopping condition."""
        model, data = self.model, self.configuration.data
        mujoco.mj_forward(model, data)
        joint_position = self.configuration.q[7:]
        if (
            not np.isfinite(self.configuration.q).all()
            or np.any(joint_position < self.lower - 1e-6)
            or np.any(joint_position > self.upper + 1e-6)
        ):
            raise PoseRejected("Projected joints outside the central 90% limits")
        foot_error = np.linalg.norm(data.xpos[self.body_ids[1:3]] - foot_positions, axis=-1).max()
        foot_rotation = (
            (Rotation.from_matrix(data.xmat[self.body_ids[1:3]].reshape(2, 3, 3)) * foot_orientations.inv())
            .magnitude()
            .max()
        )
        if foot_error > 0.001 or foot_rotation > 0.005:
            raise PoseRejected(f"Feet not fixed: position={foot_error:.4f} m, angle={foot_rotation:.4f} rad")
        minimum_clearance = min(
            mujoco.mj_geomDistance(model, data, first, second, 0.1, None)
            for first, second in self.collisions.geom_id_pairs
        )
        if minimum_clearance < 0.0025:
            raise PoseRejected(f"Collision clearance {minimum_clearance:.4f} m below 0.0025 m")
        if any(contact.dist < -0.001 for contact in data.contact):
            raise PoseRejected("Ground or self penetration exceeds 1 mm")
        corners = np.array([[-0.05, -0.03], [-0.05, 0.03], [0.12, -0.03], [0.12, 0.03]])
        rotated_corners = np.einsum("bij,kj->bki", foot_orientations.as_matrix()[:, :2, :2], corners)
        hull = ConvexHull((foot_positions[:, None, :2] + rotated_corners).reshape(-1, 2))
        support_margin = -(hull.equations[:, :2] @ data.subtree_com[1, :2] + hull.equations[:, 2]).max()
        if support_margin < 0.015:
            raise PoseRejected(f"COM support margin {support_margin:.4f} m below 0.015 m")
        minimum_singular_value = np.inf
        for body, columns in zip(self.body_ids[1:], self.limb_columns):
            position_jacobian = np.zeros((3, model.nv))
            rotation_jacobian = np.zeros((3, model.nv))
            mujoco.mj_jacBody(model, data, position_jacobian, rotation_jacobian, body)
            jacobian = np.concatenate((position_jacobian[:, columns], 0.2 * rotation_jacobian[:, columns]))
            minimum_singular_value = min(minimum_singular_value, np.linalg.svd(jacobian, compute_uv=False)[-1])
        if minimum_singular_value < 0.001:
            raise PoseRejected(f"Limb Jacobian minimum singular value {minimum_singular_value:.6f} below 0.001")
        positions = data.xpos[self.body_ids].copy()
        orientations = Rotation.from_matrix(data.xmat[self.body_ids].reshape(-1, 3, 3)).as_quat()
        result = dict(
            joint_pos=joint_position.copy(),
            joint_vel=np.zeros_like(joint_position),
            body_pos_w=positions,
            body_quat_w=orientations,
            body_lin_vel_w=np.zeros_like(positions),
            body_ang_vel_w=np.zeros_like(positions),
            foot_contact=np.ones(2),
        )
        metrics = dict(
            foot_position_error_m=float(foot_error),
            foot_orientation_error_rad=float(foot_rotation),
            collision_clearance_m=float(minimum_clearance),
            support_margin_m=float(support_margin),
            minimum_limb_singular_value=float(minimum_singular_value),
        )
        return result, metrics

    def mirror(self, pose: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], dict[str, float]]:
        """Reflect a pose, then recompute FK and revalidate against the actual robot geometry."""
        partners = [
            self.joint_names.index(
                name.replace("left_", "right_", 1) if name.startswith("left_") else name.replace("right_", "left_", 1)
            )
            for name in self.joint_names
        ]
        signs = np.array([-1 if name.endswith(("_roll_joint", "_yaw_joint")) else 1 for name in self.joint_names])
        position = self.configuration.q.copy()
        position[:3] = pose["body_pos_w"][0] * [1, -1, 1]
        quaternion = pose["body_quat_w"][0] * [-1, 1, -1, 1]
        position[3:7] = quaternion[[3, 0, 1, 2]]
        position[7:] = pose["joint_pos"][partners] * signs
        self.configuration.update(position)
        foot_orientation = Rotation.from_quat(pose["body_quat_w"][[2, 1]] * [-1, 1, -1, 1])
        result, metrics = self.validate(pose["body_pos_w"][[2, 1]] * [1, -1, 1], foot_orientation)
        result["sampled_joint_pos"] = pose["sampled_joint_pos"][partners] * signs
        return result, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--poses", type=int, default=100, help="Independent original poses, each with a validated mirror."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--stance-groups", type=int, default=0, help="Shared foot-anchor groups; zero keeps independent poses."
    )
    args = parser.parse_args()
    if args.poses < 5:
        parser.error("--poses must be at least five to cover all posture categories")
    if args.stance_groups < 0 or (args.stance_groups and args.poses % (2 * args.stance_groups)):
        parser.error("--poses must be divisible by twice --stance-groups")
    sampler = PoseSampler(load_robot(args.urdf.resolve()))
    if len(sampler.joint_names) != 29:
        raise ValueError("Expected 29 actuated G1 joints")
    rng = np.random.default_rng(args.seed)
    poses, categories, metrics, rejections, stance_ids = [], [], [], {}, []
    stances = list(zip(np.linspace(0.32, 0.60, args.stance_groups), np.linspace(0.0, 0.75, args.stance_groups)))
    for index in range(args.poses):
        group = (index // 2) % args.stance_groups if args.stance_groups else 0
        category_index = (index // (2 * args.stance_groups)) % 5 if args.stance_groups else index % 5
        category = ("random", "squat", "waist_roll", "waist_pitch", "side_stretch")[category_index]
        stance = stances[group] if args.stance_groups else None
        near_pose = poses[-2] if args.stance_groups and index % 2 else None
        for attempt in range(200):
            try:
                pose, validation = sampler.sample(rng, category, stance, near_pose)
                mirrored_pose, mirrored_validation = sampler.mirror(pose)
                break
            except (PoseRejected, mink.exceptions.NoSolutionFound) as error:
                reason = str(error)
                rejections[reason] = rejections.get(reason, 0) + 1
                if attempt % 20 == 0:
                    print(f"Rejected {category}, attempt {attempt + 1}: {error}", flush=True)
        else:
            raise RuntimeError(f"No valid {category} pose after 200 attempts; no dataset written")
        poses.extend((pose, mirrored_pose))
        categories.extend((category, category + "_mirrored"))
        metrics.extend((validation, mirrored_validation))
        stance_ids.extend((group, group))
        print(
            f"Accepted {index + 1}/{args.poses}: {category}, attempts={attempt + 1}, "
            f"pelvis_height={pose['body_pos_w'][0, 2]:.3f}",
            flush=True,
        )
    arrays = {name: np.stack([pose[name] for pose in poses]).astype(np.float32) for name in poses[0]}
    arrays.update(
        schema_version=np.array(1),
        dataset_kind=np.array("static_poses"),
        fps=np.array(60),
        quaternion_order=np.array("xyzw"),
        joint_names=np.array(sampler.joint_names),
        body_names=np.array(BODY_NAMES),
        clip_starts=np.arange(len(poses)),
        clip_lengths=np.ones(len(poses), dtype=np.int64),
        clip_kinds=np.array(categories),
        joint_limits=sampler.original_limits,
        sampling_limits=np.stack((sampler.lower, sampler.upper), axis=1),
        urdf_sha256=np.array(hashlib.sha256(args.urdf.read_bytes()).hexdigest()),
        seed=np.array(args.seed),
    )
    for name in metrics[0]:
        arrays[name] = np.array([result[name] for result in metrics])
    if args.stance_groups:
        arrays["stance_ids"] = np.array(stance_ids, dtype=np.int64)
        arrays["stance_foot_pos_w"] = np.array(
            [[[0.0, width / 2, 0.0354], [0.0, -width / 2, 0.0354]] for width, _ in stances]
        )
        arrays["stance_foot_quat_w"] = np.stack(
            [Rotation.from_rotvec([[0, 0, yaw], [0, 0, -yaw]]).as_quat() for _, yaw in stances]
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    report = dict(
        original_poses=args.poses,
        total_poses=len(poses),
        sampling_fraction=0.9,
        stance_groups=args.stance_groups,
        nearby_original_poses=args.poses // 2 if args.stance_groups else 0,
        rejections=rejections,
        minimum_collision_clearance_m=float(arrays["collision_clearance_m"].min()),
        minimum_support_margin_m=float(arrays["support_margin_m"].min()),
        minimum_limb_singular_value=float(arrays["minimum_limb_singular_value"].min()),
        pelvis_height_range_m=[
            float(arrays["body_pos_w"][:, 0, 2].min()),
            float(arrays["body_pos_w"][:, 0, 2].max()),
        ],
    )
    report["joint_ranges_rad"] = {
        name: {
            "sampling": [float(sampler.lower[index]), float(sampler.upper[index])],
            "proposed": [
                float(arrays["sampled_joint_pos"][:, index].min()),
                float(arrays["sampled_joint_pos"][:, index].max()),
            ],
            "accepted": [float(arrays["joint_pos"][:, index].min()), float(arrays["joint_pos"][:, index].max())],
        }
        for index, name in enumerate(sampler.joint_names)
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved {len(poses)} validated static poses to {args.output}", flush=True)


if __name__ == "__main__":
    main()
