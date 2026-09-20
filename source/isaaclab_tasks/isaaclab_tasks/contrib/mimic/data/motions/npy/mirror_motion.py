"""Symmetrize a torch-pickled retarget motion by mirroring one side onto the other.

The recorded state of the target side (arm + leg) is discarded and replaced by the mirror of
the source side, reflected across the sagittal (x-z) plane of the pelvis. Root, waist and the
source side are left untouched.

Each limb is mirrored in the frame of the body it hangs off (pelvis for the legs, waist_yaw_link
for the arms) rather than in the world, so the result stays correct while the root moves and the
mirrored body poses agree exactly with the sign-flipped joint angles. Linear velocity is
re-differentiated from the mirrored positions with the same ``np.gradient`` scheme the source
data uses; angular velocity is reflected analytically as a pseudovector.

Because the mirror plane is the pelvis sagittal plane, any pelvis roll tilts that plane and the
mirrored leg ends up floating above (or below) the ground the stance leg is on. So by default the
root is de-rolled first: the whole body is rigidly rotated about the pelvis origin until the
pelvis y-axis is horizontal, which leaves every joint angle untouched and makes the mirror plane
vertical. Pass ``--keep_root_roll`` to skip this.

The global root yaw is zeroed too, so the pelvis always faces world +x. These retargets carry a
spurious yaw wobble (tens of degrees over a few frames) that shows up as the base and the planted
foot twisting in place; removing it also pins the mirror plane to a fixed world-aligned plane
instead of one that swings with the heading. The root's translation path is left as recorded.
Pass ``--keep_root_yaw`` to skip this.

The waist yaw joint is zeroed for the same reason: it swings the frame the arms are mirrored in
away from the pelvis sagittal plane, so the mirrored arm no longer matches the source arm's
height and reach. Zeroing it rotates the waist and both arms rigidly about the waist origin,
which leaves every arm joint angle untouched. Pass ``--keep_waist_yaw`` to skip this.

.. code-block:: bash

    python mirror_motion.py -f leap_g1_retargeted.npy --source left
"""

import argparse
import shutil

import numpy as np
import torch

parser = argparse.ArgumentParser(description="Mirror one side of a retargeted motion (.npy torch pickle) onto the other.")
parser.add_argument("--input_file", "-f", type=str, required=True, help="Path to the input retarget .npy file.")
parser.add_argument("--output_name", type=str, help="Path to the output .npy file (defaults to in-place).")
parser.add_argument("--source", type=str, default="left", choices=["left", "right"], help="Side to keep and mirror.")
parser.add_argument("--keep_root_roll", action="store_true", help="Do not de-roll the root before mirroring.")
parser.add_argument("--keep_root_yaw", action="store_true", help="Do not zero the global root yaw before mirroring.")
parser.add_argument("--keep_waist_yaw", action="store_true", help="Do not zero the waist yaw joint before mirroring.")
parser.add_argument("--no_backup", action="store_true", help="Skip writing a .orig backup when overwriting in place.")
args_cli = parser.parse_args()

# kinematic parent of every body in node_names order (-1 = root)
PARENTS = [
    -1,             # pelvis_link
    0, 1, 2, 3, 4, 5,       # left leg:  hip_pitch/roll/yaw, knee, ankle_pitch/roll
    0, 7, 8, 9, 10, 11,     # right leg
    0,              # waist_yaw_link
    13, 14, 15, 16, 17, 18, 19,  # left arm:  shoulder_pitch/roll/yaw, elbow, wrist_roll/pitch/yaw
    13, 21, 22, 23, 24, 25, 26,  # right arm
]

# body index each limb hangs off; the limb is reflected in this body's frame, and it is left as-is
LIMB_BASES = {"leg": 0, "arm": 13}  # pelvis_link, waist_yaw_link

# the joint that yaws the arm mirror frame away from the pelvis sagittal plane
WAIST_BODY = "waist_yaw_link"
WAIST_DOF = "waist_yaw"

# reflection across the sagittal (x-z) plane of the base frame
MIRROR = torch.diag(torch.tensor([1.0, -1.0, 1.0]))


def limb_of(body_name: str) -> str:
    return "leg" if any(k in body_name for k in ("hip", "knee", "ankle")) else "arm"


def counterpart(name: str, source: str) -> str:
    """Name of the body/dof on the opposite side, or the name itself if it is not sided."""
    target = "right" if source == "left" else "left"
    if name.startswith(f"{source}_"):
        return f"{target}_{name[len(source) + 1:]}"
    return name


def dof_mirror_sign(dof_name: str) -> float:
    """Roll (about x) and yaw (about z) flip sign under a y -> -y reflection; pitch (about y) does not."""
    return -1.0 if dof_name.endswith("_roll") or dof_name.endswith("_yaw") else 1.0


def quat_to_mat(q: torch.Tensor) -> torch.Tensor:
    """xyzw quaternion -> rotation matrix, on the trailing dimension."""
    x, y, z, w = q.unbind(-1)
    return torch.stack(
        [
            torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
            torch.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
            torch.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
        ],
        -2,
    )


def mat_to_quat(m: torch.Tensor) -> torch.Tensor:
    """Rotation matrix -> xyzw quaternion, via the numerically stable branch-per-largest-component form."""
    m00, m01, m02 = m[..., 0, 0], m[..., 0, 1], m[..., 0, 2]
    m10, m11, m12 = m[..., 1, 0], m[..., 1, 1], m[..., 1, 2]
    m20, m21, m22 = m[..., 2, 0], m[..., 2, 1], m[..., 2, 2]
    # the four candidate reconstructions, each valid when its own component is the largest
    cands = torch.stack(
        [
            torch.stack([1 + m00 - m11 - m22, m01 + m10, m02 + m20, m21 - m12], -1),
            torch.stack([m01 + m10, 1 - m00 + m11 - m22, m12 + m21, m02 - m20], -1),
            torch.stack([m02 + m20, m12 + m21, 1 - m00 - m11 + m22, m10 - m01], -1),
            torch.stack([m21 - m12, m02 - m20, m10 - m01, 1 + m00 + m11 + m22], -1),
        ],
        -2,
    )
    diag = torch.stack([1 + m00 - m11 - m22, 1 - m00 + m11 - m22, 1 - m00 - m11 + m22, 1 + m00 + m11 + m22], -1)
    best = diag.argmax(-1)
    q = torch.gather(cands, -2, best[..., None, None].expand(*best.shape, 1, 4)).squeeze(-2)
    return q / q.norm(dim=-1, keepdim=True)


def subtree(root: int) -> list[int]:
    """Indices of ``root`` and everything hanging off it. PARENTS is topologically ordered."""
    bodies = [root]
    for i, p in enumerate(PARENTS):
        if p in bodies and i not in bodies:
            bodies.append(i)
    return bodies


def rigid_rotate(out: dict, delta: torch.Tensor, bodies: list[int], pivot: torch.Tensor, fps: float):
    """Rotate ``bodies`` rigidly by the per-frame rotation ``delta`` about the per-frame ``pivot``.

    Because the whole subtree moves together, every joint angle inside it is untouched; only the
    global states and the subtree root's orientation relative to its parent change. Updates ``out``
    in place. ``delta`` is (T, 3, 3) and ``pivot`` is (T, 1, 3).
    """
    dtype = out["global_translation"].dtype
    delta_b = delta.unsqueeze(1)

    # angular velocity of the correction itself, which the bodies pick up on top of their own
    delta_dot = torch.from_numpy(np.gradient(delta.numpy(), 1.0 / fps, axis=0)).to(dtype)
    skew = delta_dot @ delta.transpose(-1, -2)
    delta_w = torch.stack(
        [skew[..., 2, 1] - skew[..., 1, 2], skew[..., 0, 2] - skew[..., 2, 0], skew[..., 1, 0] - skew[..., 0, 1]], -1
    ).unsqueeze(1) / 2

    pos, rot = out["global_translation"][:, bodies], out["global_rotation_mat"][:, bodies]
    out["global_translation"][:, bodies] = pivot + (delta_b @ (pos - pivot).unsqueeze(-1)).squeeze(-1)
    out["global_rotation_mat"][:, bodies] = delta_b @ rot
    out["global_rotation"][:, bodies] = mat_to_quat(out["global_rotation_mat"][:, bodies])
    ang_vel = out["global_angular_velocity"][:, bodies]
    out["global_angular_velocity"][:, bodies] = delta_w + (delta_b @ ang_vel.unsqueeze(-1)).squeeze(-1)
    # re-differentiate rather than transform, to keep the file's own np.gradient convention
    grad = np.gradient(out["global_translation"][:, bodies].numpy(), 1.0 / fps, axis=0)
    out["global_velocity"][:, bodies] = torch.from_numpy(grad).to(dtype)


def deroll_root(data: dict) -> dict:
    """Rigidly rotate the whole body about the pelvis origin until the pelvis y-axis is horizontal.

    Removing the roll is a rotation about the pelvis *x*-axis, so the forward axis is preserved
    exactly and every joint angle (and hence ``local_rotation``, ``dof_pos``, ``dof_vels``) is
    unchanged -- only the global body states and the root orientation move.
    """
    out = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in data.items()}

    # rebuild the root frame keeping its forward axis but forcing the y-axis flat. Done directly
    # from the axes rather than via yaw-pitch-roll, which is near gimbal lock at this motion's pitch.
    root_rot = data["global_rotation_mat"][:, 0]
    fwd = root_rot[..., 0]
    up = torch.tensor([0.0, 0.0, 1.0], dtype=fwd.dtype).expand_as(fwd)
    left_axis = torch.cross(up, fwd, dim=-1)
    assert left_axis.norm(dim=-1).min() > 1e-3, "root is pitched ~90 deg, roll is not well defined"
    left_axis = left_axis / left_axis.norm(dim=-1, keepdim=True)
    flat_rot = torch.stack([fwd, left_axis, torch.cross(fwd, left_axis, dim=-1)], dim=-1)

    delta = flat_rot @ root_rot.transpose(-1, -2)
    # the root itself does not move under the de-roll, so its velocity comes back out bit-identical
    rigid_rotate(out, delta, list(range(len(PARENTS))), data["global_translation"][:, 0:1], data["fps"])
    out["global_root_velocity"] = out["global_velocity"][:, 0]
    out["global_root_angular_velocity"] = out["global_angular_velocity"][:, 0]
    return out


def zero_root_yaw(data: dict) -> dict:
    """Rigidly rotate the whole body about the pelvis origin until the pelvis faces world +x.

    The correction is about the world *z*-axis, so it does not disturb the pelvis roll a preceding
    :func:`deroll_root` removed, and it leaves every joint angle untouched. The root's translation
    path is deliberately left alone: the yaw these retargets carry is a spurious orientation
    wobble rather than real turning, so re-deriving the path from the de-yawed heading would bend
    a straight run into a swerve.
    """
    out = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in data.items()}

    fwd = data["global_rotation_mat"][:, 0, :, 0]
    horiz = fwd[:, :2].norm(dim=-1)
    assert horiz.min() > 1e-3, "root is pitched ~90 deg, yaw is not well defined"
    cos, sin = fwd[:, 0] / horiz, fwd[:, 1] / horiz
    zero = torch.zeros_like(cos)
    one = torch.ones_like(cos)
    # Rz(-yaw), built from the heading's own cos/sin so no atan2 wrapping is involved
    delta = torch.stack(
        [
            torch.stack([cos, sin, zero], -1),
            torch.stack([-sin, cos, zero], -1),
            torch.stack([zero, zero, one], -1),
        ],
        -2,
    )
    rigid_rotate(out, delta, list(range(len(PARENTS))), data["global_translation"][:, 0:1], data["fps"])
    out["global_root_velocity"] = out["global_velocity"][:, 0]
    out["global_root_angular_velocity"] = out["global_angular_velocity"][:, 0]
    return out


def zero_waist_yaw(data: dict) -> dict:
    """Zero the waist yaw joint, carrying the waist and both arms with it.

    This aligns the arm mirror frame with the pelvis frame, so both limbs are reflected across the
    same plane. The arm joint angles are unaffected because the whole waist subtree rotates as one.
    """
    out = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in data.items()}
    waist = data["node_names"].index(WAIST_BODY)
    rot = data["global_rotation_mat"]

    # rotation that takes the waist onto its parent's orientation, i.e. a zero joint angle
    delta = rot[:, PARENTS[waist]] @ rot[:, waist].transpose(-1, -2)
    rigid_rotate(out, delta, subtree(waist), data["global_translation"][:, waist : waist + 1], data["fps"])

    out["local_rotation"][:, waist] = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=out["local_rotation"].dtype)
    dof = data["dof_names"].index(WAIST_DOF)
    out["dof_pos"][:, dof] = 0.0
    out["dof_vels"][:, dof] = 0.0
    return out


def mirror(data: dict, source: str) -> dict:
    node_names, dof_names = data["node_names"], data["dof_names"]
    out = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in data.items()}

    pos, vel = data["global_translation"], data["global_velocity"]
    rot_mat, ang_vel = data["global_rotation_mat"], data["global_angular_velocity"]
    mirror_mat = MIRROR.to(pos.dtype)

    dst_idx = []
    for limb, base in LIMB_BASES.items():
        pairs = [
            (i, node_names.index(counterpart(n, source)))
            for i, n in enumerate(node_names)
            if n.startswith(f"{source}_") and limb_of(n) == limb
        ]
        src, dst = [s for s, _ in pairs], [d for _, d in pairs]
        dst_idx += dst

        base_pos, base_rot = pos[:, base : base + 1], rot_mat[:, base : base + 1]
        base_ang_vel = ang_vel[:, base : base + 1]
        base_rot_t = base_rot.transpose(-1, -2)

        # express the source limb in the base frame, reflect, and map back
        local_pos = (base_rot_t @ (pos[:, src] - base_pos).unsqueeze(-1)).squeeze(-1)
        local_rot = base_rot_t @ rot_mat[:, src]
        local_ang_vel = (base_rot_t @ (ang_vel[:, src] - base_ang_vel).unsqueeze(-1)).squeeze(-1)

        m_pos = base_pos + (base_rot @ (local_pos @ mirror_mat).unsqueeze(-1)).squeeze(-1)
        m_rot = base_rot @ mirror_mat @ local_rot @ mirror_mat
        # angular velocity is a pseudovector, so it picks up an extra sign flip under a reflection
        m_ang_vel = base_ang_vel + (base_rot @ (-local_ang_vel @ mirror_mat).unsqueeze(-1)).squeeze(-1)

        out["global_translation"][:, dst] = m_pos
        out["global_rotation_mat"][:, dst] = m_rot
        out["global_rotation"][:, dst] = mat_to_quat(m_rot)
        out["global_angular_velocity"][:, dst] = m_ang_vel
        # local_rotation is parent-relative; both the body and its parent are reflected in the same
        # base frame, so the reflection carries over to the joint-local rotation unchanged
        out["local_rotation"][:, dst] = mat_to_quat(mirror_mat @ quat_to_mat(data["local_rotation"][:, src]) @ mirror_mat)

    # re-differentiate linear velocity from the mirrored positions, matching the source convention
    fps = data["fps"]
    grad = np.gradient(out["global_translation"][:, dst_idx].numpy(), 1.0 / fps, axis=0)
    out["global_velocity"][:, dst_idx] = torch.from_numpy(grad).to(vel.dtype)

    for i, name in enumerate(dof_names):
        if not name.startswith(f"{source}_"):
            continue
        j = dof_names.index(counterpart(name, source))
        sign = dof_mirror_sign(name)
        out["dof_pos"][:, j] = sign * data["dof_pos"][:, i]
        out["dof_vels"][:, j] = sign * data["dof_vels"][:, i]

    return out


def main():
    data = torch.load(args_cli.input_file, map_location="cpu", weights_only=False)
    assert PARENTS[0] == -1 and len(PARENTS) == len(data["node_names"]), "PARENTS does not match this skeleton"

    if not args_cli.keep_root_roll:
        data = deroll_root(data)
    if not args_cli.keep_root_yaw:
        data = zero_root_yaw(data)
    if not args_cli.keep_waist_yaw:
        data = zero_waist_yaw(data)
    out = mirror(data, args_cli.source)

    output_name = args_cli.output_name or args_cli.input_file
    if output_name == args_cli.input_file and not args_cli.no_backup:
        shutil.copyfile(args_cli.input_file, args_cli.input_file + ".orig")
        print("[INFO]: Backed up original to", args_cli.input_file + ".orig")
    torch.save(out, output_name)
    print(f"[INFO]: Mirrored {args_cli.source} -> other side, saved to {output_name}")


if __name__ == "__main__":
    main()
