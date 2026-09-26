"""Extend a motion npz's stance by holding a settled frame at the start and/or the end of the clip.

Takes a chosen "settled" frame at each end (default: the first and last frames) and repeats it, with
all velocities zeroed so the hold is a true static stance rather than a coast at whatever speed the
held frame happened to have. The recorded frames themselves are copied through untouched.

The point is to give a motion-tracking policy a stretch of clip where it only has to stand still, so
the stance phase can be gated to standing/balance rewards (see `MotionCommandCfg.stance_phase_ranges`,
which this script prints ready to paste).

.. code-block:: bash

    python extend_stance.py -f leap_wide_g1_retargeted_mirror_right.npz --lead_seconds 5 --tail_seconds 5
"""

import argparse
import os

import numpy as np

POSE_KEYS = ("joint_pos", "body_pos_w", "body_quat_w")
VELOCITY_KEYS = ("joint_vel", "body_lin_vel_w", "body_ang_vel_w")

parser = argparse.ArgumentParser(description="Extend a motion npz's stance phase by holding a settled frame.")
parser.add_argument("--input_file", "-f", type=str, required=True, help="Path to the input motion npz file.")
parser.add_argument("--output_name", type=str, help="Path to the output npz file.")
parser.add_argument("--lead_seconds", type=float, default=0.0, help="Seconds of stance to prepend.")
parser.add_argument("--tail_seconds", type=float, default=0.0, help="Seconds of stance to append.")
parser.add_argument(
    "--lead_frame",
    type=int,
    default=0,
    help="Index of the frame to hold before the clip (default: 0, the first frame).",
)
parser.add_argument(
    "--tail_frame",
    type=int,
    default=-1,
    help="Index of the frame to hold after the clip (default: -1, the last frame). Pick one from the settled tail.",
)
parser.add_argument("--force", action="store_true", help="Allow overwriting an existing output file.")
args_cli = parser.parse_args()

if not args_cli.output_name:
    args_cli.output_name = args_cli.input_file.rsplit(".", 1)[0] + "_extended_stance.npz"
if os.path.exists(args_cli.output_name) and not args_cli.force:
    raise SystemExit(f"[ERROR] {args_cli.output_name} already exists. Pass --output_name or --force.")

data = dict(np.load(args_cli.input_file))
fps = int(np.asarray(data["fps"]).reshape(-1)[0])
num_frames = data["joint_pos"].shape[0]
lead_frames = int(round(args_cli.lead_seconds * fps))
tail_frames = int(round(args_cli.tail_seconds * fps))
if lead_frames == 0 and tail_frames == 0:
    raise SystemExit("[ERROR] Nothing to do: pass --lead_seconds and/or --tail_seconds.")


def hold(key: str, frame_index: int, count: int) -> np.ndarray:
    """`count` copies of frame `frame_index`, at rest for the velocity buffers."""
    if key in VELOCITY_KEYS:
        return np.zeros((count,) + data[key].shape[1:], dtype=data[key].dtype)
    return np.repeat(data[key][frame_index : frame_index + 1], count, axis=0)


for key in POSE_KEYS + VELOCITY_KEYS:
    pieces = []
    if lead_frames:
        pieces.append(hold(key, args_cli.lead_frame, lead_frames))
    pieces.append(data[key])
    if tail_frames:
        pieces.append(hold(key, args_cli.tail_frame % num_frames, tail_frames))
    data[key] = np.concatenate(pieces, axis=0)

total = data["joint_pos"].shape[0]
print(
    f"[INFO] {num_frames} -> {total} frames at {fps} fps"
    f" ({num_frames / fps:.2f}s -> {total / fps:.2f}s)"
)
if lead_frames:
    root_z = data["body_pos_w"][0, 0, 2]
    print(f"[INFO] lead:  frame {args_cli.lead_frame} held for {lead_frames} frames (root height {root_z:.3f}m)")
if tail_frames:
    root_z = data["body_pos_w"][-1, 0, 2]
    print(f"[INFO] tail:  frame {args_cli.tail_frame} held for {tail_frames} frames (root height {root_z:.3f}m)")

np.savez(args_cli.output_name, **data)
print(f"[INFO] Saved to {args_cli.output_name}")

# The held sections are stance by construction; report them in the phase units the command term
# wants. The clip's own settled tail is not included here -- measure that separately if you want it
# folded into the same range.
ranges = []
if lead_frames:
    ranges.append((0.0, (lead_frames - 1) / (total - 1)))
if tail_frames:
    ranges.append(((total - tail_frames) / (total - 1), 1.0))
print("[INFO] stance_phase_ranges=[" + ", ".join(f"({a:.3f}, {b:.3f})" for a, b in ranges) + "]")
