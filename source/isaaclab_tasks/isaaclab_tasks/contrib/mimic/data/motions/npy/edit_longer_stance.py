"""Append a static hold of the last frame to a torch-pickled retarget motion.

The motion is extended by repeating its final frame for a few seconds, so a policy tracking it
has to keep standing after the motion proper is over instead of being reset the moment it lands.
Everything about the original frames is preserved; only the tail is new.

Pose is copied verbatim from the last frame, and every velocity in the held section is zero --
the robot is meant to be standing still, not coasting at whatever speed it had on the last frame.
Linear velocity and ``dof_vels`` are re-differentiated with the same ``np.gradient`` scheme the
source data uses, so the ramp into the hold is consistent rather than a step. ``global_angular_
velocity`` does not follow that convention in these files, so it is only zeroed inside the hold
and the recorded frames are left untouched.

.. code-block:: bash

    python edit_longer_stance.py -f leap_g1_retargeted_mirror.npy --hold_seconds 5
"""

import argparse
import shutil

import numpy as np
import torch

parser = argparse.ArgumentParser(description="Extend a retargeted motion (.npy torch pickle) with a static stance.")
parser.add_argument("--input_file", "-f", type=str, required=True, help="Path to the input retarget .npy file.")
parser.add_argument("--output_name", type=str, help="Path to the output .npy file (defaults to in-place).")
parser.add_argument("--hold_seconds", type=float, default=5.0, help="Seconds of the last frame to append.")
parser.add_argument("--no_backup", action="store_true", help="Skip writing a .orig backup when overwriting in place.")
args_cli = parser.parse_args()

# velocities are zero while standing still, so they are not simply repeated like the pose is
VELOCITY_KEYS = (
    "global_velocity",
    "global_angular_velocity",
    "dof_vels",
    "global_root_velocity",
    "global_root_angular_velocity",
)


def hold_last_frame(data: dict, hold_frames: int) -> dict:
    """Repeat the final frame ``hold_frames`` times, with the robot at rest throughout the hold."""
    num_frames = data["dof_pos"].shape[0]
    out = {}
    for key, value in data.items():
        if not torch.is_tensor(value) or value.shape[:1] != (num_frames,):
            out[key] = value
            continue
        tail = value[-1:].expand(hold_frames, *value.shape[1:])
        if key in VELOCITY_KEYS:
            tail = torch.zeros_like(tail)
        out[key] = torch.cat([value, tail], dim=0)

    # re-differentiate rather than leave the seam as a step, matching the file's own convention
    fps = data["fps"]
    for key, source in (("global_velocity", "global_translation"), ("dof_vels", "dof_pos")):
        grad = np.gradient(out[source].numpy(), 1.0 / fps, axis=0)
        out[key] = torch.from_numpy(grad).to(out[key].dtype)
    out["global_root_velocity"] = out["global_velocity"][:, 0]
    out["global_root_angular_velocity"] = out["global_angular_velocity"][:, 0]
    return out


def main():
    data = torch.load(args_cli.input_file, map_location="cpu", weights_only=False)
    fps = data["fps"]
    hold_frames = int(round(args_cli.hold_seconds * fps))
    assert hold_frames > 0, "hold_seconds is too short to add a single frame"

    num_frames = data["dof_pos"].shape[0]
    out = hold_last_frame(data, hold_frames)

    # the hold is only as useful as the pose it freezes, so show what it is standing on
    node_names = data["node_names"]
    ankles = [node_names.index(f"{side}_ankle_roll_link") for side in ("left", "right")]
    heights = data["global_translation"][-1, ankles, 2]
    print(f"[INFO]: Holding the last frame for {args_cli.hold_seconds:g}s ({hold_frames} frames at {fps} fps)")
    print(f"[INFO]: {num_frames} -> {out['dof_pos'].shape[0]} frames")
    print(f"[INFO]: last-frame ankle heights: left {heights[0]:.4f} m, right {heights[1]:.4f} m")

    output_name = args_cli.output_name or args_cli.input_file
    if output_name == args_cli.input_file and not args_cli.no_backup:
        shutil.copyfile(args_cli.input_file, args_cli.input_file + ".orig")
        print("[INFO]: Backed up original to", args_cli.input_file + ".orig")
    torch.save(out, output_name)
    print(f"[INFO]: Saved to {output_name}")


if __name__ == "__main__":
    main()
