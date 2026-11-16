# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Validate a recorded MPM run and plot its measured material-on-body forces."""

import argparse
import json
import subprocess
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .provenance import sha256


def evaluate(path):
    meta = json.loads((path / "metadata.json").read_text())
    if meta["status"] != "complete":
        raise ValueError(f"Incomplete run: {path}")
    for name, digest in meta["artifacts"].items():
        if sha256(path / name) != digest:
            raise ValueError(f"Artifact hash differs: {name}")
    with h5py.File(path / "trajectory.h5") as data:
        time = data["force_time"][:]
        force = data["material_force"][:]
        poses = data["body_pose"][:]
        initial = data["particle_position"][0]
        final = data["particle_position"][-1]
        names = [s.decode() for s in data["body_names"][:]]
        active = np.flatnonzero(np.max(np.linalg.norm(force, axis=-1), axis=0) > 1.0)
        total = force[:, active].sum(1)
        # All position samples were checked for finiteness during collection.
        escaped = int(np.count_nonzero(final[:, 2] < -meta["bed_size_m"][2] - 0.05))
        displacement = np.linalg.norm(final - initial, axis=-1)
        checks = {
            "finite_saved_forces": bool(np.isfinite(force).all()),
            "particles_below_tray_floor": escaped,
            "particles_below_tray_floor_pass": escaped == 0,
            "material_contact_present": bool(np.max(np.linalg.norm(total, axis=-1)) > 1.0),
            "initial_and_final_particle_counts_match": len(final) == len(initial),
        }
        motion = {
            "max_particle_displacement_m": float(displacement.max()),
            "mean_particle_displacement_m": float(displacement.mean()),
            "reacting_body_names": [names[i] for i in active],
            "peak_total_material_force_N": float(np.linalg.norm(total, axis=-1).max()),
        }
        if meta["motion"] == "franka" and "rigid_tray_contact_count" in data:
            contacts = data["rigid_tray_contact_count"][:]
            checks["no_robot_tray_contacts"] = bool(np.all(contacts == 0))
            motion["maximum_robot_tray_contacts"] = int(contacts.max())
            motion["samples_with_robot_tray_contacts"] = int(np.count_nonzero(contacts))
        if meta["motion"] in ("g1", "quadruped"):
            root = next(i for i, n in enumerate(names) if n.split("/")[-1] in ("base", "trunk", "pelvis"))
            motion.update(
                base_displacement_m=(poses[-1, root, :3] - poses[0, root, :3]).tolist(),
                minimum_base_height_m=float(poses[200:, root, 2].min()),
            )
            up_z = 1.0 - 2.0 * (poses[200:, root, 3] ** 2 + poses[200:, root, 4] ** 2)
            motion["minimum_base_up_z"] = float(up_z.min())
            checks["robot_remains_upright"] = bool(
                motion["minimum_base_height_m"] > (0.35 if meta["motion"] == "g1" else 0.18) and up_z.min() > 0.65
            )

            checks["robot_moves_forward"] = bool(motion["base_displacement_m"][0] > 0.08)
        plt.rcParams.update(
            {"font.family": "DejaVu Serif", "font.size": 12, "axes.spines.top": False, "axes.spines.right": False}
        )
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.0), sharex=True)
        for axis, (ax, label) in enumerate(zip(axes, ("x", "y", "z"), strict=True)):
            ax.plot(time, total[:, axis], color="#17365D", lw=1.0)
            ax.set(xlabel="Simulation time (s)", ylabel=f"$F_{label}$ (N)", title=f"({chr(97 + axis)}) $F_{label}$")
            ax.grid(alpha=0.15)
        fig.suptitle(f"Implicit MPM · {meta['motion']} · {meta['terrain']}")
        fig.tight_layout(rect=(0, 0.09, 1, 0.94))
        fig.text(
            0.5,
            0.02,
            "Figure 1: Sand-on-collider force in world axes: (a) Fx; (b) Fy; (c) Fz. "
            "Sum over the listed reacting bodies; rigid tray forces excluded.",
            ha="center",
            fontsize=10,
        )
        for extension in ("png", "svg"):
            fig.savefig(path / f"forces.{extension}", dpi=160, facecolor="white")
        plt.close(fig)
    if (path / "simulation.mp4").exists():
        video = json.loads(
            subprocess.check_output(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=nb_frames,r_frame_rate,duration,width,height",
                    "-of",
                    "json",
                    str(path / "simulation.mp4"),
                ]
            )
        )["streams"][0]
        checks["video_matches_recorded_states"] = (
            int(video["nb_frames"]) == len(meta["frame_state_indices"])
            and video["r_frame_rate"] == "50/1"
            and bool(np.isclose(float(video["duration"]), meta["duration_s"]))
        )
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-y",
                "-ss",
                str(min(4.0, meta["duration_s"] * 0.7)),
                "-i",
                str(path / "simulation.mp4"),
                "-frames:v",
                "1",
                str(path / "poster.jpg"),
            ],
            check=True,
        )
    report = {
        "checks": checks,
        "measurements": motion,
        "passed": all(v for k, v in checks.items() if k != "particles_below_tray_floor"),
    }
    (path / "validation.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"run": str(path), **report}, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    if not evaluate(args.run)["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
