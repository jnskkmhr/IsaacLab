# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Summarise and sanity-check dataset episodes; optionally plot the rigid-body signals."""

import argparse
from pathlib import Path

import h5py
import numpy as np

from .hdf5_io import SCHEMA
from .summarize import BARE_BED_LIMIT_MM_PER_10S, bare_bed_drift


def summarize(path: Path) -> dict:
    with h5py.File(path, "r") as f:
        a = f.attrs
        ok = bool(a.get("complete", False)) and a["schema_version"] == SCHEMA
        pos = f["particles/pos"]
        first, last = pos[0], pos[-1]
        drift = bare_bed_drift(f, first, last)
        row = {
            "file": str(path),
            "category": a["category"],
            "motion": a["motion"],
            "shape": a["shape"],
            "split": a["split"],
            "steps": int(a["num_steps"]),
            "particles": int(a["num_particles"]),
            "MB": round(path.stat().st_size / 1e6, 1),
            "finite": bool(np.isfinite(last).all() and np.isfinite(first).all()),
            "z_min": float(last[:, 2].min()),
            "z_max": float(last[:, 2].max()),
            "mean_dz": float((last[:, 2] - first[:, 2]).mean()),
            "bare_bed_dz_mm_per_10s": drift,
            "bed_settled": drift is None or abs(drift) <= BARE_BED_LIMIT_MM_PER_10S,
            "ok": ok,
        }
        if "body" in f and f["body/pose"].ndim == 3:
            pose, wrench = f["body/pose"][:], f["body/contact_wrench"][:]
            row.update(
                bodies=int(pose.shape[1]),
                joints=int(f["joints/pos"].shape[1]) if "joints" in f else 0,
                base_travel=float(np.linalg.norm(pose[-1, 0, :2] - pose[0, 0, :2])),
                max_contact_f=float(np.linalg.norm(wrench[..., :3], axis=-1).max()),
                finite=row["finite"] and bool(np.isfinite(pose).all() and np.isfinite(wrench).all()),
            )
        elif "body" in f:
            pose, wrench = f["body/pose"][:], f["body/contact_wrench"][:]
            row.update(
                obj_z0=float(pose[0, 2]),
                obj_z_end=float(pose[-1, 2]),
                obj_xy_travel=float(np.linalg.norm(pose[-1, :2] - pose[0, :2])),
                max_contact_fz=float(wrench[:, 2].max()),
                finite=row["finite"] and bool(np.isfinite(pose).all() and np.isfinite(wrench).all()),
            )
        return row


def plot(paths: list[Path], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    object_paths = []
    for p in paths:
        with h5py.File(p, "r") as f:
            if "body" in f and f["body/pose"].ndim == 2:
                object_paths.append(p)
    fig, axes = plt.subplots(len(object_paths), 3, figsize=(13, 2.1 * len(object_paths)), squeeze=False, dpi=70)
    for row, p in enumerate(object_paths):
        with h5py.File(p, "r") as f:
            t, pose = f["time"][:], f["body/pose"][:]
            ext, contact = f["body/ext_wrench"][:], f["body/contact_wrench"][:]
            title = f"{f.attrs['category']} {f.attrs['motion']} {f.attrs['shape']} ({p.stem})"
        axes[row, 0].plot(t, pose[:, 0], label="x")
        axes[row, 0].plot(t, pose[:, 1], label="y")
        axes[row, 0].plot(t, pose[:, 2], label="z")
        axes[row, 0].set_ylabel(title, fontsize=7)
        axes[row, 1].plot(t[1:], ext[:, 2])
        axes[row, 2].plot(t[1:], contact[:, 2])
        if row == 0:
            axes[row, 0].set_title("object position [m]")
            axes[row, 0].legend(fontsize=6)
            axes[row, 1].set_title("applied force z [N]")
            axes[row, 2].set_title("sand-on-body force z [N]")
    for ax in axes[-1]:
        ax.set_xlabel("time [s]")
    fig.tight_layout()
    fig.savefig(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--plot", type=Path)
    args = parser.parse_args()
    paths = sorted(args.root.rglob("*.hdf5"))
    total = 0.0
    drifting = []
    for p in paths:
        row = summarize(p)
        total += row["MB"]
        extras = {k: round(v, 3) for k, v in row.items() if k.startswith(("obj_", "max_"))}
        drift = row["bare_bed_dz_mm_per_10s"]
        if not row["bed_settled"]:
            drifting.append(str(p.relative_to(args.root)))
        print(
            f"{p.relative_to(args.root)!s:42s} {row['motion']:10s} {row['shape']:9s} T={row['steps']:5d} "
            f"N={row['particles']:6d} {row['MB']:8.1f} MB finite={row['finite']} ok={row['ok']} "
            f"sand z[{row['z_min']:.3f},{row['z_max']:.3f}] dz={row['mean_dz']:.4f} "
            f"bare bed {'n/a' if drift is None else f'{drift:+.1f}'} mm/10s {extras}"
        )
    print(f"{len(paths)} files, {total / 1000:.2f} GB")
    print(f"bare bed moving more than {BARE_BED_LIMIT_MM_PER_10S:g} mm per 10 s: {len(drifting)} files {drifting[:10]}")
    if args.plot:
        plot(paths, args.plot)


if __name__ == "__main__":
    main()
