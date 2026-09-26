# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Export motion joint positions as a headerless CSV in the G1 command joint order."""

import argparse
from pathlib import Path

import numpy as np

from isaaclab_tasks.contrib.mimic.config.g1_29dof.env_cfg.commands_cfg import JOINT_NAMES


def convert(npz_path: str | Path, csv_path: str | Path | None = None) -> Path:
    """Export joint positions in the order used by the G1 reference command.

    Args:
        npz_path: Motion archive containing ``joint_pos`` and ``joint_names``.
        csv_path: Output path. Defaults to ``<input_stem>_joint_pos.csv`` beside the input.

    Returns:
        Path to a headerless CSV containing joint positions [rad], shape [N, D],
        where N is the number of motion frames and D is ``len(JOINT_NAMES)``.

    Raises:
        ValueError: Joint positions or joint-name metadata are missing or inconsistent.
    """
    npz_path = Path(npz_path)
    csv_path = Path(csv_path) if csv_path is not None else npz_path.with_name(f"{npz_path.stem}_joint_pos.csv")
    if csv_path.resolve() == npz_path.resolve():
        raise ValueError("Output CSV must differ from the input NPZ path.")
    with np.load(npz_path, allow_pickle=False) as data:
        if "joint_pos" not in data or "joint_names" not in data:
            raise ValueError("NPZ must contain joint_pos and joint_names to determine the source column order.")
        joint_pos = data["joint_pos"]
        joint_names = data["joint_names"]
        if joint_pos.ndim != 2:
            raise ValueError(f"Expected joint_pos with shape [N, D], got {joint_pos.shape}.")
        if joint_names.ndim != 1 or len(joint_names) != joint_pos.shape[1]:
            raise ValueError("joint_names must contain one name per joint_pos column.")
        source_names = joint_names.tolist()
        if len(set(source_names)) != len(source_names):
            raise ValueError("joint_names must be unique.")
        missing = [name for name in JOINT_NAMES if name not in source_names]
        if missing:
            raise ValueError(f"NPZ is missing required G1 joints: {missing}.")
        positions = joint_pos[:, [source_names.index(name) for name in JOINT_NAMES]]
    np.savetxt(csv_path, positions, delimiter=",")
    return csv_path


def main() -> None:
    """Convert one motion archive using command-line paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_file", "-f", type=Path, required=True, help="Input NPZ with joint-name metadata.")
    parser.add_argument("--output_file", "-o", type=Path, help="Output CSV; defaults to <input_stem>_joint_pos.csv.")
    args = parser.parse_args()
    csv_path = convert(args.input_file, args.output_file)
    print(f"Saved joint positions in JOINT_NAMES order to {csv_path}")


if __name__ == "__main__":
    main()
