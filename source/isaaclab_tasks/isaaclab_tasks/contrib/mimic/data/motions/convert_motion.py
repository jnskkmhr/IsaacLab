# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Convert already-retargeted G1 exports to labeled Mimic NPZ files."""

import argparse
import copy
import json
from pathlib import Path

from isaaclab_tasks.contrib.mimic.data.motions.dataset_cfg import DATASET_CFGS
from isaaclab_tasks.contrib.mimic.data.motions.dataset_manager import MOTION_DATA_DIR, DatasetManager


def main() -> None:
    """Load an export through a dataset term and save its canonical motion arrays."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASET_CFGS, required=True)
    parser.add_argument("--input_file", type=Path, required=True)
    parser.add_argument("--output_name", required=True, help="Relative .npz path under output_dir.")
    parser.add_argument("--output_dir", type=Path, default=MOTION_DATA_DIR)
    parser.add_argument(
        "--params", type=Path, help="JSON file with importer parameters, e.g. joint_names and field_map."
    )
    parser.add_argument("--input_fps", type=float)
    parser.add_argument("--output_fps", type=float, default=60.0)
    parser.add_argument("--quaternion_order", choices=("xyzw", "wxyz"))
    parser.add_argument("--robot_xml", help="Robot/scene MJCF for forward kinematics of root/joint-only exports.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    cfg = copy.deepcopy(DATASET_CFGS[args.dataset])
    if args.params is not None:
        cfg.params.update(json.loads(args.params.read_text()))
    for name in ("input_fps", "output_fps", "quaternion_order", "robot_xml"):
        if getattr(args, name) is not None:
            cfg.params[name] = getattr(args, name)
    manager = DatasetManager({args.dataset: cfg}, output_dir=args.output_dir)
    motion = manager.load(args.dataset, args.input_file)
    output = manager.save(motion, args.output_name, overwrite=args.overwrite)
    print(f"Saved {len(motion['joint_pos'])} frames at {motion['fps'].item():g} Hz to {output}")


if __name__ == "__main__":
    main()
