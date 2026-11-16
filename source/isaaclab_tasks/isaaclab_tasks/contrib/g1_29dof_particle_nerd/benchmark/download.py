# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Download the exact policy checkpoints used in the saved foot-force comparison."""

import argparse
import hashlib
import json
from pathlib import Path

import wandb


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifests = Path(__file__).parent / "data/checkpoints"
    api = wandb.Api()
    for manifest in sorted(manifests.glob("*/metadata.json")):
        metadata = json.loads(manifest.read_text())
        run = api.run(f"jnskkmr/{metadata['project']}/{metadata['run_id']}")
        name = Path(metadata["checkpoint"]).name
        destination = args.output / metadata["run_id"]
        destination.mkdir(parents=True, exist_ok=True)
        run.file(name).download(root=str(destination), replace=True)
        path = destination / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != metadata["sha256"]:
            raise ValueError(f"Checkpoint checksum mismatch: {path}")
        # Use the archived training configuration, even if the live run later changes.
        (destination / "training_config.json").write_bytes((manifest.parent / "training_config.json").read_bytes())
        print(path)


if __name__ == "__main__":
    main()
