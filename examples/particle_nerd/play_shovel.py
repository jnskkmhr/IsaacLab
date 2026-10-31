# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch the trained interactive shovel with ``uv run isaaclab -p``.

The sibling particle_nerd repository supplies this machine's default checkpoint
and initial episode. Override either with --checkpoint or --episode.
"""

import sys
from pathlib import Path

if __name__ == "__main__":
    examples_dir = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(examples_dir))

    from particle_nerd.runner.play import main

    artifacts = examples_dir.parent.parent / "particle_nerd"
    main(
        default_checkpoint=artifacts / "particle_nerd_plan/results/shovel/training/model/best.pt",
        default_episode=artifacts / "experiments/particle_gnn/shovel/reference/wide-circle.hdf5",
    )
