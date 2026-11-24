# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compare the kinetic energy of the sand between the Particle NeRD run and the MPM run of a benchmark case.

Reads ``nerd.npz`` and ``mpm.npz`` that ``run_case.py`` wrote into the case folder. Particle speeds are finite
differences of the saved positions, and the kinetic energy of a particle is ``0.5 * m * v**2``. The script writes

- ``kinetic_energy.png``: the total kinetic energy over time, and the share of the time-integrated kinetic energy
  at each height;
- ``kinetic_energy.json``: the time integral [J s] and the peak [J] of the total kinetic energy, per solver.

Run from the repository root:

    uv run python scripts/neural/benchmark/kinetic_energy.py logs/neural/benchmark/box-push
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PARTICLE_MASS = 1600.0 * 0.032**3
"""Mass of one sand particle of the example bed [kg]: density times the cube of the particle spacing."""
PHYSICS_DT = 0.005


def speeds(q: np.ndarray, dt: float) -> np.ndarray:
    """Particle speeds [T-1, N] from positions [T, N, 3] saved ``dt`` apart."""
    return np.linalg.norm(np.diff(q, axis=0), axis=-1) / dt


def frame_dt(run) -> float:
    """Time between two saved frames of a run of ``run_case.py`` [s]."""
    return float(run["step"][1] - run["step"][0]) * PHYSICS_DT


def compare(case_dir: Path) -> dict:
    """Write the figure and the statistics of one case; return the statistics."""
    runs = {name: np.load(case_dir / f"{name}.npz") for name in ("nerd", "mpm")}
    dt = frame_dt(runs["nerd"])
    count = min(len(runs["nerd"]["q"]), len(runs["mpm"]["q"]))
    q_nerd, q_mpm = runs["nerd"]["q"][:count], runs["mpm"]["q"][:count]
    v_nerd, v_mpm = speeds(q_nerd, dt), speeds(q_mpm, dt)
    t = (np.arange(count - 1) + 0.5) * dt
    ke_nerd = 0.5 * PARTICLE_MASS * (v_nerd**2).sum(1)
    ke_mpm = 0.5 * PARTICLE_MASS * (v_mpm**2).sum(1)

    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2), dpi=80)
    ax[0].semilogy(t, np.maximum(ke_nerd, 1e-5), label="Particle NeRD", lw=2)
    ax[0].semilogy(t, np.maximum(ke_mpm, 1e-5), label="MPM", lw=2)
    ax[0].set_xlabel("time [s]")
    ax[0].set_ylabel("total kinetic energy [J]")
    ax[0].set_title("Kinetic energy over time", fontweight="bold")
    ax[0].legend()
    ax[0].grid(alpha=0.3)
    # where the energy sits vertically (time-integrated)
    z_max = float(max(q_nerd[..., 2].max(), q_mpm[..., 2].max())) + 0.02
    edges = np.arange(-0.32, z_max + 0.02, 0.02)
    centres = 0.5 * (edges[1:] + edges[:-1])
    for q, v, name in ((q_nerd, v_nerd, "Particle NeRD"), (q_mpm, v_mpm, "MPM")):
        z = 0.5 * (q[1:, :, 2] + q[:-1, :, 2])
        weight = 0.5 * PARTICLE_MASS * v**2
        hist, _ = np.histogram(z.ravel(), bins=edges, weights=weight.ravel())
        ax[1].plot(hist / max(hist.sum(), 1e-12), centres, label=name, lw=2)
    ax[1].set_xlabel("share of time-integrated kinetic energy")
    ax[1].set_ylabel("z position [m]  (floor −0.30, bed surface ≈ 0)")
    ax[1].set_ylim(-0.32, edges[-1])
    ax[1].set_title("Energy by z position", fontweight="bold")
    ax[1].legend()
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(case_dir / "kinetic_energy.png")
    plt.close(fig)

    stats = {
        "ke_int_nerd": float(np.trapezoid(ke_nerd, t)),
        "ke_int_mpm": float(np.trapezoid(ke_mpm, t)),
        "ke_peak_nerd": float(ke_nerd.max()),
        "ke_peak_mpm": float(ke_mpm.max()),
    }
    (case_dir / "kinetic_energy.json").write_text(json.dumps(stats, indent=1))
    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("case_dir", type=Path, help="Case folder with nerd.npz and mpm.npz.")
    args = parser.parse_args(argv)
    for name in ("nerd", "mpm"):
        if not (args.case_dir / f"{name}.npz").is_file():
            parser.error(f"{args.case_dir / (name + '.npz')} is missing; run run_case.py --solver {name} first")
    stats = compare(args.case_dir)
    print(args.case_dir.name, json.dumps({key: round(value, 3) for key, value in stats.items()}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
