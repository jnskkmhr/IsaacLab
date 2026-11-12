# Newton rough locomotion baselines

Reference: W&B `g1_29dof_contact_benchmark/binkrmei`.

| Setting | Rigid rough | RFT rough |
|---|---|---|
| Task | `IsaacContrib-Velocity-Rough-G1-29dof-Rigid-Newton` | `IsaacContrib-Velocity-Rough-G1-29dof-SoftContact-Newton` |
| Package | `g1_29dof_rigid_newton` | `g1_29dof_soft_contact_newton` |
| Contact | Newton MJWarp | Newton MJWarp + analytical 3D RFT |
| Surface grid | Same seeded 10-row × 10-column surface | Same surface; alternate columns have buried rigid support |
| Rigid support depth | 0 m | 0–0.12 m over terrain rows in RFT columns |
| Base-height reward | 0.75 m above scanned surface | 0.75 m above scanned surface |
| Foot clearance | 0.10 m + standing ankle offset above each foot's scanned surface | Same |

All 27 reward weights, the original robot asset and PD gains, action scaling, commands,
common randomization, physics timestep/substeps, PPO hyperparameters, and camera settings
are preserved from `binkrmei`. This includes its foot-separation reward (-2.0 at 0.20 m).
The rigid task replaces hybrid contact observations/rewards with their rigid-sensor branch
and omits only RFT material randomization.

Actor history has 930 values, critic history 940, and privileged history 160. The seven
command values are current-only. Actor and critic inputs total 937 and 1107 respectively.
Symmetry augmentation and mirror loss (0.1) are enabled. No terrain scan is added to the actor.

Terrain types and parameters match the main checkout's moving-patch training generator:
20% each wave (amplitude 0.1–0.4 m, four waves), random roughness (0.02–0.10 m),
pyramid slopes and inverted slopes (0–0.4), and flat. Tiles are 8×8 m, heightfield resolution
0.1 m horizontally / 0.005 m vertically, with a 5 m border. Both baselines use ten columns
(two per terrain type) instead of the reference's five, so each type can have rigid and RFT
columns. The two columns of a type are independently generated, not identical copies.
Both tasks start at terrain row zero and progress with the moving-patch distance curriculum.
Command and reward-standard-deviation curricula retain the `binkrmei` schedules.

Rough-only corrections:

- Reset Z is sampled at the randomized robot XY, preserving 0.76 m root clearance.
- Base-height reward uses a 0.2×0.2 m pelvis scan; foot measurements use one ray per ankle.
- Rays start above the body in the yaw-aligned pattern, so pitch/roll does not move them sideways.
- Sensors explicitly use the existing mesh-targeted Newton ray caster to exclude buried support.
- RFT queries the visible surface mesh directly. The former physics-model query could hit the
  buried support instead (0.1067 m error reproduced); the corrected query agreed with the
  reference surface to within 0.000006 m in the local probe. Flat RFT's z=0 path is unchanged.
- Leaving the finite terrain ends the episode as a timeout, before height queries leave the map.

Validation: config comparisons against the flat reference, finite local PPO updates with video,
and `test_rft_rough_surface.py` covering a surface at z=0.30 m above support at z=0.18 m.
These are setup checks, not evidence of converged gait quality. Full training was submitted after review on November 12, 2026: `isaac-lab-g1-rigid-rough-6` and `isaac-lab-g1-rft-rough-1`.

See `osmo/osmo_contact_benchmark.md` for the prepared four-GPU commands.
