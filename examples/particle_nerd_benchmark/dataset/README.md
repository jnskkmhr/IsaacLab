# Particle NeRD Dataset v1

Reference sand trajectories simulated with implicit MPM (Newton), one HDF5 file per episode. They are the training and
evaluation data for Particle NeRD. MPM is the data source only; nothing here is produced by a learned model.

- **1,218 episodes** on three terrains (`flat`, `slope`, `wave`): 1,008 for training and validation and 210 for testing.
- **14 motions per terrain**: 12 object or sand-only motions plus two robots (Franka shovel push, G1 walking).
- Objects vary in type (cube, box, sphere, cylinder) and size (small, medium, large).
- Every bed is **0.30 m deep** (nominal) and is pre-settled before recording.

This file lives at the dataset root. `manifest.jsonl` lists every episode, and `splits.json` counts them per split.

## Directory structure

```
<dataset>/
├── README.md
├── manifest.jsonl                  # one line per planned episode: path, split, terrain, motion, shape, size, ...
├── splits.json                     # episode counts per split and terrain
├── terrain/<terrain>.json          # bed size, tray boxes, spacing, material, gravity (one per terrain)
├── terrain/<terrain>_robot_<robot>.json
├── logs/                           # collection logs and one JSON line of statistics per file
├── docs/{gifs,videos}/             # example previews
└── <split>/                        # train | validation | test
    └── <terrain>/                  # flat | slope | wave
        ├── lateral_push/NNNN.hdf5
        ├── vertical_push/NNNN.hdf5
        ├── impact/NNNN.hdf5
        ├── rest/NNNN.hdf5
        ├── freefall/NNNN.hdf5
        ├── rest_bed/NNNN.hdf5
        └── robot/{franka,g1}/NNNN.hdf5
```

`NNNN` is the episode index within its category (or robot) and terrain; it is unique across splits, so files can be
merged without clashes. Each file stores its `split` and every sampled parameter (`params` attribute).

## Splits

Every (terrain, motion) has 29 episodes: 21 train, 3 validation and 5 test.

| Split | Per terrain and motion | Per terrain | Total |
|---|---|---|---|
| train | 21 | 294 | 882 |
| validation | 3 | 42 | 126 |
| test | 5 | 70 | 210 |

Train plus validation is 1,008 episodes. The test set is 3 terrains × 14 motions × 5 = 210. Object shape and size are
cycled so that the training episodes of a motion cover all 12 (shape, size) combinations; validation and test cover
subsets of them. The split follows the order of the episode within its motion, so there is no overlap by construction.

## Terrains

| Terrain | Bed (x, y, depth) | Surface |
|---|---|---|
| `flat` | 2.0 × 1.2 × 0.30 m | flat |
| `slope` | 2.0 × 1.2 × 0.30 m | ±10° along x; the sign alternates between episodes (`terrain_variant`: `slope_p10`, `slope_m10`) |
| `wave` | 2.0 × 1.2 × 0.30 m | 0.06 m amplitude, 1.2 m wavelength along x |

The robot beds are larger: Franka 1.2 × 1.0 m and G1 3.6 × 1.6 m, also 0.30 m deep. On `slope` the Franka bed is ±10°
and the G1 bed is ±5°, because the G1's 3.6 m bed would run out of sand depth at 10°. The bed sits in a rigid tray
(floor and four walls). Origin: tray centre on the nominal surface (z = 0). `terrain/<terrain>.json` has the geometry.

## Categories and motions

Shape, size, yaw and start position are drawn per episode. Episodes are 200 Hz (`dt` = 0.005 s) and start after the
pre-settle below.

**Objects.** Shapes `cube`, `box`, `sphere` and `cylinder`, each at three scales (0.75, 1.0, 1.3), so 12 object types:

| Shape | Size at scale 1 (x, y, z) | Mass at scale 1 |
|---|---|---|
| cube | 0.18 m edge | 2.0 kg |
| box | 0.24 × 0.18 × 0.12 m | 2.0 kg |
| sphere | 0.20 m diameter | 2.0 kg |
| cylinder | 0.20 m diameter, 0.12 m tall, vertical axis | 2.0 kg |

Sizes scale linearly and the mass with the cube of the scale (0.84, 2.0 and 4.4 kg).

| Category | Motion | Description | Duration |
|---|---|---|---|
| `lateral_push` | `push_x`, `push_y` | Object starts on the sand with its bottom 2–6 cm below the surface, stays still for the first second, then is pushed along ±x or ±y at constant height (travel 0.33–0.63 of the bed length). Prescribed pose, no vertical motion. | 6 s |
| | `circle_cw`, `circle_ccw` | Same, along a circle of radius 0.18–0.28 m, clockwise or counter-clockwise. | 6 s |
| `vertical_push` | `push_down` | From 0.30 m above the bed the object descends at a constant 0.10–0.40 m/s to a depth of 3–10 cm below the surface and holds. Prescribed pose. | 6 s |
| | `pull_up` | Pressed in at 0.4 m/s to the same depth, holds 0.3–0.8 s, then rises at a constant 0.15–0.40 m/s back to the start height. | 6 s |
| `impact` | `drop` | Released at 1 s from 0.18–0.35 m at zero velocity, then free (dynamic body). | 6 s |
| | `throw` | Released at 1 s with a random horizontal speed (0.4–1.6 m/s), upward speed 0–0.5 m/s and optional spin. | 6 s |
| `rest` | `rest` | Placed on the sand surface (bottom 5 mm below the nominal surface) at zero velocity with no external force. Only vertical motion (sinking) is allowed: x, y and orientation are held every step. | 10 s |
| `freefall` | `drop`, `toss` | A block of sand (cube, sphere or cylinder, 0.34–0.52 m wide) released 0.06–0.30 m above the bed and dropped or tossed with 0.4–1.6 m/s. It lands on the terrain's own bed; no rigid body. | 6 s |
| `rest_bed` | `rest` | The settled bed alone. Nothing touches it (the rigid object of the scene stays parked 5 m above the bed and is not recorded). | 10 s |

Release heights are of the object centre above the local surface. In `freefall` the block particles come after the bed
particles in `particles/pos`, and `num_block_particles` gives their count.

### Motion type and coupling

Every file has a `motion_type` attribute that says how the object was driven:

| `motion_type` | Categories | Meaning | Use |
|---|---|---|---|
| `kinematic` | `lateral_push`, `vertical_push` | Pose and velocity are prescribed each step. The sand reacts to the object, the object does not react to the sand (one-way coupling). | One-way (open-loop) evaluation |
| `dynamic` | `impact`, `rest`, `robot/*` | A free rigid body or an articulated robot (MuJoCo Warp) coupled to MPM both ways (two-way coupling). | One-way and two-way (closed-loop) evaluation |
| `none` | `freefall`, `rest_bed` | No rigid body. | Sand-only dynamics |

Open-loop evaluation replays the recorded `body/pose` and `body/vel` into the model and compares particles with
MPM. It works for every file with a `body/` group. Closed-loop evaluation needs `dynamic` files, because the
reference body motion has to come from the coupling.

## Robots

`<split>/<terrain>/robot/<robot>/NNNN.hdf5`. Two robots, on every terrain:

| Robot | Bed (x, y, depth) | Particles (flat) | Duration | Controller | Control rate | Action |
|---|---|---|---|---|---|---|
| `franka` | 1.2 × 1.0 × 0.30 m | 10,323 | 10 s | Franka Panda with a welded open-box shovel, three long arc sweeps around the base | 200 Hz | arm joint position targets [rad], 7 |
| `g1` | 3.6 × 1.6 × 0.30 m | 50,400 | 6 s | Unitree G1 walking, saved MPM-trained RL policy, forward command 0.25–0.45 m/s | 50 Hz | joint-position policy action, 29 |

**Franka shovel.** The tool is an open box (floor 0.22 × 0.28 m, back wall and two side walls 0.10 m high, 0.85 kg in
total) welded to the hand, with its open front facing the direction of travel. The robot stands at x = -0.72 m beside
the tray with its hand pointing down, which limits how far it can reach away from the base. A straight push along x
therefore only moves 10–15 cm, so each episode sweeps the shovel along three arcs around the base at different radii
(0.42–0.62 m), each 0.4–0.6 m long, lifting between them. Joint targets come from an analytic IK (`franka_ik.py`), and
the arm uses soft joint gains (stiffness 800, damping 80, effort limit 400 N m). The base joint still lags the
command by 10–30° while the shovel is loaded, so the achieved arcs are shorter than the planned ones. On `slope` and
`wave` the sweeps follow the local surface height.

**G1.** The policy was trained on a flat bed, not on slopes or waves. On `slope` and `wave` the robot starts on the
local surface and may stumble or fall; the file statistics (`logs/*.files.jsonl`: `min_up_axis`, `base_travel`) show
whether it stayed upright and moved.

Robot files use a body axis and add joint and control groups:

```
robot/<robot>/NNNN.hdf5
├── attrs     as below, plus body_names, joint_names, policy (checkpoint path or null), control_decimation
├── time              [T]
├── particles/pos     [T, N, 3]
├── body/
│   ├── pose            [T, B, 7]    every rigid body, xyz + quaternion xyzw
│   ├── vel             [T, B, 6]    linear xyz, angular xyz (world frame)
│   └── contact_wrench  [T-1, B, 6]  MPM sand-on-body wrench per body (impulse / dt), tray excluded
├── joints/{pos, vel}   [T, J]
└── control/action      [T-1, A]     the action applied at each physics step
```

## Pre-settle

The bed is generated as a jittered lattice and slumps for about a second. To keep that transient out of the data, the
bed is simulated for `settle_s` = 2 s before recording starts, so `t = 0` is the settled bed.

Object and sand-only episodes (`lateral_push`, `vertical_push`, `impact`, `rest`, `rest_bed`, `freefall`) are collected
by one script (`collect_objects.py`) in one scene: the same sand material, tray colliders and MPM solver settings. Only
the bed surface and the object (shape, size, motion) differ.

- Objects: the object is parked 5 m above the bed during the settle (held up by a wrench when dynamic, prescribed
  when kinematic) and then placed at its start pose with zero velocity.
- `rest_bed`: the object stays parked for the whole episode.
- `freefall`: the block's particles wait at the release position while the bed settles (they are put back after every
  step, so they neither fall nor touch the bed) and are released at `t = 0`.
- Robots: the settle is a sand-only run with a separate MPM setup (`presettle.py`) before the robot is loaded, because
  a robot cannot be parked above the bed. The robot starts in contact with the sand, and the first frames contain a
  transient (about 20 mm/s mean particle speed at `t = 0`, against 1–3 mm/s for objects).

A settled bed still moves a little where nothing touches it: 3–5 mm in 10 s on `flat` and 5–10 mm on `slope` and
`wave`. `logs/*.files.jsonl` reports this per file as `bare_bed_dz_mm_per_10s` (the bed surface away from every body
and from the released sand block) and sets `bed_settled` to false above 12 mm per 10 s.

### Version 1.1 (`freefall`, `rest_bed`)

In version 1 these two categories were collected by a separate sand-only script whose MPM settings differed from the
object scene. Its bed had not settled after 2 s and kept sinking 13 mm (`flat`) to 28 mm (`slope`) in 10 s, while the
bed of the object episodes moved 3–6 mm. Version 1.1 regenerates both categories (261 files) with the shared scene.
Do not use the version 1 `freefall` and `rest_bed` files together with the object episodes.

## Material and particles

Sand material (all episodes): density 1600 kg/m³, friction 0.48, Young's modulus 1e15 Pa, Poisson ratio 0.3, yield
pressure 1e15, no hardening, dilatancy or viscosity. Particle spacing 0.032 m (voxel 0.04 m), jittered within 20 % of
the spacing; mass 1600 · 0.032³ kg per particle. A `flat` object bed has 20,646 particles (`slope` and `wave`
beds 21,300–21,700).

## File format

One HDF5 file per episode, uncompressed, `particles/pos` chunked along time.

```
NNNN.hdf5
├── attrs     schema_version, episode_id, terrain, terrain_variant, category, motion, shape, split,
│             motion_type, settle_s, seed, particle_seed, dt, duration_s, num_steps, num_particles, num_bodies,
│             solver ("mpm"), code_commit, params (JSON: every sampled parameter), material (JSON),
│             particle_spacing_m, voxel_size_m, particle_mass_kg, bed_size_m, object_mass_kg, object_size_m,
│             size_level, hover_until_s, convention notes, complete
├── time                 [T]          s, T = num_steps + 1
├── particles/
│   └── pos              [T, N, 3]    float32, world frame [m]
└── body/                (absent for freefall and rest_bed)
    ├── pose             [T, 7]       xyz, quaternion xyzw
    ├── vel              [T, 6]       linear xyz, angular xyz, world frame
    ├── ext_wrench       [T-1, 6]     applied force [N] and torque [N m] about the centre of mass
    └── contact_wrench   [T-1, 6]     sand-on-body wrench from the MPM impulses (impulse / dt)
```

- Particle velocity is not stored. Use finite differences of `particles/pos`.
- `ext_wrench` and `contact_wrench` row `k` acts between state `k` and state `k+1`.
- `ext_wrench` is zero for the prescribed categories (`lateral_push`, `vertical_push`) and for `rest`. For
  `impact` it holds the anti-gravity hover wrench until `hover_until_s` (1 s), after which it is zero.
- `contact_wrench` excludes forces from the tray.
- Frame: world axes, origin at the tray centre on the nominal surface (`terrain/<terrain>.json`).

### Reading an episode

```python
from particle_nerd_benchmark.dataset.hdf5_io import read_episode

episode = read_episode("train/flat/impact/0000.hdf5")
positions = episode["pos"]  # [T, N, 3]
pose, wrench = episode["body"]["pose"], episode["body"]["contact_wrench"]
print(episode["attrs"]["motion"], episode["attrs"]["shape"])
```

## Size

Measured in a test run on the three terrains: 6 s object files are 300–320 MB, 10 s files (`rest`, `rest_bed`) 500–525 MB,
`freefall` (bed plus block) 330–360 MB, Franka 250–265 MB and G1 about 0.74 GB. The full dataset is about 450 GB,
uncompressed.

## Collecting

The dataset is collected on OSMO. `dataset/osmo/make_workflow.py` writes the workflow (one task per terrain and task
group), and `dataset/run_jobs.py` runs one collector process per GPU inside each task. Collectors skip finished files,
so a resubmission resumes.

```bash
PYTHONPATH=examples python examples/particle_nerd_benchmark/dataset/osmo/make_workflow.py --out dataset.yaml \
    --select lateral vertical_impact rest_sand franka g1 --ckpt-url <swift folder with the G1 checkpoint>
examples/particle_nerd_benchmark/learning/osmo/run_collection.sh --workflow dataset.yaml <swift dataset root> --pool <pool>

# One terrain, one episode per motion (a quick check)
PYTHONPATH=examples python -m particle_nerd_benchmark.dataset.run_jobs --terrain flat --per-motion 1 --output <dir>

# Plan of the whole dataset
PYTHONPATH=examples python -m particle_nerd_benchmark.dataset.manifest --out <dir>
```

Kinematic objects (`lateral_push`, `vertical_push`) are driven with `skip_forward=True` root-state writes
(`dataset/actions.py`). The default writes invalidate cached data every step and make the MPM bed slowly sink.

## Previews

Animated previews from the first prototype (flat bed, objects at scale 1) show each motion. Each links to the real-time MP4.

### robots
| franka (shovel, arc sweeps) | g1 (walking) |
|---|---|
| ![](docs/gifs/robot__franka.gif) [video](docs/videos/robot__franka.mp4) | ![](docs/gifs/robot__g1.gif) [video](docs/videos/robot__g1.mp4) |

### lateral_push
| push_x | push_y |
|---|---|
| ![](docs/gifs/lateral_push__push_x.gif) [video](docs/videos/lateral_push__push_x.mp4) | ![](docs/gifs/lateral_push__push_y.gif) [video](docs/videos/lateral_push__push_y.mp4) |
| **circle_cw** | **circle_ccw** |
| ![](docs/gifs/lateral_push__circle_cw.gif) [video](docs/videos/lateral_push__circle_cw.mp4) | ![](docs/gifs/lateral_push__circle_ccw.gif) [video](docs/videos/lateral_push__circle_ccw.mp4) |

### vertical_push
| push_down | pull_up |
|---|---|
| ![](docs/gifs/vertical_push__push_down.gif) [video](docs/videos/vertical_push__push_down.mp4) | ![](docs/gifs/vertical_push__pull_up.gif) [video](docs/videos/vertical_push__pull_up.mp4) |

### impact
| drop | throw |
|---|---|
| ![](docs/gifs/impact__drop.gif) [video](docs/videos/impact__drop.mp4) | ![](docs/gifs/impact__throw.gif) [video](docs/videos/impact__throw.mp4) |

### rest and rest_bed
| rest (object on sand) | rest_bed (sand only) |
|---|---|
| ![](docs/gifs/rest__rest.gif) [video](docs/videos/rest__rest.mp4) | ![](docs/gifs/rest_bed__rest.gif) [video](docs/videos/rest_bed__rest.mp4) |

### freefall
| drop | toss |
|---|---|
| ![](docs/gifs/freefall__drop.gif) [video](docs/videos/freefall__drop.mp4) | ![](docs/gifs/freefall__toss.gif) [video](docs/videos/freefall__toss.mp4) |
