# G1 motion datasets

Import **already-retargeted G1 motions**. Raw human BVH/LAFAN/Kimodo motions must
be retargeted before using this pipeline. All motion assets live in `data/`;
`DatasetManager.save()` writes canonical `.npz` files there. Original CSV, NPY,
and PKL inputs are preserved alongside the existing NPZ references, grouped by source:

```text
data/
├── cmu_bvh/
├── lafan/
├── claw/
├── omniretarget/
└── xsens/
```

Keep converted and edited variants in the same source directory as their originals.
`xsens/` contains the leap recordings, including mirrored and extended-stance variants.
`lafan/` is reserved for imported clips; none are bundled yet.

`dataset_cfg.py` defines LAFAN, CMU_BVH, and Kimodo dataset terms. Each
`DatasetTermCfg` supplies `func(path, **params)`, returning a motion dictionary.
These presets share an importer because the retargeting tool determines the
export schema, not the original human dataset. Set term parameters or provide a
new callable for a different exporter.

## Convert and replay

From the repository root, convert a labeled body-state NPZ to the task's 60 Hz:

```bash
uv run python source/isaaclab_tasks/isaaclab_tasks/contrib/mimic/data/motions/convert_motion.py \
  --dataset lafan --input_file /path/to/retargeted_motion.npz \
  --output_name lafan/walking/motion_60fps.npz --output_fps 60

uv run python source/isaaclab_tasks/isaaclab_tasks/contrib/mimic/data/motions/play_motion_file.py \
  --motion_file source/isaaclab_tasks/isaaclab_tasks/contrib/mimic/data/motions/data/lafan/walking/motion_60fps.npz \
  --headless --video
```

The task defaults to `data/lafan/{walking,running,sprinting}/motion_60fps.npz`.
Import those clips or override `LAFAN_MOTION_DIR` / `commands.motion.motion_files`.
The repository does not include those three LAFAN clips.

For untagged files, declare `--quaternion_order xyzw` or `wxyz`. Files without
frame-rate metadata also require `--input_fps`. Metadata in the file takes
precedence. `--params params.json` supplies importer arguments such as
`joint_names` (source column order), `body_names`, `field_map`,
`joint_name_map`, and explicit `default_joint_positions` for missing joints.

Supported input schemas:

- Body-state NPZ: the canonical fields below. Older files need explicit missing
  joint/body labels and quaternion order.
- CSV or numeric NPY: root XYZ, root quaternion, joint positions, optionally
  followed by joint velocities. Joint-only CSV exports cannot reconstruct the
  root trajectory and are not accepted as complete motions.
- NPZ/dictionaries: `root_pos`, `root_rot`, `dof_pos`, `fps`, and joint labels.
  `field_map` maps these keys to an exporter's keys.
- Torch exports: `global_translation`, `global_rotation`, `dof_pos`, `dof_names`,
  and `fps`. Loading uses `weights_only=True`.
- Primitive-container PKL: a dictionary with `frames` and `fps`. Executable
  pickle globals, including pickled NumPy objects, are rejected; export those as
  NPZ instead. For XYZ + three-component rotation + joints, explicitly set
  `rotation_format` to `rotvec` or `euler_xyz` in the JSON parameters. The latter
  means extrinsic XYZ Euler angles in radians; no rotation convention is guessed.

Root/joint-only inputs require `--robot_xml /path/to/g1_scene.xml`. This must
match the task's G1 model, with a floating root and scalar joints. Include any
scene geometry referenced by MJCF contact pairs (for example, `floor`). The
importer uses MuJoCo forward kinematics, linearly interpolates positions/joint
angles, interpolates rotations with SLERP, and computes world-frame link-origin
velocities. It does not launch simulation or solve retargeting IK.

## Canonical NPZ

Every new output includes `fps`, `quaternion_order="xyzw"`, `joint_names`, and
`body_names`, plus:

| Field | Shape |
| --- | --- |
| `joint_pos`, `joint_vel` | `(frames, joints)` |
| `body_pos_w`, `body_lin_vel_w`, `body_ang_vel_w` | `(frames, bodies, 3)` |
| `body_quat_w` | `(frames, bodies, 4)` |

Positions are meters, angles radians, and velocities per second. Body states
refer to link origins in world coordinates. Names must be unique; arrays must
have matching frame counts and finite values; quaternions must be unit length.
Saving refuses to overwrite an existing file unless `--overwrite` is supplied.

In `MotionCommandCfg`, `joint_names` selects tracked robot joints and their
policy order. `source_joint_names` (renamed from `motion_joint_names`) only labels
columns in older untagged files. New canonical files carry those labels, so the
fallback is unnecessary. Use `motion_files=[path]` for a single clip; the singular
`motion_file`, `motion_stance_phase_ranges`, `stance_phase_ranges`, and
`stance_blend_time` config fields were removed from the shared command. The soft G1
task owns its stance intervals and blending in its local command subclass.

## CSV export and editing utilities

CSV conversion belongs to `DatasetManager`, with joint ordering specified by name:

```python
from isaaclab_tasks.contrib.mimic.config.g1_29dof.env_cfg.commands_cfg import JOINT_NAMES
from isaaclab_tasks.contrib.mimic.data.motions.dataset_cfg import DATASET_CFGS
from isaaclab_tasks.contrib.mimic.data.motions.dataset_manager import DatasetManager, MOTION_DATA_DIR

manager = DatasetManager(DATASET_CFGS, output_dir="/tmp/motion_exports")
motion = manager.load("lafan", MOTION_DATA_DIR / "lafan/walking/motion_60fps.npz")
manager.save_csv(motion, "joint_pos.csv", joint_names=JOINT_NAMES, joint_positions_only=True)
manager.save_csv(motion, "full_motion.csv", joint_names=JOINT_NAMES, body_name="pelvis")
```

Full CSV columns are body XYZ, body quaternion (xyzw), joint positions, and joint
velocities. Select the root body for a file that can be re-imported. CSV does not
store frame rate or names; pass these explicitly when importing it again. Both
export modes refuse to overwrite existing files unless `overwrite=True`.

Motion-editing scripts live under `util/`: `mirror_motion.py`,
`edit_longer_stance.py` (Torch exports), and `extend_stance.py` (NPZ references).
Appending static frames does not enable a separate stance reward phase.
