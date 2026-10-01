# LAFAN1 G1 walking reproduction on Newton

These scripts reproduce the 2048-environment, 3000-update walking experiment.
They are version-specific helpers, not changes to the mimic rewards or policy.

## Code and environment

Runtime repository: https://github.com/jnskkmhr/IsaacLab

Runtime commit: `e1f1d6d2a805d88f37b38ea418462ab5a2766808` (junnosuke/mimic).

Keep the runtime checkout at `~/IsaacLab` at that commit, with its own working
`.venv`. The helpers deliberately check the runtime commit and editable imports.
**Do not run training from this sharing branch:** it has a different commit.
Copy this directory to `~/lafan_tools`, keeping the four Python files together,
and execute them from the pinned runtime checkout as shown below. On an existing
machine, inspect local changes and ongoing jobs before changing any checkout.

The successful laptop run used Linux, Python 3.12.14, Torch 2.11.0+cu128 and an
RTX PRO 2000 with 8 GB VRAM. This is not a portable environment installer; first
set up and verify the pinned repository's Newton, mimic and rsl-rl dependencies.
Commands use `--frozen --no-sync` to preserve that environment. Kinematic video
recording additionally requires an `ffmpeg` executable; GL recording uses EGL on
Linux without DISPLAY. Import success alone does not validate GPU simulation.

## Data and conversion

Download `g1/walk1_subject1.csv` from
https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset
to `~/datasets/LAFAN1_Retargeting_Dataset/g1/walk1_subject1.csv`.
Use the dataset's own license and attribution terms. Data and weights are not
included here.

CSV layout: root xyz, root quaternion xyzw, 29 robot joints; 30 Hz.
Conversion reuses the pinned repository converter's MotionLoader and joint-name
list. It uses Newton forward kinematics with no physics steps, resamples to
60 Hz, and derives body velocities from FK poses by finite differences. It checks
arrays, joint/body mappings and NPZ-loader consistency and records file hashes.
This is not an independent validation of retargeting quality.

For a standalone conversion:

```bash
cd ~/IsaacLab
uv run --frozen --no-sync python ~/lafan_tools/convert_lafan_newton.py \
  --csv ~/datasets/LAFAN1_Retargeting_Dataset/g1/walk1_subject1.csv \
  --frame-range 121 301
```

The frame range is 1-based, inclusive: source time 4-10 s. Output is 360 frames
at 60 Hz. The printed output directory contains `motion_60fps.npz` and
`conversion.json`. Visually inspect new motions before training.

```bash
env -u DISPLAY uv run --frozen --no-sync python ~/lafan_tools/record_lafan_reference.py \
  --motion /absolute/path/to/motion_60fps.npz \
  --output "$HOME/Videos/lafan-reference.mp4"
```

## Reproduce the walking training

```bash
cd ~/IsaacLab
uv run --frozen --no-sync python -u ~/lafan_tools/train_lafan_walk_3000.py
```

This reconverts source frames 121-301 and launches a new run with:

- Task `IsaacContrib-Mimic-G1-29dof-v0`, physics `newton_mjwarp`.
- 2048 environments, 3000 PPO iterations, seed 42, TensorBoard logging.
- Pinned task/agent defaults, including symmetry and training assistance.
- Motion-specific overrides: converted NPZ and `stance_phase_ranges=[]`.
- `--viz none` and `env.video_recorders=[]` during training.

The launcher checks for GPU compute jobs before starting. It is intentionally a
fixed walking reproduction, not a general multi-motion launcher. Running/jogging
and 20,000-update experiments have not been validated by this package.

Manifests: `~/lafan_walk_runs/walk4to10s_2048env_3000updates_<timestamp>/`.
Checkpoints: `~/IsaacLab/logs/rsl_rl/lafan_walk/<run>/model_2999.pt`.
`completed.json` is written after successful training and checkpoint discovery.

## Inspect the latest completed run

```bash
cd ~/IsaacLab
uv run --frozen --no-sync python ~/lafan_tools/check_lafan_training.py
uv run --frozen --no-sync python ~/lafan_tools/check_lafan_training.py --record
```

The recording helper selects the latest walking run, verifies its motion hash
and checkpoint, disables assistive wrench, and uses the standard PLAY task.
PLAY relaxes several training termination conditions, so the video is qualitative
playback, not an unassisted success-rate evaluation under training conditions.
The helper removes DISPLAY only for its playback child process. It does not use
the unsupported `--headless` flag or indexed visualizer CLI overrides.
On success, `POLICY_VIDEO_READY` identifies `~/Videos/lafan-walk-policy-3000.mp4`.

## Evidence and limitations

The original 2048-environment / 3000-update training completed on the lab laptop.
Kinematic conversion and reference playback were exercised there. The final
policy-recording CLI correction is included, but successful recording with this
packaged version has not yet been confirmed. No numerical policy success rate is
claimed. Other machines require their own runtime and rendering validation.
