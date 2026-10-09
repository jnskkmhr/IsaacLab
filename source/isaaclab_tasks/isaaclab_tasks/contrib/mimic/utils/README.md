# G1 LAFAN motion conversion and reference recording

These utilities convert retargeted G1 robot CSVs into named motion NPZ files and
record prescribed kinematic motion in Newton. They do not retarget human motion
or train a policy. Run them from the repository root in its configured Newton
environment; do not copy them into a separate home directory.

The input format is the 30 Hz, 36-column G1 CSV from the
[LAFAN1 retargeting dataset](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset):
root position xyz [m], root orientation xyzw, then 29 joint positions [rad].
The existing CSV converter supplies the loader and G1 joint order. Its loader
class is extracted without executing its Isaac Sim `AppLauncher`; the original
script is not yet safe to import. Input data and checkpoints are not included.
Follow the dataset's license and attribution requirements.

## Convert a reference

Set the input path and a **new** output directory. Paths can be anywhere on the
machine; no user account, checkout name, or Git commit is required.

```bash
csv="$HOME/datasets/LAFAN1_Retargeting_Dataset/g1/walk1_subject1.csv"
output="$PWD/outputs/lafan_walk"

uv run --frozen --no-sync python \
  source/isaaclab_tasks/isaaclab_tasks/contrib/mimic/utils/convert_lafan_csv_to_npz.py \
  --csv "$csv" --output_dir "$output" --frame_range 121 301
```

`--frame_range` is 1-based and inclusive. This example selects source time
4–10 s and produces 360 frames at 60 Hz; omit the option for the entire CSV.
Other G1 LAFAN clips use the same command with their own path and frame range.

Outputs are `motion_60fps.npz` and `conversion.json`. The converter:

- Resamples the original CSV loader's output from 30 Hz to 60 Hz.
- Uses the `IsaacContrib-Mimic-G1-29dof-v0` task's robot and Newton forward
  kinematics without integrating physics.
- Writes explicit joint/body names and `quaternion_order="xyzw"` metadata.
- Computes body linear and angular velocities from FK poses by finite differences.
- Checks state writes, array validity, and the actual training loader's joint
  permutation and body selection; records source/output hashes and the Git revision.

Use the same robot configuration for conversion and training. The checks do not
establish retargeting quality or prove compatibility with a different robot asset.
Review each new reference visually before starting training.

## Record the prescribed reference

```bash
env -u DISPLAY uv run --frozen --no-sync python \
  source/isaaclab_tasks/isaaclab_tasks/contrib/mimic/utils/record_reference_motion.py \
  --motion "$output/motion_60fps.npz" --output "$output/reference.mp4"
```

This records a 30 fps MP4 from the 60 Hz reference and writes a JSON report
beside it. It needs `ffmpeg` on PATH or the installed `imageio-ffmpeg` package.
Newton GL uses offscreen rendering; Linux SSH sessions need a working EGL setup.
The camera follows the reference root. Root and joint poses are prescribed on
every frame, body poses are checked against the NPZ, and no physics steps or
policy inference are performed. This is a reference video, not learned behavior.
An existing output video is never overwritten.

## Train with the existing CLI

Training uses the repository's existing task and agent entry point. Choose the
environment count and iteration budget for the available GPU. For example, the
earlier walking experiment used:

```bash
uv run --frozen --no-sync isaaclab train \
  --rl_library rsl_rl --task IsaacContrib-Mimic-G1-29dof-v0 \
  --num_envs 2048 --max_iterations 3000 --seed 42 \
  --run_name lafan_walk --logger tensorboard --viz none \
  physics=newton_mjwarp \
  "env.commands.motion.motion_file=\"$output/motion_60fps.npz\"" \
  'env.commands.motion.stance_phase_ranges=[]' \
  'env.video_recorders=[]' agent.experiment_name=lafan_walk
```

The empty stance ranges are appropriate for this moving clip: they disable
standing-specific reward gating. Choose stance ranges separately for motions
with intended standing segments. Rewards, symmetry settings and assistance
curriculum otherwise come from the selected checkout; the utilities do not
change them. Training results can differ between repository revisions.

For policy playback, use `isaaclab play` with an explicit checkpoint and the same
motion file. To disable assistance, override `env.events.assistive_wrench=null`.
The standard PLAY task relaxes some training termination conditions, so a PLAY
video alone is not a success-rate evaluation under training conditions.

## Migration from the initial PR layout

| Previous helper under `tools/lafan_reproduction` | Replacement |
| --- | --- |
| `convert_lafan_newton.py` | `utils/convert_lafan_csv_to_npz.py` |
| `record_lafan_reference.py` | `utils/record_reference_motion.py` |
| `train_lafan_walk_3000.py` | `isaaclab train` with explicit arguments above |
| `check_lafan_training.py` | Inspect the selected run's logs/checkpoints and use `isaaclab play` |

The conversion CLI replaces `--output-root` with `--output_dir` (the exact new
destination, without an automatically generated timestamp), and `--frame-range`
with `--frame_range`. Both utilities derive the source checkout from their own
package location; `--repo` and the old commit allowlist are removed. Keep old
standalone helpers with their original experiments if those runs still need them.

The previous conversion and reference recording were exercised on the lab's
RTX PRO 2000. This reorganization preserves their numerical implementation;
GPU conversion and rendering must still be rechecked on the target checkout.
