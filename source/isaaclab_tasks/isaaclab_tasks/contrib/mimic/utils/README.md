# G1 LAFAN motion conversion

This utility converts retargeted G1 robot CSVs into named motion NPZ files.
Use the existing NPZ player for reference playback and recording. These tools do
not retarget human motion or train a policy. Run them from the repository root in its configured Newton
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
  source/isaaclab_tasks/isaaclab_tasks/contrib/mimic/data/motions/npz/play_motion_file.py \
  --motion_file "$output/motion_60fps.npz" --video --headless \
  --video_folder "$output/reference_video"
```

The existing player records one full pass at the NPZ frame rate (60 fps here)
to `reference_video/motion_60fps.mp4`. Choose a new video folder to avoid
overwriting an earlier recording. It already supports a following camera,
offscreen Newton GL recording, joint-name mapping, and quaternion-order metadata.
Linux SSH sessions need a working EGL setup and the configured environment's
imageio/FFmpeg video support. This is prescribed reference playback, not learned
policy behavior.

The player uses `UNITREE_G1_29DOF_CFG`; the converter resolves the robot from the
training task. Visual playback does not establish FK consistency with that task's
robot asset. Unlike the removed PR helper, the existing player does not produce
an FK-error JSON report. Conversion checks remain in `conversion.json`.

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
| `record_lafan_reference.py` | Existing `data/motions/npz/play_motion_file.py --video --headless` |
| `train_lafan_walk_3000.py` | `isaaclab train` with explicit arguments above |
| `check_lafan_training.py` | Inspect the selected run's logs/checkpoints and use `isaaclab play` |

The conversion CLI replaces `--output-root` with `--output_dir` (the exact new
destination, without an automatically generated timestamp), and `--frame-range`
with `--frame_range`. The converter derives the source checkout from its own
package location; `--repo` and the old commit allowlist are removed. Keep old
standalone helpers with their original experiments if those runs still need them.

The intermediate PR's `utils/record_reference_motion.py` was removed in favor of
the existing NPZ player. Use `--motion_file` instead of `--motion`, and
`--video_folder` instead of `--output`; the player derives the MP4 name from the NPZ.

The previous converter was exercised on the lab's RTX PRO 2000. Its numerical
implementation is unchanged here. GPU conversion and the documented existing
player command must still be checked together on the target checkout.
