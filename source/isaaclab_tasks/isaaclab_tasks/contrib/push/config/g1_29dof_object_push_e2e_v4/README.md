# G1 box pushing on rigid flat ground, single policy, v4

`IsaacContrib-Push-Box-Flat-G1-29dof-V4` is
[`g1_29dof_object_push_e2e`](../g1_29dof_object_push_e2e/README.md) (`IsaacContrib-Push-Box-Flat-G1-29dof`)
with one setting changed. `-Play` selects the playback configuration. The MDP functions are imported
from that package, and checkpoints are interchangeable between the two tasks.

## The one difference

| Setting | Earlier task | This task | Where |
| --- | --- | --- | --- |
| Free foot trailing distance of `feet_trailing` | 0.30 m | 0.25 m | `MAX_FOOT_TRAILING_DISTANCE` in `env_cfg/reward_cfg.py` |

`feet_trailing` (weight -50) penalizes the squared distance by which either foot trails behind the
pelvis beyond this limit. The earlier task learned to push with both feet well behind the pelvis
and its weight resting on the box; the narrower band asks it to stand more over its own feet.

An earlier attempt also lowered the foot and ground friction and narrowed the forward-lean band.
It collapsed to a policy that barely moved the box, so those changes are not included here.

## Run

```bash
uv run --with wandb isaaclab train --rl_library rsl_rl \
  --task IsaacContrib-Push-Box-Flat-G1-29dof-V4 \
  --num_envs 2048 --logger wandb --viz newton_gl --video
```

```bash
uv run --with wandb isaaclab play --rl_library rsl_rl \
  --task IsaacContrib-Push-Box-Flat-G1-29dof-V4-Play \
  --num_envs 1 --viz newton_rtx --video \
  --wandb_run YOUR_PUSH_RUN --wandb_project g1_29dof_object_push_e2e
```

Runs log to the same W&B project as the earlier task, `g1_29dof_object_push_e2e`, so that the
curves overlay. `Episode_Reward/feet_trailing` uses a different band here, so its value is not
comparable between the two tasks; judge the stance from the recorded video.
