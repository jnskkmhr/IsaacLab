# G1 29dof locomotion on soft terrain 

### Vanilla MLP policy

```bash
# training
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Velocity-Flat-G1-29dof-Soft-v1 --num_envs 4096 --headless --video

# evaluation
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py --task Isaac-Velocity-Flat-G1-29dof-Soft-Play-v1 --num_envs 1 --video
```

### Teacher-student distillation 

Teacher:
```bash
# training
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Velocity-Flat-G1-29dof-Soft-Teacher-v1 --num_envs 4096 --headless --video --agent rsl_rl_distillation_cfg_entry_point

# evaluation
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py --task Isaac-Velocity-Flat-G1-29dof-Soft-Teacher-Play-v1 --num_envs 1 --agent rsl_rl_distillation_cfg_entry_point
```

Student:
```bash
# training
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Velocity-Flat-G1-29dof-Soft-Student-v1 --num_envs 4096 --headless --video --agent rsl_rl_distillation_cfg_entry_point --checkpoint /path/to/teacher/policy

# evaluation
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py --task Isaac-Velocity-Flat-G1-29dof-Soft-Student-Play-v1 --num_envs 1 --agent rsl_rl_distillation_cfg_entry_point
```


### Local smoke tests

The migration from `IsaacLab_RAL` was tested with Python 3.11, Isaac Sim 5.1.0,
PyTorch 2.7.0+cu128, Warp 1.18.0, and the RSL-RL `feature/encoder` dependency
declared in `source/isaaclab_rl/setup.py`. The configuration tests also require
the `holosoma-agent` extra declared in that file.

From the repository root, activate the compatible environment and run:

```bash
source .venv/bin/activate
./isaaclab.sh -p -m pytest source/isaaclab_tasks/test/test_g1_29dof_soft.py -q -s

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Velocity-Flat-G1-29dof-Soft-v1 --num_envs 16 \
  --max_iterations 2 --logger tensorboard --run_name migration_smoke --headless
```

The 17 regression tests cover all eight registered environment variants and their
agent configurations. Each variant runs four environments for 100 control steps
with an explicit reset halfway through. Tests require finite observations, rewards,
robot root states, and foot wrenches, plus nonzero soft-contact forces on the robot's
feet. The short PPO run checks training and checkpoint creation; it does not establish
learned locomotion quality or long-run training stability.

### W&B playback and GUI compatibility

Use `play_wandb.py` for `--wandb_run`; `play.py` does not accept that option.
For playback without rendering:

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_wandb.py \
  --task Isaac-Velocity-Flat-G1-29dof-Soft-Play-v1 \
  --num_envs 1 --wandb_run 4blqnxbi --headless
```

The checkpoint `model_19999.pt` from this run passed a 200-step headless rollout
with finite actions, observations, and rewards. GUI startup on the tested RTX 5090
with driver 595.99.02 crashes in `librtx.scenedb.plugin.so` before task loading.
A renderer-only probe reproduces the crash. This matches the
[Isaac Sim 5.1 / 595-series driver issue documented by NVIDIA](https://forums.developer.nvidia.com/t/isaac-sim-5-1-0-crashes-on-startup-with-rtx-4090-on-ubuntu-24-04-4-segfaulting-in-librtx-scenedb-plugin-so-after-iommu-was-disabled/371957/3).
GUI playback requires a compatible driver/runtime combination; headless simulation
tests do not validate GUI rendering. Adding `--video` or `--enable_cameras` invokes
rendering even with `--headless`, so those options are not covered by this workaround.
