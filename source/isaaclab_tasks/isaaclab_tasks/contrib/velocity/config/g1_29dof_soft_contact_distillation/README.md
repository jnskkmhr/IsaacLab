# G1 VAE teacher and TCN student

Separate flat rigid/RFT tasks using compositional PPO. PhysX is the default; append
`-Newton` to either task ID to use Newton/MJWarp with the same task terms.
These tasks use analytical RFT foot forces, **not MPM**.

Reference configurations: teacher run `0jxks0mm` and student run `2uiyzumn` in
`g1_29dof_soft_vae_teacher` and `g1_29dof_soft_vae_student_tcn` respectively.
The older source is [IsaacLab ral26_gm_loco](https://github.com/jnskkmhr/IsaacLab/tree/ral26_gm_loco/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/g1_29dof_soft)
and [rsl_rl feature/encoder](https://github.com/jnskkmhr/rsl_rl/tree/feature/encoder).

## Network and observation contracts

| Component | Inputs | Output |
|---|---|---|
| Policy observation | Angular velocity (3), projected gravity (3), velocity/heading command (7), joint positions (29), joint velocities (29), previous actions (29) | 100 values |
| Teacher history | Angular velocity, projected gravity, joint positions and velocities; 10 frames, flattened per term | 640 values |
| Dynamics observation | Base linear velocity (3), foot heights (2), contacts (2), forces (6), air times (2) | 15 values |
| Terrain observation | Normalized RFT stiffness, with rigid ground mapped to 1 | 1 value |
| VAE encoder | Teacher history + dynamics + terrain (656 values) | Mean (64), log variance (64) |
| Teacher actor | Policy observation + mean (164 values) | 29 Gaussian action means and learned standard deviation |
| VAE decoder | Sampled 64-dimensional latent during training | Reconstructed terrain stiffness (1) |
| TCN encoder | 30 oldest-to-newest frames, each containing angular velocity, projected gravity, joint positions and velocities | 64 values |
| Student actor | Policy observation + TCN output (164 values) | 29 Gaussian action means and learned standard deviation |
| Critic | Critic observation (101, quaternion replaces projected gravity) + dynamics (15) + terrain (1) | Scalar value |

The TCN uses two causal residual Conv1d blocks (256, 128 channels), kernel size 3,
dilation 1, residual LayerNorm and ELU. It projects every frame to 64 channels,
then flattens all 30 frames into a linear 64-value readout with ELU, matching the
reference architecture. History groups are clean as in the saved runs; the policy
observation has the reference uniform observation noise.

| Loss | Weight / schedule | Trained terms |
|---|---|---|
| Teacher PPO | 1 | Actor, VAE encoder, critic |
| Teacher reconstruction MSE | 1 | VAE encoder and decoder |
| Teacher KL to unit Gaussian, averaged over latent dimensions | 0.01 | VAE encoder |
| Student action-mean MSE | 1 → 0.1 over 270,000 environment steps (11,250 iterations) | Student actor and TCN |
| Student latent MSE against teacher mean | 1 | TCN |
| Student PPO | 0 → 0.9 over the same interval | Student actor, TCN, critic |

The teacher encoder and actor are frozen during student training. The student
checkpoint contains their weights, so resuming and deployment need no separate
teacher file. The exported student only consumes `policy` and `student_history`;
privileged observations are used only for training.

## Train

Run from `/home/jkamohara/isaac/experiment/ral-gm-loco-lab-3.0`.
Initialize this checkout's Isaac Sim USD libraries before using PhysX:

```bash
uv sync --locked --extra isaacsim --extra rsl-rl --extra wandb --extra video
export OMNI_KIT_ACCEPT_EULA=YES
kit_usd_root=$(uv run --no-sync python -c 'import pathlib,sys; print(next((pathlib.Path(sys.prefix)/"lib/python3.12/site-packages/isaacsim/extscache").glob("omni.usd.libs-*")))')
python_lib_dir=$(uv run --no-sync python -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))')
export PYTHONPATH="$kit_usd_root${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$kit_usd_root/bin:$python_lib_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

On OSMO, the existing template's `kit_usd=true` option performs this setup.
The library is pinned to a remote Git commit, so no local editable checkout is needed.

```bash
uv run --extra isaacsim --extra rsl-rl --with wandb isaaclab train \
  --rl_library rsl_rl \
  --task IsaacContrib-Velocity-Flat-G1-29dof-SoftContact-Teacher \
  --num_envs 4096 --logger wandb --viz kit --video
```

Train the student from a **teacher checkpoint created by this task**:

```bash
teacher_checkpoint=/absolute/path/to/teacher/model_24000.pt
uv run --extra isaacsim --extra rsl-rl --with wandb isaaclab train \
  --rl_library rsl_rl \
  --task IsaacContrib-Velocity-Flat-G1-29dof-SoftContact-Student \
  --num_envs 4096 --logger wandb --viz kit --video \
  "agent.networks.teacher_encoder.pretrained_cfg.path=$teacher_checkpoint" \
  "agent.networks.teacher_policy.pretrained_cfg.path=$teacher_checkpoint"
```

For Newton, append `-Newton` to the task and use `--viz newton_gl`. The Newton
environment records that visualizer; PhysX records Kit. W&B projects default to
`g1_29dof_soft_vae_teacher` and `g1_29dof_soft_vae_student_tcn`.

When playing or resuming a **student** checkpoint, disable initial teacher-file loading:

```bash
uv run --extra isaacsim --extra rsl-rl --with wandb isaaclab play \
  --rl_library rsl_rl \
  --task IsaacContrib-Velocity-Flat-G1-29dof-SoftContact-Student \
  --checkpoint /absolute/path/to/student/model_24999.pt \
  --num_envs 1 --viz kit \
  agent.networks.teacher_encoder.pretrained_cfg=null \
  agent.networks.teacher_policy.pretrained_cfg=null \
  'env.video_recorders=[]'
```

For training resume, use the same two `null` overrides together with the normal
RSL-RL resume/checkpoint options. Do not use the null overrides to start a new
student: that would initialize a random teacher.

## Differences from the legacy implementation

- This uses the current compositional checkpoint format, not the old combined
  `MLPVAEModel` checkpoint. Old teacher checkpoints need a conversion before use.
- PPO and auxiliary losses train the encoder jointly. The old fork cached encoder
  outputs in the rollout and reused those outputs during PPO updates.
- Student training includes the requested action and feature losses; it does not
  add the old student's optional decoder reconstruction loss.
- The physics library versions and Lab 3.0 APIs differ. Newton uses the same
  analytical RFT action term but MJWarp for rigid-body dynamics; numerical equality
  between PhysX and Newton is not claimed.

Existing flat/rough PPO tasks are unchanged.

## Verification

- 17 focused compositional-library tests passed, including VAE reconstruction/KL,
  real PPO updates with both student losses, frozen teachers, checkpoint reload,
  and TorchScript teacher/student inference.
- Teacher and student each completed two PPO updates on PhysX and Newton with
  eight environments: 384 transitions per run. All reported losses were finite.
- PhysX teacher training recorded a 20-frame Kit video. Student checkpoint playback
  on Newton recorded 20 GL frames after loading from the pinned installed package,
  with teacher-file initialization disabled.
- These are integration smoke tests, not evidence of a converged gait.
