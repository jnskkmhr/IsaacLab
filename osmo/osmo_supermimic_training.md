# SuperMimic G1 experiments

Both tasks use PR #255's differentiable velocity rewards, adapted to our 29-joint
convex-foot G1 with delayed BeyondMimic implicit actuators. They control all 29
joints with the PR's default-position offset and 0.5 action multiplier. Rigid foot
contact margin is 0.016 m; MPM foot margin remains half the particle voxel size.
The mixed task retains the existing moving-patch coupling and terrain allocation.
Its state, contact rewards and termination labels use combined rigid/MPM forces.
The PD torque and policy-step acceleration terms are approximations, as in the PR.
Both tasks additionally penalize simultaneous foot flight (`no_jump`, weight -2.0).
Physical rewards use current contact clocks; imagined rewards use the soft product
of the two no-contact probabilities.

Training stops by iteration count. W&B and 200-frame videos are enabled below.
The learner starts policy updates after 500 world-model learner iterations; early
collection/model warmup is not evidence of policy convergence.

## Rigid, one GPU

```bash
bash osmo/run_multinode.sh osmo/osmo_multi_gpu.yaml \
  --pool isaac-dev-rtx6000-01 \
  --set num_nodes=1 num_gpu=1 num_cpu=12 memory=96 \
  --set-string workflow_name=isaac-supermimic-rigid platform=x86-6000 \
    output_url=swift://pdx.s8k.io/AUTH_team-isaac-sqa/isaac-sqa/jkamohara/supermimic/runs \
    wandb_credential=wb-auth wandb_key=wb_api_key \
    args="--task IsaacContrib-Velocity-G1-29dof-SuperMimic-Rigid --agent rsl_rl_supermimic_cfg_entry_point --num_envs 4096 --logger wandb --viz newton_gl --video --video_length 200"
```

## Mixed MPM, 40 GPUs

```bash
bash osmo/run_multinode.sh osmo/osmo_multi_gpu.yaml \
  --pool isaac-lab-l40s-03 \
  --set num_nodes=10 num_gpu=4 num_cpu=32 memory=232 storage=128 \
  --set-string workflow_name=isaac-supermimic-mixed platform=ovx-l40s \
    output_url=swift://pdx.s8k.io/AUTH_team-isaac-sqa/isaac-sqa/jkamohara/supermimic/runs \
    wandb_credential=wb-auth wandb_key=wb_api_key \
    args="--task IsaacContrib-Velocity-G1-29dof-SuperMimic-MixedTerrain --agent rsl_rl_supermimic_cfg_entry_point --num_envs 64 --logger wandb --viz newton_gl --video --video_length 200 agent.algorithm.world_model_batch_size=2560 agent.algorithm.policy_batch_size=5120 agent.algorithm.replay_buffer_size=102400"
```

`--num_envs` is per GPU. The two learner batch sizes are global, split evenly
across ranks (64 dynamics windows and 128 policy starts per rank here). These
values deliberately differ from the PR's 2048/4096 to divide by 40. Replay
capacity is per rank: 102,400 here preserves 4,096,000 transitions across 40 ranks. The standard `train_multigpu` entrypoint synchronizes all
three optimizers and observation statistics; histories and replay remain local.

For a bounded two-node smoke run, use `num_nodes=2 num_gpu=1`, `--num_envs 8`,
`--max_iterations 200`, and the explicit smoke overrides
`agent.algorithm.world_model_batch_size=64 agent.algorithm.policy_batch_size=64
agent.algorithm.world_model_warmup_iterations=2 agent.algorithm.replay_buffer_size=4096`.
Remove those smoke overrides for learning experiments.

## Agile rewards, observations and actuation, including waist yaw

This task retains all 26 Agile reward terms, its five-frame actor observations,
critic policy history and height scan, randomized arm targets, and delayed
BeyondMimic implicit actuators. The policy controls all 15 leg/waist joints.
The experiment explicitly uses convex feet with 0.016 m robot/terrain margins.
Agile's `jumping` penalty already has weight -20; the PR task's extra `no_jump`
term is not added to this recipe.

The task-owned world model additionally predicts physical torques, foot poses,
terrain scans and substep vertical forces so the original reward formulas can be
used during imagination. Threshold contacts and failures use an event predictor.
Physical reward parity does not imply accurate learned dynamics; inspect both
model error and actual rollout videos. Scalar rewards themselves are not model
outputs. Reward curricula update the same term configurations used by the learner.

```bash
bash osmo/run_multinode.sh osmo/osmo_multi_gpu.yaml \
  --pool isaac-dev-rtx6000-01 \
  --set num_nodes=1 num_gpu=1 num_cpu=12 memory=96 storage=128 \
  --set-string workflow_name=isaac-supermimic-agile platform=x86-6000 \
    output_url=swift://pdx.s8k.io/AUTH_team-isaac-sqa/isaac-sqa/jkamohara/supermimic/runs \
    wandb_credential=wb-auth wandb_key=wb_api_key \
    args="--task IsaacContrib-Velocity-Rough-G1-29dof-Rigid-Agile-SuperMimic --agent rsl_rl_supermimic_cfg_entry_point --num_envs 4096 --logger wandb --viz newton_gl --video --video_length 200"
```
