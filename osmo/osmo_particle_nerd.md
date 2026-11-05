```bash
# export osmo_node="isaac-lab-l40s-03"
# export platform="ovx-l40s"
export osmo_node="isaac-lab-l40-07"
export platform="ovx-l40"
bash osmo/run_multinode.sh osmo/osmo_multi_gpu.yaml \
  --pool "$osmo_node" \
  --set num_nodes=8 num_gpu=1 \
  --set-string \
    workflow_name=g1-particle-nerd-mixed \
    platform="$platform" \
    nodes_excluded="osmogroot-dgxc-k8s-forge-az41-prd1-w-0928,osmogroot-dgxc-k8s-forge-az41-prd1-w-2250" \
    output_url=swift://pdx.s8k.io/AUTH_team-isaac-sqa/isaac-sqa/jkamohara/particle-nerd-moving-patch/runs \
    wandb_credential=wb-auth \
    wandb_key=wb_api_key \
    args="--task IsaacContrib-Velocity-G1-29dof-ParticleNeRD-MovingPatch-MixedTerrain --num_envs 512 --wandb_run tyiby5k6 --logger wandb --viz newton_gl --video --wandb_project g1_29dof_soft_flat_ppo --run_name nerd_l40s_8gpu_512"
```


```bash
export osmo_node="isaac-dev-rtx6000-01"
export platform="x86-6000"
bash osmo/run_multinode.sh osmo/osmo_multi_gpu.yaml \
  --pool "$osmo_node" \
  --set num_nodes=1 num_gpu=1 num_cpu=16 memory=256 \
  --set-string \
    workflow_name=g1-particle-nerd-mixed \
    platform="$platform" \
    output_url=swift://pdx.s8k.io/AUTH_team-isaac-sqa/isaac-sqa/jkamohara/particle-nerd-moving-patch/runs \
    wandb_credential=wb-auth \
    wandb_key=wb_api_key \
    args="--task IsaacContrib-Velocity-G1-29dof-ParticleNeRD-MovingPatch-MixedTerrain --num_envs 4096 --wandb_run tyiby5k6 --logger wandb --viz newton_gl --video --wandb_project g1_29dof_soft_flat_ppo --run_name nerd_rtx6000_1gpu_4096"
```

For the RTX PRO 6000 run, `memory=256` requests 256 GiB of **host RAM**; it does
not change GPU memory. The original 4,096-environment job
`g1-particle-nerd-mixed-2` used the default 64 GiB and exited with code 137 during
scene initialization. A bounded rerun of the same source snapshot and physics
settings completed two PPO iterations with 16 CPUs and 256 GiB host RAM. It
measured 95.54 GiB peak cgroup memory and 81.42 GiB peak sampled GPU memory
(10-second sampling), with zero OOM events. This validates initialization and a
short training run, not long-horizon stability. Keep `--num_envs 4096`; the
resource override is in the OSMO `--set` group above.

Validation workflow:
[g1-particle-nerd-rtx6000-memory-check-1](https://us-west-2-aws.osmo.nvidia.com/workflows/g1-particle-nerd-rtx6000-memory-check-1).
