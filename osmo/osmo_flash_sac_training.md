Run these commands from `~/isaac/IsaacLab`. Both runs use FlashSAC with
`--video --viz newton_gl`, passed explicitly through `args`.
The YAML configures the container and resources; training flags come from the CLI.
Training stops at `max_iterations` (100,000 by default), with no wall-time cutoff.
To change the limit, add `--max_iterations NUMBER` inside `args`.
Checkpoints use `save_interval=1250` iterations. Command and contact-reward
curricula are configured in the task's `env_cfg/curriculum_cfg.py`, using
environment steps 5,000–10,000 and the standard reset-time curriculum manager.

## Single GPU, 4,096 environments

```bash
bash osmo/run_multinode.sh osmo/flash_sac_g1.yaml \
  --pool isaac-lab-l40s-03 \
  --set num_gpu=1 \
  --set-string \
    platform=ovx-l40s \
    output_url=swift://pdx.s8k.io/AUTH_team-isaac-sqa/isaac-sqa/jkamohara/isaac_flash_sac_g1/runs \
    wandb_credential=wb-auth wandb_key=wb_api_key \
    args="--rl_library rsl_rl --task IsaacContrib-Velocity-Flat-G1-29dof-Rigid-Agile --agent rsl_rl_flash_sac_cfg_entry_point --num_envs 4096 --logger wandb --video --viz newton_gl --wandb_project g1_29dof_rigid_agile_flat_flash_sac"


bash osmo/run_multinode.sh osmo/flash_sac_g1.yaml \
  --pool isaac-dev-rtx6000-01 \
  --set num_gpu=1 \
  --set-string \
    platform=x86-6000 \
    output_url=swift://pdx.s8k.io/AUTH_team-isaac-sqa/isaac-sqa/jkamohara/isaac_flash_sac_g1/runs \
    wandb_credential=wb-auth wandb_key=wb_api_key \
    args="--rl_library rsl_rl --task IsaacContrib-Velocity-Rough-G1-29dof-Rigid-Agile --agent rsl_rl_flash_sac_cfg_entry_point --num_envs 2048 --logger wandb --video --viz newton_gl"


# flash SAC
bash osmo/run_multinode.sh osmo/flash_sac_g1.yaml \
  --pool isaac-dev-rtx6000-01 \
  --set num_gpu=1 \
  --set-string \
    platform=x86-6000 \
    output_url=swift://pdx.s8k.io/AUTH_team-isaac-sqa/isaac-sqa/jkamohara/isaac_flash_sac_g1/runs \
    wandb_credential=wb-auth wandb_key=wb_api_key \
    args="--rl_library rsl_rl --task IsaacContrib-Velocity-Rough-G1-29dof-Rigid --agent rsl_rl_flash_sac_cfg_entry_point --num_envs 2048 --logger wandb --video --viz newton_gl"

# fast SAC
bash osmo/run_multinode.sh osmo/flash_sac_g1.yaml \
  --pool isaac-dev-rtx6000-01 \
  --set num_gpu=1 \
  --set-string \
    platform=x86-6000 \
    output_url=swift://pdx.s8k.io/AUTH_team-isaac-sqa/isaac-sqa/jkamohara/isaac_flash_sac_g1/runs \
    wandb_credential=wb-auth wandb_key=wb_api_key \
    args="--rl_library rsl_rl --task IsaacContrib-Velocity-Rough-G1-29dof-Rigid --agent rsl_rl_fast_sac_cfg_entry_point env.actions=joint_range_action --num_envs 2048 --logger wandb --video --viz newton_gl"


bash osmo/run_multinode.sh osmo/flash_sac_g1.yaml \
  --pool isaac-dev-rtx6000-01 \
  --set num_gpu=1 \
  --set-string \
    platform=x86-6000 \
    output_url=swift://pdx.s8k.io/AUTH_team-isaac-sqa/isaac-sqa/jkamohara/isaac_flash_sac_g1/runs \
    wandb_credential=wb-auth wandb_key=wb_api_key \
    args="--rl_library rsl_rl --task IsaacContrib-Velocity-Flat-G1-29dof-Rigid-Agile-Native --agent rsl_rl_fast_sac_cfg_entry_point --num_envs 2048 --logger wandb --video --viz newton_gl"
```

## Four GPUs on one node, 1,024 environments per GPU

This trains one shared policy with four synchronized learner processes and
4,096 environments total. Each GPU has its own environments and replay buffer;
FlashSAC averages gradients across GPUs. The workflow selects
`isaaclab train_multigpu --num_gpus 4` automatically.

```bash
bash osmo/run_multinode.sh osmo/flash_sac_g1.yaml \
  --pool isaac-lab-l40s-03 \
  --set num_gpu=4 num_cpu=32 memory=128 \
  --set-string \
    platform=ovx-l40s \
    output_url=swift://pdx.s8k.io/AUTH_team-isaac-sqa/isaac-sqa/jkamohara/isaac_flash_sac_g1/runs \
    wandb_credential=wb-auth wandb_key=wb_api_key \
    args="--rl_library rsl_rl --task IsaacContrib-Velocity-Flat-G1-29dof-Rigid-Agile --agent rsl_rl_flash_sac_cfg_entry_point --num_envs 1024 --logger wandb --video --viz newton_gl --wandb_project g1_29dof_rigid_agile_flat_flash_sac"



bash osmo/run_multinode.sh osmo/flash_sac_g1.yaml \
  --pool isaac-dev-rtx6000-01 \
  --set num_gpu=4 num_cpu=32 memory=128 \
  --set-string \
    platform=x86-6000 \
    output_url=swift://pdx.s8k.io/AUTH_team-isaac-sqa/isaac-sqa/jkamohara/isaac_flash_sac_g1/runs \
    wandb_credential=wb-auth wandb_key=wb_api_key \
    args="--rl_library rsl_rl --task IsaacContrib-Velocity-Flat-G1-29dof-Rigid-Agile --agent rsl_rl_flash_sac_cfg_entry_point --num_envs 1024 --logger wandb --video --viz newton_gl --wandb_project g1_29dof_rigid_agile_flat_flash_sac"
```

`--num_envs` is per GPU. Batch size and replay capacity in the algorithm config
are also per GPU. This workflow requests one node; it does not take a
`num_nodes` argument. CPU and host memory requests cover all four processes.
