# Newton flat RFT matching PhysX run hiwy2c7j

Run from this checkout (`ral_gm_loco_lab_3.0`). The source snapshot is transferred as one gzip archive and extracted before
installation, avoiding incomplete multi-file directory downloads. It includes the standalone
`g1_29dof_soft_contact_newton` task and original robot asset. This starts from
scratch with 4 GPUs × 1024 environments, 20,000 PPO iterations, seed 42, W&B and
video. It retains the reference rewards, observations, events and curricula.
Training terrain mixes rigid columns and RFT columns with a lowered rigid floor.

```bash
bash osmo/run_multinode.sh osmo/osmo_multi_gpu.yaml \
  --pool isaac-lab-l40s-03 \
  --set num_nodes=1 num_gpu=4 num_cpu=32 memory=128 storage=128 \
  --set-string \
    workflow_name=isaac-lab-g1-newton-soft-flat \
    platform=ovx-l40s \
    output_url=swift://pdx.s8k.io/AUTH_team-isaac-sqa/isaac-sqa/jkamohara/contact-benchmark/runs \
    wandb_credential=wb-auth wandb_key=wb_api_key \
    args="--task IsaacContrib-Velocity-Flat-G1-29dof-SoftContact-Newton --num_envs 1024 --seed 42 --logger wandb --log_project_name g1_29dof_contact_benchmark --run_name newton_soft_flat_hiwy2c7j --viz newton_gl --video"
```

Append `--dry-run` to validate/render the job without uploading or submitting.
The GL/RTX camera uses the recorded PhysX run's eye `(4,-4,3)`, lookat `(0,0,0)`,
12 mm focal length, 1280×720 frame, and robot-root tracking. No environment
visibility restriction is added. Renderer lighting/textures still differ from Kit.

## Current-command and symmetry ablation

Keep the original Newton run and launch this additional task from scratch. The
L40S pool was full at submission, so this run uses four L40 GPUs in `isaac-lab-l40-06`.
The seed and population remain 42 and 4 × 1024 environments.

```bash
bash osmo/run_multinode.sh osmo/osmo_multi_gpu.yaml \
  --pool isaac-lab-l40-06 \
  --set num_nodes=1 num_gpu=4 num_cpu=32 memory=128 storage=128 \
  --set-string \
    workflow_name=isaac-lab-g1-newton-soft-symmetry \
    platform=ovx-l40 \
    output_url=swift://pdx.s8k.io/AUTH_team-isaac-sqa/isaac-sqa/jkamohara/contact-benchmark/runs \
    wandb_credential=wb-auth wandb_key=wb_api_key \
    args="--task IsaacContrib-Velocity-Flat-G1-29dof-SoftContact-Newton-Symmetry --num_envs 1024 --seed 42 --logger wandb --log_project_name g1_29dof_contact_benchmark --run_name newton_soft_flat_current_command_symmetry --viz newton_gl --video"
```

## PhysX TCN student: teacher initialization and decoder matching

Run from `/home/jkamohara/isaac/experiment/ral-gm-loco-lab-3.0`.
This uses completed teacher `u6hqne30/model_27999.pt`, initializes the trainable
student actor/distribution and critic from that teacher, and keeps the teacher
encoder, actor, and decoder frozen. Decoder-output matching joins action and
latent matching, with the legacy PPO/imitation schedule. The TCN starts fresh.

```bash
teacher_checkpoint=/tmp/isaaclab/logs/u6hqne30/model_27999.pt
bash osmo/run_multinode.sh osmo/osmo_multi_gpu.yaml \
  --pool isaac-lab-l40s-03 \
  --set num_nodes=1 num_gpu=4 num_cpu=32 memory=128 storage=160 kit_usd=true \
  --set-string \
    workflow_name=isaac-lab-g1-physx-tcn-student \
    platform=ovx-l40s \
    output_url=swift://pdx.s8k.io/AUTH_team-isaac-sqa/isaac-sqa/jkamohara/contact-benchmark/runs \
    teacher_checkpoints_url=swift://pdx.s8k.io/AUTH_team-isaac-sqa/isaac-sqa/jkamohara/contact-benchmark/teachers/u6hqne30-model27999/teacher_bundle/ \
    wandb_credential=wb-auth wandb_key=wb_api_key \
    uv_args="--extra isaacsim --extra rsl-rl --extra wandb --extra video" \
    args="--task IsaacContrib-Velocity-Flat-G1-29dof-SoftContact-Student \
--num_envs 1024 --seed 42 --logger wandb \
--log_project_name g1_29dof_soft_vae_student_tcn \
--run_name physx_tcn_student_u6hqne30_it27999_teacher_init_decoder --viz kit --video \
agent.networks.actor.pretrained_cfg.path=$teacher_checkpoint \
agent.networks.critic.pretrained_cfg.path=$teacher_checkpoint \
agent.networks.teacher_encoder.pretrained_cfg.path=$teacher_checkpoint \
agent.networks.teacher_policy.pretrained_cfg.path=$teacher_checkpoint \
agent.networks.student_decoder.pretrained_cfg.path=$teacher_checkpoint \
agent.networks.teacher_decoder.pretrained_cfg.path=$teacher_checkpoint"
```

Validation: two actual PhysX PPO updates with finite action/feature/decoder losses
and a recorded Kit video; exact actor/distribution/critic initialization checked;
frozen teacher and decoder weights remained bitwise unchanged after updates.


## Rough PPO baselines — submitted November 12, 2026

Run from `/home/jkamohara/isaac/experiment/ral-gm-loco-lab-3.0` after reviewing
`g1_29dof_rigid_newton/README.md`. Each job uses four GPUs and 1024 environments
per GPU (4096 total), with W&B and video. Submitted on `isaac-lab-l40s-03` after review: `isaac-lab-g1-rigid-rough-6` and `isaac-lab-g1-rft-rough-1`. Both were scheduled and downloading inputs at the initial status check.

```bash
cd /home/jkamohara/isaac/experiment/ral-gm-loco-lab-3.0
export osmo_node=isaac-lab-l40s-03
export platform=ovx-l40s

bash osmo/run_multinode.sh osmo/osmo_multi_gpu.yaml \
  --pool "$osmo_node" \
  --set num_nodes=1 num_gpu=4 num_cpu=32 memory=128 storage=128 \
  --set-string \
    platform="$platform" \
    workflow_name=isaac-lab-g1-rigid-rough \
    output_url=swift://pdx.s8k.io/AUTH_team-isaac-sqa/isaac-sqa/jkamohara/contact-benchmark/runs \
    wandb_credential=wb-auth wandb_key=wb_api_key \
    args="--task IsaacContrib-Velocity-Rough-G1-29dof-Rigid-Newton --rl_library rsl_rl --num_envs 1024 --seed 42 --logger wandb --log_project_name g1_29dof_contact_benchmark --run_name rigid_rough --viz newton_gl --video"

bash osmo/run_multinode.sh osmo/osmo_multi_gpu.yaml \
  --pool "$osmo_node" \
  --set num_nodes=1 num_gpu=4 num_cpu=32 memory=128 storage=128 \
  --set-string \
    platform="$platform" \
    workflow_name=isaac-lab-g1-rft-rough \
    output_url=swift://pdx.s8k.io/AUTH_team-isaac-sqa/isaac-sqa/jkamohara/contact-benchmark/runs \
    wandb_credential=wb-auth wandb_key=wb_api_key \
    args="--task IsaacContrib-Velocity-Rough-G1-29dof-SoftContact-Newton --rl_library rsl_rl --num_envs 1024 --seed 42 --logger wandb --log_project_name g1_29dof_contact_benchmark --run_name rft_rough --viz newton_gl --video"
```
