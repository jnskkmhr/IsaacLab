# Goal-conditioned perceptive Mimic

Train a single goal policy on `parc/jump_001/motion.npz` and its paired terrain.
The destination is the reference's final root position and heading, not its final joint pose.
The actor observes destination XYZ in the robot yaw frame, sine/cosine of heading error,
proprioception, and the height scan. Reference state, phase, and episode mode are critic-only.

Training starts with imitation episodes for 120,000 control steps per environment
(5,000 PPO iterations at 24 steps/iteration), then decreases imitation sampling linearly
to 50% over another 120,000 steps. Imitation uses reference-state initialization and
tracking rewards/terminations. Task episodes start at the route beginning, randomize
initial XY/yaw and destination XY, and use progress, position, heading, and settling rewards.
Task episodes continue past the reference clip, with a 10-second timeout and physical
failure conditions. Both modes share joint/action regularization; no assistive wrench is used.

`Metrics/motion/task_success_rate` measures task episodes that reached the position/yaw
tolerances with a foot in contact at least once. It excludes imitation episodes; clip completion
is not goal success. No hold duration is required. Check `task_episode_fraction` alongside it:
there are no task episodes during warmup. Play always uses task mode.

```bash
uv run --with wandb isaaclab train --rl_library rsl_rl \
  --task IsaacContrib-Mimic-G1-29dof-Perception-Goal-Condition \
  --num_envs 4096 --logger wandb --viz newton_gl --video

uv run --with wandb isaaclab play --rl_library rsl_rl \
  --task IsaacContrib-Mimic-G1-29dof-Perception-Goal-Condition-Play \
  --num_envs 1 --viz newton_gl --wandb_run RUN_ID \
  --wandb_project g1_mimic_perception_goal_condition --video

bash osmo/run_multinode.sh osmo/osmo_multi_gpu.yaml \
  --pool isaac-lab-l40s-03 \
  --set num_nodes=8 num_gpu=1 \
  --set-string \
    workflow_name=g1-mimic-goal-jump001 \
    platform=ovx-l40s \
    output_url=swift://pdx.s8k.io/AUTH_team-isaac-sqa/isaac-sqa/jkamohara/g1-perception/runs \
    wandb_credential=wb-auth wandb_key=wb_api_key \
    args="--task IsaacContrib-Mimic-G1-29dof-Perception-Goal-Condition --num_envs 512 --logger wandb --viz newton_gl --video"
```

This is a first goal-conditioned baseline inspired by GfR, not a reproduction of its
full method. The initial experiment uses one fixed gap geometry; destination noise is
bounded to the landing platform. Generalization to different gaps is not measured here.
