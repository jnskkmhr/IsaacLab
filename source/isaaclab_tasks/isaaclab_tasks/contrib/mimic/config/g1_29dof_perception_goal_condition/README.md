# Goal-conditioned perceptive Mimic

Train a single goal policy on the four `kimodo/supplied_stairs_pyramid_stand/motion_yaw_*.npz` clips.
The original climb is followed by a two-second settling transition and a two-second
standing hold (7.48 seconds total). The final root is 0.35 m farther into the landing
than in the original clips; task goals use this new standing endpoint.
They share a four-sided staircase with a 2.5 m square top landing. The paired
terrain uses a 2 m ground margin to support the full randomized starting footprint. Each reset samples
a 0°, 90°, 180°, or 270° approach, with the full reference and goal rotated together.
Imitation destinations use the reference endpoint. Task destinations are shifted 0.75 m
forward in the final reference heading frame, then sampled within ±0.35 m in world XY
to stay inside the landing. Goal Z retains the reference endpoint root height.
The actor observes destination XYZ in the robot yaw frame, sine/cosine of heading error,
proprioception, and the height scan. Reference state, phase, and episode mode are critic-only.
Reference joint positions, phase, and anchor pose observations are zeroed in task episodes;
the critic retains robot state, goal, and terrain observations with unchanged input dimensions.

Training starts with imitation episodes for 120,000 control steps per environment
(5,000 PPO iterations at 24 steps/iteration), then decreases imitation sampling linearly
to 50% over another 120,000 steps. After 10,000 PPO iterations, each new episode
is sampled as imitation with probability 0.5. This is an episode sampling probability,
not a loss weight or an exact fraction of rollout samples. To test 20% imitation, append
`env.commands.motion.final_imitation_probability=0.2` to the training command. Training videos show the first 50 environments. Imitation uses reference-state initialization and
tracking rewards/terminations. Task resets use a per-environment success curriculum rather than a timed expansion.
At difficulty zero, they retain the first reference posture and velocities, and the
reference endpoint is shifted 0.2 m into the landing. Reaching the goal and finishing
without physical termination raises difficulty by 0.02; an unsuccessful or physically
terminated task episode lowers it by 0.05. Imitation episodes do not change difficulty.

As difficulty increases, task resets blend toward the default standing joint pose,
zero reference velocity, and a start 0.9 m behind the reference in its yaw frame.
XY/yaw noise grows to ±0.3 m / ±0.3 rad. The goal offset grows to 0.75 m, with XY
noise up to ±0.35 m. Above difficulty 0.8, half of task resets gradually receive base
velocity noise up to ±0.15 m/s horizontally and ±0.2 rad/s in yaw. At difficulty one,
the remaining half start at rest and all task joint velocities are zero.
`Curriculum/task_resets` reports mean difficulty. This per-environment curriculum
starts at zero when creating a new environment, including checkpoint resumes.
Play disables progression and evaluates the full standing-start position/yaw range.
Task rewards include position, heading, settling, and local motion style; progress reward is disabled.
`task_motion_style` has weight 2.0 and compares joint positions (0.35 rad scale),
joint velocities (1.5 rad/s scale), and ankle/wrist positions relative to the root
in its yaw frame (0.15 m scale). A separate continuous reference phase advances
at 0.5, 0.75, 1.0, or 1.25 times the recorded rate, selecting the best local match
at each control step. Joint velocity targets scale with that rate. Global reference
position and yaw are not tracked by this reward. Reference phases cannot jump
backward or skip arbitrarily far ahead. A stationary reference frame earns style
reward only within the goal position tolerances; there, the stationary final frame
provides the standing target. The phase otherwise clamps at the clip end. This is
a finite-clip style objective, not a learned motion prior or a guarantee of style
on arbitrary stair lengths. It changes no actor/critic observations. Set
`env.rewards.task_motion_style.weight=0.0` to ablate it.
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
    workflow_name=g1-mimic-goal-pyramid-stairs \
    platform=ovx-l40s \
    output_url=swift://pdx.s8k.io/AUTH_team-isaac-sqa/isaac-sqa/jkamohara/g1-perception/runs \
    wandb_credential=wb-auth wandb_key=wb_api_key \
    args="--task IsaacContrib-Mimic-G1-29dof-Perception-Goal-Condition --num_envs 512 --logger wandb --viz newton_gl --video"
```

This is a first goal-conditioned baseline inspired by GfR, not a reproduction of its
full method. The stair baseline uses one fixed pyramid staircase and its embedded terrain metadata.
Starting-state randomization and landing goals are configured in `env_cfg/commands_cfg.py`.
The standing-start offsets are specific to these four stair clips and must be rechecked
when changing the motion dataset. Generalization to different staircases is not measured here.

## Stair ascent fine-tuning

Both fine-tuning variants disable `goal_progress`. Position, heading, settling,
and regularization rewards remain active. This applies to newly launched runs;
existing checkpoints and running OSMO snapshots retain their previous training rewards.

`...-Stairs-Finetune` uses the built-in terrain generator instead of motion-paired terrain.
It loads no motion files, so `task_motion_style` is disabled. Each 8 m square tile
has an ascending pyramid staircase and a flat ground border. Step depths are 0.22, 0.28, and 0.34 m; step heights progress
through eight difficulty rows over the configured 0.12–0.23 m range. Robots start
at rest on a sampled flat ground patch on any of the four sides, facing inward
with ±0.2 rad yaw noise. Goals are sampled in the central 1 m square of the top
landing. Spawn and goal root Z are the sampled surface height plus 0.8 m.
No episode commands a descent; falls and leaving the landing remain possible failures.

Supported goal arrivals advance a terrain level at reset; unsuccessful episodes
reduce it. Initial levels are 0–2. Play samples all levels and disables progression.
The 15-second episodes continue after success so the existing settling and joint
regularization rewards still apply. There are no imitation rewards or reference
terminations. Actor inputs remain 285-dimensional and critic inputs 598-dimensional;
the critic reference slots are always zero. Both pretrained networks are retained.

Use a late checkpoint from the masked-critic training run after task performance
has stabilized. This fine-tuning variant is task-only and may lose reference style;
changing the pretraining imitation ratio does not add imitation to this variant. Start with a fresh optimizer at the configured
1e-4 learning rate (the inherited adaptive KL schedule may subsequently change it):

```bash
uv run --with wandb isaaclab train --rl_library rsl_rl \
  --task IsaacContrib-Mimic-G1-29dof-Perception-Goal-Condition-Stairs-Finetune \
  --checkpoint wandb:jnskkmr/g1_mimic_perception_goal_condition/RUN_ID \
  --no-resume_optimizer_state \
  --num_envs 4096 --logger wandb --viz newton_gl --video

uv run --with wandb --extra ovrtx isaaclab play --rl_library rsl_rl \
  --task IsaacContrib-Mimic-G1-29dof-Perception-Goal-Condition-Stairs-Finetune-Play \
  --wandb_run FINETUNE_RUN_ID \
  --wandb_project g1_mimic_perception_goal_condition_stairs_finetune \
  --num_envs 50 --viz newton_rtx --video
```

`--checkpoint /absolute/path/model_XXXXX.pt` also works. Optimizer reset does not
reset the checkpoint iteration counter; terrain progression here depends on episode
success, not that counter. These commands start a new fine-tuning run and leave the
original training task unchanged.

## Mixed stair fine-tuning

`...-Stairs-Mixed-Finetune` retains 50% imitation episode sampling on the unchanged
reference staircase. The remaining episodes climb the generated staircases described
above, with no reference observations or global tracking rewards. Local task style
reward remains enabled using the loaded motion clips. Only task outcomes update
stair difficulty. The actor remains reference-free in both modes; the critic receives
a reference only during imitation. No new motion files are required because the
reference global root trajectories are not imposed on the randomized stairs.

The ratio is fixed during this fine-tuning run and can be changed with
`env.commands.motion.imitation_probability`. A zero ratio removes imitation entirely;
that is a separate experiment and may lose the learned style. The `-Play` task forces
zero imitation and evaluates the generated stairs.

```bash
uv run --with wandb isaaclab train --rl_library rsl_rl \
  --task IsaacContrib-Mimic-G1-29dof-Perception-Goal-Condition-Stairs-Mixed-Finetune \
  --checkpoint 'wandb:jnskkmr/g1_mimic_perception_goal_condition/89ljv3g6?checkpoint=22500' \
  --no-resume_optimizer_state \
  --num_envs 4096 --logger wandb --viz newton_gl --video

uv run --with wandb --extra ovrtx isaaclab play --rl_library rsl_rl \
  --task IsaacContrib-Mimic-G1-29dof-Perception-Goal-Condition-Stairs-Mixed-Finetune-Play \
  --wandb_run FINETUNE_RUN_ID \
  --wandb_project g1_mimic_perception_goal_condition_stairs_finetune \
  --num_envs 50 --viz newton_rtx --video
```
