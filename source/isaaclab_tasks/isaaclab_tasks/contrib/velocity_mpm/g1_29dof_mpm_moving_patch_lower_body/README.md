# G1 lower-body moving-patch MPM

This task trains 15 joint-position actions: six joints per leg and three waist
joints. The 14 shoulder, elbow, and wrist joints follow randomized, smooth
position targets. Random arm motion continues during walking and contributes
no policy action dimensions.

The task owns its environment, scene, physics, MDP terms, and agent configs.
It reuses the mixed-terrain runtime from `g1_29dof_mpm_moving_patch`, while
keeping standalone lower-body task configurations.
Shared rigid-task MDP functions and the existing AGILE arm-trajectory controller
remain dependencies.

## Preserved behavior

Velocity commands retain the original `UniformVelocityYawCommandCfg`, including
linear velocity, angular velocity, yaw/heading representation, sampling ranges,
standing probability, and velocity curriculum. Terrain, material randomization,
coupling, physics/control rates, and unrelated curricula retain the original configuration.
Half the environments use MPM and half use rigid terrain, with contiguous groups
assigned to separate terrain columns as in `jqur5jhm`. The first half has sand;
the second half has no particles. MPM foot margins are 0.02 m and rigid margins
are 0.0 m. An even `num_envs >= 2` is required; training defaults to 64 and play to 2.

The policy and critic still observe **all 29 joint positions and velocities**,
including randomized arms. Observation ordering, scaling, noise, history, and
privileged terms are preserved. The previous-action term contains the 15 policy
actions; the separate command group appends one height value after the existing
seven velocity/heading values. Default group widths are:

| Group | Width |
|---|---:|
| `policy` | 790 |
| `critic` | 800 |
| `command` | 8 |
| `privileged` | 160 |

Existing 29-action checkpoints cannot be loaded unchanged.

## Intentional differences

- `env_cfg/action_cfg.py`: 15 policy actions and 14 randomized arm targets.
  Arm targets resample every 2–4 seconds within configurable per-joint ranges.
  The existing trapezoidal trajectory generator limits velocity and acceleration.
- `env_cfg/commands_cfg.py`: height commands sample 0.60–0.80 m every 10 seconds.
  Height is the root's vertical distance above the **mean finite ray-hit height**,
  matching the reference run's `base_height_l2`; no particle-depth offset is added.
  In MPM regions this is the rigid support below the sand, so a 0.75 m command
  corresponds to 0.50 m above the initial 0.25 m-deep sand surface on flat terrain.
  It does not measure deformation of the MPM particle surface.
- `env_cfg/reward_cfg.py`: the fixed-height squared penalty is replaced by
  `10 * exp(-height_error**2 / std**2)`. Power, joint velocity, acceleration,
  position limits, and posture regularization include only legs and waist.
  Action-rate terms use policy action columns; the existing upper-body term
  name is retained but now covers only the three waist actions.
- `env_cfg/curriculum_cfg.py`: height-tracking `std` decreases from 0.20 m to
  0.05 m over 360,000 environment steps. Play uses 0.05 m without this curriculum.
- `env_cfg/termination_cfg.py`: the existing extreme-action condition checks
  the 15 controlled joints in action order. Its threshold is unchanged.

All other reward terms retain their original settings. Arm motions can still
indirectly affect locomotion rewards through the robot's physical response.

## Run

From the IsaacLab repository root:

```bash
uv run --with wandb isaaclab train --rl_library rsl_rl \
  --task IsaacContrib-Velocity-Sand-G1-29dof-MPM-MovingPatch-LowerBody \
  --num_envs 4
```

The corresponding play task is
`IsaacContrib-Velocity-Sand-G1-29dof-MPM-MovingPatch-LowerBody-Play`.
The default runner is standard RSL-RL PPO, with the
`g1_29dof_mpm_moving_patch_lower_body` experiment/W&B project. Compositional PPO
is available explicitly as `rsl_rl_compositional_cfg_entry_point`.
The Gaussian height reward and existing foot-slide penalty are retained.

Short integration checks establish controller/configuration compatibility;
they do not establish trained locomotion performance.


## OSMO: 40 GPUs, mixed-contact lower-body training

Run from this IsaacLab checkout. Each GPU simulates 64 environments: 32 rigid and 32 MPM.

```bash
bash osmo/run_multinode.sh osmo/osmo_multi_gpu.yaml \
  --pool isaac-lab-l40s-03 \
  --set num_nodes=40 num_gpu=1 \
  --set-string \
    workflow_name=isaac-lab-mpm-lower-body \
    platform=ovx-l40s \
    nodes_excluded="osmogroot-dgxc-k8s-forge-az41-prd1-w-0928,osmogroot-dgxc-k8s-forge-az41-prd1-w-2250" \
    output_url=swift://pdx.s8k.io/AUTH_team-isaac-sqa/isaac-sqa/jkamohara/mpm-moving-patch/runs \
    wandb_credential=wb-auth wandb_key=wb_api_key \
    args="--task IsaacContrib-Velocity-Sand-G1-29dof-MPM-MovingPatch-LowerBody --num_envs 64 --logger wandb --viz newton_gl --video --run_name mixed50_standard_ppo_support_height_40gpu"
```
