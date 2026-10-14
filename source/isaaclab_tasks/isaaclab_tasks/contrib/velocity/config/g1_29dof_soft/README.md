## Training
```bash
uv run --with wandb isaaclab train --rl_library rsl_rl --task IsaacContrib-Velocity-Flat-G1-29dof-Soft --num_envs 4096 --viz newton_gl --video
```

## Training with Isaac Sim PhysX

Run from the repository root. The **flat** task supports `physics=isaacsim_physx`;
`--extra isaacsim` installs the Isaac Sim dependencies. Select this preset explicitly
for comparison with `IsaacLab_GT`: `physics=physx` auto-selects an available PhysX
backend and may select standalone OV PhysX instead.

First run a small training smoke check with local TensorBoard logging:

```bash
uv run --extra isaacsim isaaclab train --rl_library rsl_rl \
    --task IsaacContrib-Velocity-Flat-G1-29dof-Soft \
    physics=isaacsim_physx --num_envs 64 --max_iterations 2 --seed 42 \
    --logger tensorboard --experiment_name g1_29dof_soft_flat_physx_smoke \
    --viz newton_gl --video
```

Then train with a separate W&B project and local experiment directory:

```bash
uv run --extra isaacsim --with wandb isaaclab train --rl_library rsl_rl \
    --task IsaacContrib-Velocity-Flat-G1-29dof-Soft \
    physics=isaacsim_physx --num_envs 4096 --seed 42 \
    --logger wandb --log_project_name g1_29dof_soft_flat_physx_ppo \
    --experiment_name g1_29dof_soft_flat_physx_ppo \
    --viz newton_gl --video
```

These commands keep the same Newton GL visualizer and video recorder used for
Newton training. `newton_gl` can visualize PhysX through the scene-data provider;
selecting PhysX does not require switching to Kit visualization.
`--log_project_name` selects the destination
W&B project; `--wandb_project` selects the source project when loading a checkpoint
with `--wandb_run`.

The PhysX preset already sets `enable_external_forces_every_iteration=True` so the
soft-contact wrench on each robot foot is applied throughout the PhysX solver
iterations. The RFT model remains enabled, with its flat contact surface at world
z=0. Switching physics does not switch the robot USD or disable symmetry
augmentation. For a backend comparison, keep the robot asset, RFT model, material
randomization, rewards, observations, PPO settings, and seed matched; use
`physics=newton_mjwarp` for the Newton run. This does not reproduce all of the older
`IsaacLab_GT` task settings automatically.

The **rough** task is currently Newton-only because its RFT terrain queries use
Newton raycasting. These PhysX instructions apply only to the flat task.

## Inference
```bash
uv run --with wandb isaaclab play --rl_library rsl_rl --task IsaacContrib-Velocity-Flat-G1-29dof-Soft-Play --num_envs 4 --viz newton_gl --wandb_run 4blqnxbi
uv run --with wandb --extra ovrtx isaaclab play --rl_library rsl_rl --task IsaacContrib-Velocity-Flat-G1-29dof-Soft-Play --num_envs 4 --viz newton_rtx --wandb_run 4blqnxbi
```


## Symmetry augmentation

The registered PPO runner enables left/right data augmentation using the shared
[term-driven mirror implementation](../vel_mdp/README.md). Policy, critic,
privileged, and logging observation terms all declare mirror functions. Joint
signals swap left/right channels with the appropriate signs; quaternions use
XYZW order. Hybrid contact observations swap feet and reflect force vectors,
while terrain material scalars use an identity mirror.

`physics_callback` computes soft-contact forces during simulation but has **zero
policy action dimensions**. `MirrorPhysicsCallbackActionCfg` gives its empty
action slice `mirror_identity`; augmentation does not execute or modify the
contact solver. Its derived contact observations are mirrored separately because
they are visible to the critic.

Hybrid force observations now threshold the magnitude of each force component,
retaining both positive and negative components before signed-log compression.
This is necessary for reflected observations to match observations computed from
reflected forces. Observation/action dimensions are unchanged, but resuming an
existing run uses corrected force inputs and enables augmented training samples.
Set `algorithm.symmetry_cfg=None` in the runner config to disable augmentation.

## Rough terrain

The rough tasks use Newton raycasting against the generated terrain mesh. Each
foot collider sample casts a downward ray from `terrain_ray_height` meters above
the sample. Penetration is the terrain hit's world z coordinate minus the sample's
world z coordinate. Missed rays produce no RFT force. Queries run inside the RFT
CUDA graph and exclude per-environment robot shapes.

```bash
uv run --with wandb isaaclab train --rl_library rsl_rl --task IsaacContrib-Velocity-Rough-G1-29dof-Soft --num_envs 4096
uv run isaaclab -p scripts/environments/zero_agent.py --task IsaacContrib-Velocity-Rough-G1-29dof-Soft-Play --num_envs 4 --max_steps 200 --viz newton_gl
```

This path requires Newton physics and the `3D-warp` contact backend. The rough
terrain's rigid collider is disabled to avoid applying both rigid and RFT forces
at the same surface. RFT acts on the configured ankle bodies; it does not add
terrain contact forces to other robot bodies. Robot self-collision remains enabled.
The terrain mesh must be static in Newton's global world.

The force equations and material randomization are unchanged. Depth remains
gravity-aligned, including on slopes: this models terrain height profiles, not
overhangs or a separate vertical-wall contact law. The ray start offset (default
20 m) must exceed the terrain relief and expected foot penetration. Spawn heights follow the terrain. `G1RewardsRoughCfg` matches the rigid rough task's
reward weights and parameters, with hybrid contact handling for the RFT feet.
The stance-angle penalty applies throughout contact, and standing contact-break
and foot-motion penalties discourage stepping at zero command. Base height and
foot clearance use the same height scanner as the rigid task: a 0.2 × 0.2 m grid
at 0.1 m resolution. The flat task retains its separate `G1RewardsCfg`.

The flat task keeps its original RFT surface at world z=0 and its rigid ground
beneath that surface. It performs no additional terrain raycasts.
