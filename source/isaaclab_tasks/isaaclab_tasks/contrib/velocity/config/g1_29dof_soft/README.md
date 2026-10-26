## Agent configuration

`agents/compositional_ppo_cfg.py` configures compositional PPO for the flat and rough tasks,
with independent actor and critic MLPs. Commands are a separate, single-frame observation
group; policy, critic, and privileged observations retain their ten-frame histories.
Matching ordinary RSL-RL checkpoints with separate MLP actor/critic weights can be loaded directly.

## Flat training baseline

Flat training matches run `4blqnxbi` in material sampling, reset randomization,
the baseline reward weights, and curricula, except for the intentional changes below. The RFT model uses `3D-warp` without EMA filtering,
with stiffness sampled from `(0.1, 0.9)`. The terrain retains equal rigid and soft
columns. The robot starts at z=0.76 m.

`feet_pitch_contact` penalizes ankle-roll body pitch throughout rigid or soft contact,
including continued stance; swing feet contribute zero. `feet_pitch` and
`feet_pitch_diff` are commented out. Tracking weights stay at 4/4/1 for linear
velocity, heading, and angular velocity; their standard deviations change from
`sqrt(0.5)` to 0.25 only after 480,000 environment steps. Flat training disables
symmetry augmentation and mirror loss, matching the vanilla PPO baseline.

Intentional differences are Newton physics with the current Newton robot asset,
compositional PPO, commands outside history, the two disabled pitch rewards, and
continuous contact gating for `feet_pitch_contact` instead of touchdown-only gating,
and the trailing-foot penalty described below.
The current robot asset has different rigid foot collision geometry from the
historical PhysX asset. This is a training-configuration alignment, not an exact
cross-backend reproduction. The historical checkpoint's input dimensions include
command history and therefore do not match this configuration directly.

## Trailing stance feet

Flat soft training includes `feet_trailing_distance` with weight -2.0. It penalizes
only the squared distance beyond 0.30 m behind the pelvis for contacting ankle-roll
body origins, measured along the robot's yaw-only forward axis. Swing feet and feet
within the allowance contribute zero; contact comes from the active rigid or soft
solver. A foot 0.40 m behind the pelvis contributes -0.02 before reward-manager time
scaling. This is a placement penalty, not a heel-contact or support-polygon constraint.

The initial allowance was calibrated using run `7cawp916`, checkpoint `model_16500.pt`,
on `IsaacContrib-Velocity-Sand-Flat-G1-29dof-MPM-Play`: two rollouts per forward command,
excluding the first second and contacts near the rigid-platform/sand boundary.
Across rigid and sand samples, the larger per-surface 99th-percentile trailing distances
were 0.14, 0.23, 0.31, and 0.42 m at commands of 0.5, 1.0, 1.5, and 2.5 m/s.
The allowance leaves ordinary walking mostly unaffected but can penalize normal
late stance at running speeds. Its weight is an initial tuning choice; improved
training behavior has not yet been established. Rough soft training is unchanged.

## Training
```bash
uv run --with wandb isaaclab train --rl_library rsl_rl \
--task IsaacContrib-Velocity-Flat-G1-29dof-Soft --num_envs 4096 \
--viz newton_gl --video
```

## Training with Isaac Sim PhysX

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
z=0. Switching physics does not switch the robot USD or the task's symmetry settings.
For a backend comparison, keep the robot asset, RFT model, material
randomization, rewards, observations, PPO settings, and seed matched; use
`physics=newton_mjwarp` for the Newton run. This does not reproduce all of the older
`IsaacLab_GT` task settings automatically.

The **rough** task is currently Newton-only because its RFT terrain queries use
Newton raycasting. These PhysX instructions apply only to the flat task.

## Inference
```bash
uv run --with wandb isaaclab play --rl_library rsl_rl --task IsaacContrib-Velocity-Flat-G1-29dof-Soft-Play --num_envs 4 --viz newton_gl --checkpoint /path/to/compatible/model.pt
uv run --with wandb --extra ovrtx isaaclab play --rl_library rsl_rl --task IsaacContrib-Velocity-Flat-G1-29dof-Soft-Play --num_envs 4 --viz newton_rtx --checkpoint /path/to/compatible/model.pt
```


## Symmetry augmentation

The rough PPO runner enables left/right data augmentation using the shared
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

Critic contact-force observations match `IsaacLab_GT`: each force component is
retained only when `force > threshold`, then passed through signed-log compression.
Negative components are discarded. This changes critic preprocessing only, not the
physical contact wrench applied to the robot or the raw force logging observation.
Observation/action dimensions are unchanged. With this GT-compatible filter,
mirroring a filtered force observation is not equivalent to filtering a reflected
force. Flat training disables symmetry. For rough training, set `algorithm.symmetry_cfg=None`
in the runner config to disable augmentation and mirror loss.

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
