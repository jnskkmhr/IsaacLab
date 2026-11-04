# G1 moving patch with Particle NeRD

This task defines its own environment, scene, learning terms, physics, and PPO
configuration. It follows [the task construction guide](../../../../../skills/env_construction.md):
no environment or term configuration is inherited from another task.

```text
g1_29dof_particle_nerd_moving_patch/
    agents/
        rsl_rl_ppo_cfg.py
    env_cfg/
        action_cfg.py
        commands_cfg.py
        curriculum_cfg.py
        event_cfg.py
        observation_cfg.py
        physics_cfg.py
        reward_cfg.py
        scene_cfg.py
        termination_cfg.py
        terrain_cfg.py
        visualizer_cfg.py
    mdp/
        __init__.py
        events.py
    __init__.py
    particle_nerd_env_cfg.py
    particle_nerd_env.py
```

`particle_nerd_env_cfg.py` assembles the local configs and defines train/play
settings. `env_cfg/terrain_cfg.py` contains the terrain generators and moving-patch
configuration. `particle_nerd_env.py` derives directly from `ManagerBasedRLEnv` and owns
particle reset, recycling callbacks, and collider contact bookkeeping. `agents/`
contains the PPO architecture, observation groups, symmetry augmentation, and
training hyperparameters.

The task retains the previous rewards, observations, actions, curriculum, PPO
settings. Playback uses a deterministic terrain-relative reset and a constant
1 m/s forward command with zero yaw rate. `mdp/__init__.py` reuses locomotion functions from
`g1_29dof_mpm_moving_patch_particle_cache`; `mdp/events.py` samples reset heights
against this task's terrain configuration. Terrain import, particle sampling,
and viewer implementations now come from the simpler
`g1_29dof_mpm_moving_patch` task, with their settings defined locally. Neural
dynamics and recycling remain in
[the reusable Particle NeRD package](../particle_nerd/README.md).

Registered tasks:

- `IsaacContrib-Velocity-Sand-G1-29dof-ParticleNeRD-MovingPatch`
- `IsaacContrib-Velocity-Sand-G1-29dof-ParticleNeRD-MovingPatch-Play`

The default neural checkpoint is bundled at `../particle_nerd/checkpoint/best.pt`
and selected directly in `env_cfg/physics_cfg.py`. From the repository root:

```bash
uv run isaaclab random_agent --task IsaacContrib-Velocity-Sand-G1-29dof-ParticleNeRD-MovingPatch --num_envs 2
```

To use another neural checkpoint, set the solver configuration directly:

```python
from isaaclab_tasks.contrib.g1_29dof_particle_nerd_moving_patch.particle_nerd_env_cfg import G1ParticleNeRDEnvCfg

cfg = G1ParticleNeRDEnvCfg()
cfg.scene.num_envs = 2
nerd = cfg.sim.physics.solver_cfg.entries[1].solver_cfg
nerd.checkpoint = "/absolute/path/to/best.pt"
nerd.active_k = "auto"
nerd.effective_mass_scale = 100.0
nerd.tangent_mass_scale = 40.0
```

`env_cfg/physics_cfg.py` defines MJWarp rigid integration, the neural particle
entry, and lagged two-way proxy coupling. There are zero native MPM warm-up or
settling steps. Particle-state archives, native-MPM grid capacity, and settling controls are absent from
this task's configuration. The existing `MPMObjectCfg` and material schema provide
particle storage and material features only.

The current checkpoint requires CUDA, two box colliders per world, and its trained
timestep. Particle material randomization remains disabled. Defaults use particle
rendering; viewer settings live in `env_cfg/visualizer_cfg.py` and the top-level
train/play configs. PPO logging destinations retain the previous values and can
be changed in `agents/rsl_rl_ppo_cfg.py`.


## Run
```bash
uv run --with wandb --extra ovrtx isaaclab play \
  --rl_library rsl_rl \
  --task IsaacContrib-Velocity-Sand-G1-29dof-ParticleNeRD-MovingPatch-Play \
  --num_envs 1 \
  --wandb_run 5f1xyc21 \
  --wandb_project g1_29dof_mpm_mixed_terrain \
  --viz newton_rtx \
  "env.video_recorders=[]"
```
## Fixed-size moving patch

The task now uses the non-caching moving-patch behavior. The current particle pool
follows the robot; overlapping particles keep their deformation, and departing
slots are reset with fresh terrain positions and three identical neural-history
frames at the entering edge. Previously visited terrain is not saved. Footprints
disappear after leaving the patch and are not restored on a return visit.

The number of particles stays fixed, and travel does not invalidate solver graphs.
The GL/RTX viewers draw the current patch only, with particles enabled by default.
Existing task IDs and policy playback commands continue to work. No native MPM
construction, settling, or steps are added.

## Alignment with the fixed-terrain task

The shared solver supplies the same corrected proxy-inertia refresh, beginning-of-step
collider inputs, and triangle-mesh collision classification as the validated fixed-terrain
implementation. The task proxies only the two ankle-roll bodies into the neural entry.
The robot's physical link masses are unchanged; contact reconstruction uses the virtual
inertia assigned by the coupler.

Task defaults explicitly match the fixed-terrain contact settings: normal material mass
scale 100, tangential scale 40, horizontal scale 8, upward scale 2, 200 contact iterations,
and relaxation 0.02. Top-K uses `active_k="auto"`, 512 anchors, and refreshes every four
physics steps. The foot margin is 0.02 m, sampling voxel size is 0.04 m, and the sand layer
is 0.3 m deep in both train and play configurations. Physics runs at 200 Hz and policy
updates at 50 Hz.

The moving patch still uses its own terrain height chart, kinematic particle boundary,
and recycling rather than the fixed testbed's retaining walls. Matching solver settings
does not establish identical locomotion or network accuracy between those two scenes.

## Mixed neural sand and rigid terrain

The mixed task follows `g1_29dof_mpm_moving_patch`'s terrain layout: the first half
of environment IDs contain neural sand, and the second half contain rigid ground.
It requires an even `num_envs >= 2`. Assignment stays fixed across resets; the first
half of terrain columns support sand and the remaining columns support rigid ground.
With five columns, the two groups see two and three terrain types respectively.
The sand support mesh is lowered by the particle depth; the rigid surface stays at
the generated height. The training wave amplitude range matches the MPM task at
0.1–0.4 m.

Only sand environments allocate particles, neural history, and inference buffers.
They retain the validated checkpoint, Top-K selection, 100/40 material mass scales,
0.3 m particle depth, and 0.02 m foot margin. Rigid environments use a zero foot
margin. The shared contact observations and rewards include both neural reactions
and MJWarp ground contact forces. Resetting a rigid environment leaves the other
environments' particle history untouched. There are zero native MPM steps and no
particle-state archive.

The task's existing PPO architecture, observations, and rewards are retained.
Material randomization remains disabled because neural material features are fixed
at construction. This adds the MPM task's mixed contact layout, not its newer
observation-bias curriculum or compositional PPO training settings.

```bash
uv run --with wandb isaaclab train --rl_library rsl_rl \
  --task IsaacContrib-Velocity-G1-29dof-ParticleNeRD-MovingPatch-MixedTerrain \
  --num_envs 256 --video --video_length 200 --wandb_run 5f1xyc21 --wandb_project g1_29dof_mpm_mixed_terrain

uv run --with wandb --extra ovrtx isaaclab play --rl_library rsl_rl \
  --task IsaacContrib-Velocity-G1-29dof-ParticleNeRD-MovingPatch-MixedTerrain-Play \
  --num_envs 2 --wandb_run 5f1xyc21 \
  --wandb_project g1_29dof_mpm_mixed_terrain --viz newton_rtx \
  'env.video_recorders=[]'
```

`mixed_env_cfg.py` selects the terrain generator and group-specific foot margins.
`env_cfg/mixed_scene_cfg.py` defines cloning and terrain import. `mixed_env.py`
combines contact reporting and maps scene reset IDs to compact particle storage.
The viewer tracks environment 0 by default; set
`++env.sim.visualizer_cfgs.1.origin_env_index=1` to inspect the rigid robot in a
two-environment RTX run.
