# G1 flat RFT on Isaac Sim PhysX

Reproduces the task recipe saved by [4blqnxbi](https://forge.coreweave.com/wandb/jnskkmr/g1_29dof_soft_vanilla_ppo/runs/4blqnxbi) using Lab 3.0. Reference source commit: `b14910d2f3f82dae810801a73ec3f1a91bac677e`, with the run's saved `IsaacLab.diff` selecting 3D Warp RFT and stiffness randomization.

This is a separate manager-based task. It does not inherit another task's configuration. Rigid contacts use Isaac Sim PhysX; the soft layer uses analytical 3D RFT forces. No MPM particles or Newton physics solver are used.

| Setting | Reference and this task |
|---|---|
| Robot | Original `g1_29dof_rev_1_0.usd`, not the later box-foot variant; same actuator gains, armature, effort and velocity limits |
| Terrain | 10 × 10 tiles, each 8 × 8 m; 50% hard ground, 50% hard support lowered from 0 to −0.12 m by terrain curriculum |
| Soft surface | World z = 0; box-foot RFT integration with 150 points per foot; no EMA filtering |
| Material sampling | Friction 0.1–1.0, internal friction/stiffness parameter 0.1–0.9, bulk density 1000–3000 kg/m³, packing ratio 0.5–1.0 |
| Timing | 200 Hz physics, 50 Hz policy, 20 s episodes |
| Actor | 1000 inputs: ten samples of angular velocity, projected gravity, 7-component command, joint position/velocity, and previous action |
| Critic | 1010 proprioceptive/command inputs plus 160 privileged inputs |
| Quaternion observations | Scalar-first (wxyz), converted at the observation boundary; simulation remains Lab 3.0 xyzw |
| Corruption | Disabled, as saved by the reference run |
| Rewards | All 27 reference terms and weights, including foot separation −2, air time +0.5, touchdown-only foot pitch −4, foot sliding −0.25, base-height target 0.75 m above world z = 0 |
| Curriculum | Original command stages at 240000 and 360000 policy steps; tracking std changes after 480000 steps (not a linear ramp) |
| PPO | Standard PPO, [512, 256, 128] ELU actor/critic, no observation normalization, no symmetry augmentation or mirror loss, entropy 0.005 |
| Batch | 4096 total environments × 24 steps = 98304 transitions/update; four GPUs × 1024 environments |

## Run locally

Run from `/home/jkamohara/isaac/experiment/ral-gm-loco-lab-3.0`:

```bash
uv sync --locked --extra rsl-rl --extra wandb --extra video --extra isaacsim
export OMNI_KIT_ACCEPT_EULA=YES
kit_usd_root=$(uv run --no-sync python -c 'import pathlib,sys; print(next((pathlib.Path(sys.prefix)/"lib/python3.12/site-packages/isaacsim/extscache").glob("omni.usd.libs-*")))')
export PYTHONPATH="$kit_usd_root${PYTHONPATH:+:$PYTHONPATH}"
python_lib_dir=$(uv run --no-sync python -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))')
export LD_LIBRARY_PATH="$kit_usd_root/bin:$python_lib_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
uv run --no-sync isaaclab train --rl_library rsl_rl \
  --task IsaacContrib-Velocity-Flat-G1-29dof-SoftContact-PhysX \
  --num_envs 128 --viz kit --video --max_iterations 50 --logger tensorboard
```

The worktree pins `usd-exchange==2.3.0`, matching Isaac Sim 6.2's USD version. Selecting Kit's USD Python and native-library directories is also necessary: merely downgrading the standalone wheel still produces incompatible Python type registrations. The OSMO template's opt-in `kit_usd=true` performs the same setup for every worker. No driver downgrade is needed.

## OSMO

```bash
bash osmo/run_multinode.sh osmo/osmo_multi_gpu.yaml \
  --pool isaac-lab-l40s-03 \
  --set num_nodes=1 num_gpu=4 num_cpu=32 memory=192 kit_usd=true \
  --set-string \
    workflow_name=isaac-lab-g1-physx-rft-flat platform=ovx-l40s \
    uv_args="--extra rsl-rl --extra wandb --extra video --extra isaacsim" \
    output_url=swift://pdx.s8k.io/AUTH_team-isaac-sqa/isaac-sqa/jkamohara/contact-benchmark/runs \
    wandb_credential=wb-auth wandb_key=wb_api_key \
    args="--task IsaacContrib-Velocity-Flat-G1-29dof-SoftContact-PhysX --num_envs 1024 --seed 42 --logger wandb --log_project_name g1_29dof_contact_benchmark --run_name physx_soft_flat_4blqnxbi --viz kit --video"
```

## Verification and limits

- Compared 825 saved task/actuator/PhysX parameter values against `4blqnxbi`: no mismatches. Function paths, visualization settings, and API-added fields are excluded from this numeric/configuration audit; relevant function behavior was inspected separately.
- The original robot USD's Git blob matches `c0bc5dcd5759e292a082063c58bf3fc18416e652`; its original configuration layers are included.
- Three local PPO updates with 16 environments passed. A longer run completed 50 PPO updates with 128 environments, including episode resets and a 40-frame Kit recording, in 61.16 s of training (startup excluded).
- Compared the old and current flat RFT implementation on 3840 sampled foot states: all forces and torques agree within `atol=1e-4, rtol=1e-5`. Maximum absolute wrench-component difference was 0.00024414; maximum `abs(new-old)/(abs(old)+1)` was 2.345e-6. This covers the force calculation, not full closed-loop PhysX equivalence.
- The software stack is intentionally newer (Isaac Sim/PhysX, Lab, Torch, Warp, and RSL-RL). Four-worker random streams and PPO reductions differ from the original single-GPU run, even though the aggregate rollout batch matches. Identical learning curves are not guaranteed.
- Contact-force debug markers are disabled; camera placement follows an asset so recorded robots remain visible. All environments remain visible.

## Relevant Newton recipe differences

| Setting | PhysX reference `4blqnxbi` | Newton `tyiby5k6` | Newton `ufmtmwr9` |
|---|---|---|---|
| Air-time reward | 0.5 | 2.0 | 2.0 |
| Commands | Inside ten-step observation history | Separate command group | Separate command group |
| Symmetry augmentation | Off | On | On |
| Mirror loss | Off | Off | 0.1 |
| RFT EMA filter | Off | Off | Off |
| Angular-velocity XY penalty | −0.05 | −0.05 | −0.05 |
| Linear-velocity Z penalty | −1.0 | −1.0 | −1.0 |

These are differences to control, not evidence that one of them caused shaking. The reference recipe retains foot separation intentionally; it is not the later no-separation benchmark recipe.
