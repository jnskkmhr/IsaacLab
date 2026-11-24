# Particle NeRD

A trained network that steps a particle bed (sand) coupled to rigid bodies, used as a Newton solver in Isaac Lab.
The network replaces the implicit MPM solver at run time. MPM is used only to generate the training data.

The method is described in `docs/_build/particle_nerd_2.0.pdf`, and `design_block.md` maps each of its blocks to
the code. All commands below are run from the repository root.

## 1 Data collection

`scripts/neural/record.py` simulates the reference episodes with MPM and writes one HDF5 file per episode to
`<dataset>/<split>/<terrain>/<category>/NNNN.hdf5`. The terrains are `flat`, `slope` and `wave`. The categories
are `lateral_push`, `vertical_push`, `impact` and `rest` (a rigid object on the bed), and `freefall` and
`rest_bed` (sand only). Finished files are skipped, so a run can be repeated.

Record every episode of one terrain, one process per GPU:

```bash
uv run python scripts/neural/record.py --dataset /data/particle_nerd --terrain flat --select all --gpus 2
```

Record the episodes of one category and object:

```bash
uv run python scripts/neural/record.py --dataset /data/particle_nerd --terrain flat \
    --category lateral_push --shape cube --size-level 1
```

The OSMO workflows in `scripts/neural/osmo/` read the dataset from two private storage folders, set as
`object_url` (the object categories) and `sand_url` (`freefall` and `rest_bed`) in `train.yaml` and
`evaluate.yaml`. Replace the `swift://PRIVATE_BUCKET/...` placeholders there with your own locations.

## 2 Training

`scripts/neural/train.py` trains the network on a recorded dataset. Checkpoints (`best.pt`, `last.pt`), the
configuration and the metrics go to `logs/neural/<run name>/`.

```bash
uv run python scripts/neural/train.py --dataset /data/particle_nerd --run-name my_run \
    --rest-windows --rest-fraction 0.02 --rest-steps 800 --rest-segment 50
```

On four GPUs of one node:

```bash
uv run torchrun --standalone --nproc_per_node=4 scripts/neural/train.py --dataset /data/particle_nerd \
    --run-name my_run --rest-windows --rest-fraction 0.02 --rest-steps 800 --rest-segment 50
```

The same four-GPU training on OSMO, reading the dataset from OSMO storage (`record.yaml` and `evaluate.yaml` in
the same folder submit the other two steps):

```bash
scripts/neural/osmo/submit.sh --workflow scripts/neural/osmo/train.yaml swift://<private output folder>
```

## 3 Evaluation

`scripts/neural/evaluate.py` rolls the network out on the test episodes, which are not used in training, and
compares it with MPM: position error, and how far the bed sinks. The table is written to
`logs/neural/<run name>/<label>.json`.

```bash
uv run python scripts/neural/evaluate.py --dataset /data/particle_nerd \
    --checkpoint logs/neural/my_run/best.pt --sleep-speed 0.01
```

`--sleep-speed 0.01` keeps particles still that are predicted to move slower than 1 cm/s; it removes the slow
sinking of a resting bed. `--active-k 2048` steps only the particles near the object, as the solver option
`active_k` does.

The closed-loop benchmark in `scripts/neural/benchmark/` runs an example task with Particle NeRD and with MPM,
with the object coupled to the sand both ways, and compares the two runs. A case is an object (`box`, `sphere`,
`cylinder`) and a motion (`push`, `circle`, `stamp`, `drop`); results go to `logs/neural/benchmark/<case>/`.

```bash
uv run python scripts/neural/benchmark/run_case.py --shape box --motion push --solver nerd
uv run python scripts/neural/benchmark/run_case.py --shape box --motion push --solver mpm
uv run python scripts/neural/benchmark/kinetic_energy.py logs/neural/benchmark/box-push
uv run --with scikit-image python scripts/neural/benchmark/surface_distance.py logs/neural/benchmark/box-push
```

## 4 Integration with Isaac Lab

The solver is configured with `ParticleNeRDSolverCfg` and loads `checkpoint/best.pt` by default. The example
tasks are in `example/`. Every task has an `-MPM-v0` twin that runs the same scene with implicit MPM.

```bash
--task IsaacContrib-ParticleNeRD-Box-v0
--task IsaacContrib-ParticleNeRD-Sphere-v0
--task IsaacContrib-ParticleNeRD-Cylinder-v0
--task IsaacContrib-ParticleNeRD-Franka-v0
--task IsaacContrib-ParticleNeRD-Go2-v0
--task IsaacContrib-ParticleNeRD-G1-v0
```

Run a task with zero or random actions, and its MPM twin for comparison:

```bash
uv run python scripts/environments/zero_agent.py --task IsaacContrib-ParticleNeRD-Box-v0 --num_envs 4
uv run python scripts/environments/zero_agent.py --task IsaacContrib-ParticleNeRD-Box-MPM-v0 --num_envs 4
uv run python scripts/environments/random_agent.py --task IsaacContrib-ParticleNeRD-Go2-v0 --num_envs 4 \
    --visualizer newton_gl
```

Train a policy with RSL-RL PPO. The tasks are set up to log to Weights & Biases, which needs the `wandb` package;
`agent.logger=tensorboard` logs to `logs/rsl_rl/` instead:

```bash
uv run isaaclab train --rl_library rsl_rl --task IsaacContrib-ParticleNeRD-Go2-v0 --num_envs 16 \
    agent.logger=tensorboard
```

Play a trained policy:

```bash
uv run isaaclab play --rl_library rsl_rl --task IsaacContrib-ParticleNeRD-G1-v0 --num_envs 4 \
    --checkpoint /path/to/model.pt --viz newton_gl
```
