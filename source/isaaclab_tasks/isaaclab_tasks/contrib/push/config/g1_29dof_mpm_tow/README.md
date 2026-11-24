# G1 tow on streaming MPM sand

G1 walks on sand and tows a sled by a rope tied to the back of its pelvis. The robot's feet and the sled
both deform the sand, and each carries its own streaming pool with a tile cache, so the trench the sled
leaves stays in the terrain.

| ID | Config | Use |
|---|---|---|
| `IsaacContrib-Tow-G1-29dof-MPM` | `tow_env_cfg.py:G1TowEnvCfg` | training |
| `IsaacContrib-Tow-G1-29dof-MPM-Play` | `tow_env_cfg.py:G1TowEnvCfg_PLAY` | straight forward command, no robot randomization |

## What the task is

- **Robot, actions, observations, rewards, terminations:** those of the walking task
  `IsaacContrib-Velocity-G1-29dof-MPM`, copied into this package. The observation layout is unchanged, so
  a walking checkpoint loads as it is. The sled is not observed.
- **Commands:** forward only, 0 to 1 m/s, yaw rate within 0.5 rad/s. The rope cannot push.
- **Terrain:** flat. **Sand:** one fixed material (`env_cfg/scene_cfg.py:SAND_MATERIAL`), no material
  randomization.
- **Sled:** a box, `SLED_SIZE`, `SLED_MASS`, `SLED_FRICTION` in `env_cfg/scene_cfg.py`. It is put behind the
  robot at every reset (`mdp/events.py:reset_object_behind_robot`).
- **Rope:** `env_cfg/tether_cfg.py:TetherCfg`. A one-sided spring and damper between a point on the pelvis
  and a point on the sled's front face (`contrib/push/util/tether.py`). `tow_env.py:G1TowEnv` recomputes
  the tension once per environment step and holds it for the step.
- **Logged:** `Metrics/tow/tension`, `Metrics/tow/taut_fraction`, `Metrics/tow/sled_speed`.

## Physics presets

Select with `physics=NAME`: `solver_cfg_all_active_cache` (default, every loaded particle is simulated) or
`solver_cfg_cache` (settled sand sleeps). Both use the tile cache, which several pools need.

## Run

```bash
uv run python scripts/benchmarks/t2/tow_check.py --checkpoint <walking model.pt> --masses 0,5 \
    --out logs/t2/tow/check
```

`tow_check.py` runs a walking policy on the playback task with one sled mass per environment, on sand or,
with `--ground rigid`, on rigid ground with the same rope and sled.

## Not done

- No rigid-ground task package; the rigid comparison exists only inside `tow_check.py`.
- No tow policy has been trained.
