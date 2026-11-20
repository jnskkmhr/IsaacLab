# Major changes proposed for the MPM PR

## Published draft PRs

Prepared against Isaac Lab `develop` at `5afb1f7949e8ab02d9b97253caf2e71b5aabfd45` and Newton `main` at `abe78de5`.

| Item | Draft PR | Base | Feature commit |
| --- | --- | --- | --- |
| Newton 1: coupled particle material synchronization | [Newton #4607](https://github.com/newton-physics/newton/pull/4607) | Upstream `main` | `1d11d48a` |
| Isaac Lab 1.1: MPM material API | [Isaac Lab #8405](https://github.com/isaac-sim/IsaacLab/pull/8405) | Upstream `develop` | `a44b8dc0e` |
| Isaac Lab 3: W&B checkpoint loading | [Isaac Lab #8406](https://github.com/isaac-sim/IsaacLab/pull/8406) | Upstream `develop` | `78aed6d16` |
| Isaac Lab 4: shared symmetry augmentation | [Isaac Lab #8407](https://github.com/isaac-sim/IsaacLab/pull/8407) | Upstream `develop` | `e98c17441` |
| Isaac Lab 5: contributed G1 asset | [Isaac Lab #8408](https://github.com/isaac-sim/IsaacLab/pull/8408) | Upstream `develop` | `d04682eed` |
| Isaac Lab 2: fixed-bed G1 MPM task | [Fork #24](https://github.com/jnskkmhr/IsaacLab/pull/24) | `jnskkmhr/PR_mpm_prerequisites` | `d1754d1b4` |
| Isaac Lab 6: moving-patch task and randomization | [Fork #25](https://github.com/jnskkmhr/IsaacLab/pull/25) | `jnskkmhr/PR_mpm_prerequisites` | `8b192b9f5` |
| Isaac Lab 7: mixed rigid/MPM terrain | [Fork #26](https://github.com/jnskkmhr/IsaacLab/pull/26) | `jnskkmhr/PR_g1_moving_patch_mpm` | `2fc62af3e` |

GitHub requires the PR base branch to exist in the target repository. Therefore the three dependent drafts are in the fork, where the prerequisite branches exist. After their prerequisites merge upstream, open replacement upstream PRs containing only the remaining feature commits; a PR cannot change its base repository.

The shared generic material-event commit is `4f5dabe66`, included in both task branches. The integration branch is `3d69ff99d`; it has no separate PR. Direct coupling (Isaac Lab 8) and the standalone Newton G1 example remain deferred.

Validation completed:

- Focused material, checkpoint, symmetry, asset, reward, and GPU reset/contact tests passed.
- Newton's two material synchronization regressions failed on the unchanged solver and passed with the fix.
- Every feature commit passed its repository formatting checks. Isaac Lab changelog checks used each PR's actual base.
- The contributed asset wheel contained all nine USD payloads.
- Generated task documentation checks and the final Sphinx HTML build passed without documentation warnings.

Validation limits:

- Task GPU tests used the Newton synchronization branch and official Warp nightly `1.19.0.dev20261008`. Dependency pins were not changed in these PRs; compatible upstream dependency revisions are still required before the task PRs merge.
- Full training and GL/RTX rendering smoke tests were not run for the extracted branches.
- Simulation tests emitted existing dependency warnings; the PR descriptions do not claim warning-free runtime execution.



Scope: MPM-related changes on `junnosuke/mpm_rl_task` compared with `develop`. The file column lists the main implementation files, not every changed file or test.

| No. | Category | Main changes | Files containing the modifications |
| --- | --- | --- | --- |
| 1.1 | **MPM material setter/getter in the built-in MPM manager** | Task-independent API for reading and updating particle material parameters, including density-to-mass updates, validation, and solver synchronization. No moving-patch dependency. | [MPM manager — physics/mpm_manager.py](../../../../isaaclab_newton/isaaclab_newton/physics/mpm_manager.py)<br>[Manager API tests — test/physics/test_mpm_material_parameters.py](../../../../isaaclab_newton/test/physics/test_mpm_material_parameters.py) |
| 2 | **Fixed-bed G1 MPM locomotion example** | Depends on standalone Isaac Lab PRs 1.1, 3, 4, and 5. Adds generic uniform/log-uniform material randomization events using the material API from 1.1, and demonstrates them in the fixed-bed task without moving-patch dependencies. Retains the rigid approach platform. Aligns observations, rewards, terminations, symmetry, and sand parameters with the moving-patch task; includes both rigid-platform and MPM foot forces. | [Environment — g1_mpm_env.py](../g1_29dof_mpm/g1_mpm_env.py)<br>[Task configuration — g1_mpm_env_cfg.py](../g1_29dof_mpm/g1_mpm_env_cfg.py)<br>[Scene and material — env_cfg/scene_cfg.py](../g1_29dof_mpm/env_cfg/scene_cfg.py)<br>[Physics — env_cfg/physics_cfg.py](../g1_29dof_mpm/env_cfg/physics_cfg.py)<br>[Randomization — env_cfg/event_cfg.py](../g1_29dof_mpm/env_cfg/event_cfg.py)<br>[Observations — env_cfg/observation_cfg.py](../g1_29dof_mpm/env_cfg/observation_cfg.py)<br>[Rewards — env_cfg/reward_cfg.py](../g1_29dof_mpm/env_cfg/reward_cfg.py)<br>[Terminations — env_cfg/termination_cfg.py](../g1_29dof_mpm/env_cfg/termination_cfg.py)<br>[Contact rewards — mdp/rewards.py](../g1_29dof_mpm/mdp/rewards.py)<br>[Agent — agents/rsl_rl_ppo_cfg.py](../g1_29dof_mpm/agents/rsl_rl_ppo_cfg.py)<br>[Generic randomization event — envs/mdp/events.py](../../../../isaaclab_newton/isaaclab_newton/envs/mdp/events.py)<br>[Event exports — envs/mdp/__init__.pyi](../../../../isaaclab_newton/isaaclab_newton/envs/mdp/__init__.pyi)<br>[Randomization tests — test/physics/test_mpm_material_parameters.py](../../../../isaaclab_newton/test/physics/test_mpm_material_parameters.py) |
| 3 | **Checkpoint loading for fine-tuning and playback** | Download a W&B checkpoint and load it for RSL-RL fine-tuning or playback. Includes run/project/entity selection and checkpoint resolution. No OSMO or distributed-training changes. | [Checkpoint download — isaaclab_rl/isaaclab_rl/entrypoints/common.py](../../../../isaaclab_rl/isaaclab_rl/entrypoints/common.py)<br>[RSL-RL playback — isaaclab_rl/isaaclab_rl/entrypoints/backends/play_rsl_rl.py](../../../../isaaclab_rl/isaaclab_rl/entrypoints/backends/play_rsl_rl.py)<br>[RSL-RL training — isaaclab_rl/isaaclab_rl/entrypoints/backends/train_rsl_rl.py](../../../../isaaclab_rl/isaaclab_rl/entrypoints/backends/train_rsl_rl.py) |
| 4 | **Shared observation and action symmetry augmentation** | Reusable reflection rules, manager-based augmentation, and an RSL-RL callback in `isaaclab_contrib.mdp.symmetry`. Standalone; includes runtime/configuration code and tests only. No task term configs, G1 mappings, or `rsl_rl_ppo_cfg.py` changes. Add a configuration example to the PR description later. | [Augmentation — mdp/symmetry/symmetry.py](../../../../isaaclab_contrib/isaaclab_contrib/mdp/symmetry/symmetry.py)<br>[Configuration classes — mdp/symmetry/symmetry_cfg.py](../../../../isaaclab_contrib/isaaclab_contrib/mdp/symmetry/symmetry_cfg.py)<br>[Package exports — mdp/symmetry/__init__.pyi](../../../../isaaclab_contrib/isaaclab_contrib/mdp/symmetry/__init__.pyi)<br>[Shared MDP exports — mdp/__init__.pyi](../../../../isaaclab_contrib/isaaclab_contrib/mdp/__init__.pyi)<br>New unit tests: `source/isaaclab_contrib/test/mdp/test_symmetry.py`. |
| 5 | **Contributed G1 robot assets and configuration** | Standalone G1 asset configuration and required USD geometry/collision payloads in `isaaclab_contrib`. Retains upstream Unitree definitions in `isaaclab_assets`. No task scene changes, temporary task-local assets, or terrain textures. | Planned configuration: `source/isaaclab_contrib/isaaclab_contrib/assets/unitree.py`.<br>Proposed USD data directory: `source/isaaclab_contrib/isaaclab_contrib/assets/data/unitree/`.<br>[Current G1 configuration — isaaclab_assets/robots/unitree.py](../../../../isaaclab_assets/isaaclab_assets/robots/unitree.py)<br>[Current G1 box-foot USD and adjacent payloads](../../../../isaaclab_assets/data/unitree/g1_29dof_rev_1_0_box_foot_improved_collision/g1_29dof_rev_1_0_box_foot_improved_collision.usda)<br>[Contrib asset exports — assets/__init__.pyi](../../../../isaaclab_contrib/isaaclab_contrib/assets/__init__.pyi)<br>[Contrib package data — pyproject.toml](../../../../isaaclab_contrib/pyproject.toml) |
| 6 | **Moving-patch MPM locomotion and material randomization (includes former 1.2)** | Depends on standalone Isaac Lab PRs 1.1, 3, 4, and 5; independent of PR 2. Includes moving particle patches, recycling, terrain queries, resets, observations/rewards, symmetry wiring, GL/RTX particle rendering, and material randomization with density-consistent recycled masses. Shares the generic backend-event commit with PR 2. | [Environment — mpm_env.py](mpm_env.py)<br>[Task configuration — mpm_env_cfg.py](mpm_env_cfg.py)<br>[Task physics presets — env_cfg/physics_cfg.py](env_cfg/physics_cfg.py)<br>[Particle management — util/particles.py](util/particles.py)<br>[Warp kernels — util/kernel.py](util/kernel.py)<br>[Observations — mdp/observations.py](mdp/observations.py)<br>[Rewards — mdp/rewards.py](mdp/rewards.py)<br>[Task registration — __init__.py](__init__.py)<br>[Moving-patch GL/RTX visualizers — util/visualizer.py](util/visualizer.py)<br>[Material event wrappers — mdp/events.py](mdp/events.py)<br>[Sampling ranges — env_cfg/event_cfg.py](env_cfg/event_cfg.py)<br>[Generic randomization event — envs/mdp/events.py](../../../../isaaclab_newton/isaaclab_newton/envs/mdp/events.py)<br>[Event exports — envs/mdp/__init__.pyi](../../../../isaaclab_newton/isaaclab_newton/envs/mdp/__init__.pyi) |
| 7 | **Mixed rigid/MPM terrain training** | Stacks on combined PR 6. One policy across rigid and granular terrain, with terrain assignment, curriculum, combined contact forces, MPM-only material randomization, and fixed rigid/MPM robot foot collider margins. | [Mixed environment — mixed_env.py](mixed_env.py)<br>[Mixed task configuration — mixed_env_cfg.py](mixed_env_cfg.py)<br>[Mixed scene — env_cfg/mixed_scene_cfg.py](env_cfg/mixed_scene_cfg.py)<br>[Terrain importers — util/terrain.py](util/terrain.py)<br>[Curriculum — mdp/curriculums.py](mdp/curriculums.py) |
| 8 | **Reusable direct MJWarp–MPM coupling — on hold** | Excluded from the active PR series. Explicit, lagged contact-wrench exchange following Newton's standalone two-way MPM example. A simpler alternative to proxy/ADMM algorithms for experimentation, using the existing solver-entry infrastructure. Independent of the G1 moving-patch task. | [Direct coupler — custom_coupling/coupled_mjwarp_mpm_manager.py](../../../../isaaclab_contrib/isaaclab_contrib/custom_coupling/coupled_mjwarp_mpm_manager.py)<br>[Reusable coupler configuration — custom_coupling/newton_manager_cfg.py](../../../../isaaclab_contrib/isaaclab_contrib/custom_coupling/newton_manager_cfg.py)<br>[Wrench-exchange kernels — custom_coupling/kernels.py](../../../../isaaclab_contrib/isaaclab_contrib/custom_coupling/kernels.py)<br>[Configuration exports — custom_coupling/__init__.pyi](../../../../isaaclab_contrib/isaaclab_contrib/custom_coupling/__init__.pyi)<br>[Custom coupler configuration validation — coupling/coupler.py](../../../../isaaclab_contrib/isaaclab_contrib/coupling/coupler.py)<br>[Coupling tests — test/custom_coupling/test_mpm_coupling.py](../../../../isaaclab_contrib/test/custom_coupling/test_mpm_coupling.py) |

Asset scope: category 5 is reusable robot configuration and robot asset data. `env_cfg/scene_cfg.py` belongs to the moving-patch task in category 6, not the asset change. The contrib relocation is planned, not implemented. It must update imports, package-data inclusion, and USD payload references. Remove or replace task references to excluded terrain textures when extracting the PR.

Coupling dependency boundary: category 8 excludes the task-specific `env_cfg/physics_cfg.py`. That file remains part of category 6, where it selects and configures the task's physics. The direct coupler can be reviewed independently of the G1 task.

Terrain scope: reusable/core terrain infrastructure changes are excluded. Task-local terrain code in `util/terrain.py` remains part of categories 6–7. Extracting the task PR must resolve any reliance on the excluded core terrain changes against current `develop`.

Material-randomization dependency boundary: 1.1 contains only the material getter/setter API and its tests. The generic uniform/log-uniform event in `isaaclab_newton/envs/mdp/events.py`, its exports, and event tests form one shared commit used by PRs 2 and 6. Former PR 1.2 is merged into PR 6. After either task PR merges, refresh the other branch to exclude the already-merged backend commit.

Excluded: terrain texture assets, reusable/core terrain infrastructure changes, the standalone soft-contact model, unrelated locomotion improvements, motion-imitation tasks, OSMO/distributed-training tooling, moving-patch Kit visualization, and general Kit/OpenGL initialization fixes. Retain or relocate only shared helpers required by the MPM tasks. Include only the robot assets these tasks need; terrain texture additions are out of scope.

Visualization scope: moving-patch particle rendering is included in category 6. Retain only the OpenGL/RTX portions of `util/visualizer.py` and their task integration. The current branch also changes `kit_launcher.py` and the shared `newton_visualizer.py`; those changes still differ from current `develop`, but should be reviewed separately from this MPM PR. This list defines the intended PR scope; it does not remove the excluded implementations from the working branch.

Dependency note: [pyproject.toml](../../../../../pyproject.toml) and [uv.lock](../../../../../uv.lock) now select the official NVIDIA Warp nightly wheel `1.19.0.dev20261008`, which includes the multi-environment lookup fix and needs no manual native-library build. Newton still uses the `jnskkmhr/fix-coupler-solver-param-sync` fork. Review these dependencies when preparing the upstream PR.

## Newton-side PRs

| No. | Category | Main changes | Files containing the modifications |
| --- | --- | --- | --- |
| 1 | **Particle material synchronization between coupler and solvers** | Propagates particle material and mass updates from `coupler.model` to individual solver models when model changes are notified. Enables runtime material updates with coupled physics. Supports Isaac Lab 1.1 and the material randomization in PRs 2 and 6; independent of the moving-patch algorithm. | `newton/_src/solvers/coupled/solver_coupled.py`<br>`newton/_src/solvers/coupled/solver_coupled_proxy.py` |
| 2 | **G1 locomotion example in standalone Newton** | Demonstrates G1 policy playback with MJWarp–MPM coupling without Isaac Lab. **Low priority; defer for now.** Proposed cleanup: replace `policy.pt`/TorchScript with ONNX and replace Torch observation tensor operations to remove the PyTorch dependency. Optional; not required by the Isaac Lab PRs. | Existing example: [example_mpm_g1_policy.py](/home/jkamohara/isaac/newton/newton/examples/mpm_mujoco/example_mpm_g1_policy.py) |

## PR logistics
* Always comply contribution guide (newton: https://github.com/newton-physics/newton/blob/main/CODING_GUIDELINES.rst) and IsaacLab (https://isaac-sim.github.io/IsaacLab/main/source/refs/contributing.html)
* Follow coding guide for naming conventions, modularity etc. Newton (https://github.com/newton-physics/newton/blob/main/CODING_GUIDELINES.rst) and IsaacLab (https://isaac-sim.github.io/IsaacLab/main/source/refs/contributing.html#coding-style)
* Since we are excluding ground texture, some PRs will miss ground texture. In that case, you can do visual_material=None. This happens in terrain class.


## PR branches and dependencies

Prepare Newton in `/home/jkamohara/isaac/tmp/isaac_PR/newton` and Isaac Lab in `/home/jkamohara/isaac/tmp/isaac_PR/IsaacLab`.

| PR | Branch | Initial PR target / dependency |
| --- | --- | --- |
| Newton 1 | `jnskkmhr/PR_coupler_solver_material_sync` | Newton upstream base; separate repository |
| Isaac Lab 1.1 | `jnskkmhr/PR_coupler_solver_material_sync` | `develop`; standalone material API |
| Isaac Lab 3 | `jnskkmhr/PR_wandb_integ` | `develop`; standalone checkpoint loading |
| Isaac Lab 4 | `jnskkmhr/PR_symmetry_aug` | `develop`; standalone symmetry, no task wiring |
| Isaac Lab 5 | `jnskkmhr/PR_robot_asset` | `develop`; standalone contributed asset |
| Integration base, no feature PR | `jnskkmhr/PR_mpm_prerequisites` | Merge 1.1, 3, 4, and 5 into a temporary base from `develop`. |
| Isaac Lab 2 | `jnskkmhr/PR_g1_mpm_env` | Integration base; depends on 1.1, 3, 4, and 5 |
| Isaac Lab 6, including former 1.2 | `jnskkmhr/PR_g1_moving_patch_mpm` | Integration base; depends on 1.1, 3, 4, and 5; independent of 2 |
| Isaac Lab 7 | `jnskkmhr/PR_g1_mixed_rigid_mpm_terrain` | PR 6 branch |
| Isaac Lab 8 | On hold | No branch or implementation in this series. |

PRs 2 and 6 share the backend material-event commit while remaining separate task PRs. Rebase dependent commits and retarget to `develop` as prerequisites merge. See [PR_plan.md](PR_plan.md) for file lists, tests, changelog fragments, and stack maintenance.
