# PR implementation plan

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


Plan based on [PR_major.md](PR_major.md). Isaac Lab comparison: the refreshed PR checkout `develop` at `5afb1f794` versus `junnosuke/mpm_rl_task` at `330491555`.

Use stacked Isaac Lab PRs. Standalone PRs target `develop`; dependent PRs target the prerequisite branch described below. The branches, commits, and draft PRs are now published; see the status below.

Implementation checkout locations:

- Isaac Lab: `/home/jkamohara/isaac/tmp/isaac_PR/IsaacLab`
- Newton: `/home/jkamohara/isaac/tmp/isaac_PR/newton`

## Path notation

Paths below are relative to these directories. Braces list individual files, not an instruction to copy an entire directory.

| Prefix | Repository path |
| --- | --- |
| `N` | `source/isaaclab_newton` |
| `C` | `source/isaaclab_contrib` |
| `R` | `source/isaaclab_rl` |
| `T` | `source/isaaclab_tasks` |
| `F` | `source/isaaclab_tasks/isaaclab_tasks/contrib/velocity_mpm/g1_29dof_mpm` |
| `M` | `source/isaaclab_tasks/isaaclab_tasks/contrib/velocity_mpm/g1_29dof_mpm_moving_patch` |

**Add** means absent from the current `develop`; review each stacked PR against its own base to exclude prerequisite files. **Modify** means changing an existing file. Proposed files that do not yet exist in this branch are marked **new implementation**.

## Newton PRs

Newton paths below are relative to its own repository. Its upstream base must be checked in that checkout; the Isaac Lab `develop` revision does not identify a Newton base.

| No. | Category / branch | Plan | Files to add | Files to modify |
| --- | --- | --- | --- | --- |
| 1 | **Coupler material synchronization**<br>`jnskkmhr/PR_coupler_solver_material_sync` | Synchronize parent-model particle material and mass changes into solver models. Required by Isaac Lab 1.1 when using coupled physics. | No new runtime module planned. | `newton/_src/solvers/coupled/solver_coupled.py`<br>`newton/_src/solvers/coupled/solver_coupled_proxy.py`<br>`newton/tests/test_coupled_solver.py` — focused synchronization coverage. |
| 2 | **Standalone G1 policy example**<br>Deferred | Keep outside the initial series. Replace TorchScript inference and Torch observation operations with ONNX-compatible code when this item is resumed. | Proposed example: `newton/examples/mpm_mujoco/example_mpm_g1_policy.py`, if absent from the selected upstream base. | Determine supporting example/dependency changes when resumed. Do not include `policy.pt`. |

## Isaac Lab branch stack

| PR | Branch | Starts from / initial PR target | Dependencies |
| --- | --- | --- | --- |
| 1.1 | `jnskkmhr/PR_coupler_solver_material_sync` | `develop` | Standalone in Isaac Lab. Coupled operation still requires Newton's material-synchronization fix. |
| 3 | `jnskkmhr/PR_wandb_integ` | `develop` | Standalone |
| 4 | `jnskkmhr/PR_symmetry_aug` | `develop` | Standalone; reusable symmetry only |
| 5 | `jnskkmhr/PR_robot_asset` | `develop` | Standalone; contributed G1 asset only |
| Integration base, no feature PR | `jnskkmhr/PR_mpm_prerequisites` | `develop`, with branches 1.1, 3, 4, and 5 merged into it | Temporary common base so GitHub excludes all four prerequisites from task reviews. |
| 2 | `jnskkmhr/PR_g1_mpm_env` | `jnskkmhr/PR_mpm_prerequisites` | 1.1, 3, 4, 5 |
| 6, including former 1.2 | `jnskkmhr/PR_g1_moving_patch_mpm` | `jnskkmhr/PR_mpm_prerequisites` | 1.1, 3, 4, 5; independent of PR 2 |
| 7 | `jnskkmhr/PR_g1_mixed_rigid_mpm_terrain` | `jnskkmhr/PR_g1_moving_patch_mpm` | 6 |
| 8 | Not created | On hold | Do not implement or submit in this series. |

Create standalone branches first, then the integration base, then sibling task branches 2 and 6, and finally 7. PR descriptions must link their prerequisites and explain the temporary target branch. Do not submit the integration branch as a separate feature PR.

When prerequisites merge, rebuild the integration base from updated `develop` plus any remaining prerequisite branches, and replay only dependent commits. Once all four standalone PRs merge, open replacement upstream PRs for 2 and 6 against `develop`. When 6 merges, replay only PR 7's commits onto updated `develop` and open a replacement upstream PR for 7. Track the old base before rebasing; after a squash merge, use that boundary to exclude original prerequisite commits rather than replaying them.

Generic material-randomization events remain excluded from 1.1. They are needed by both 2 and 6. Prepare one shared backend-event commit on the integration base, then start both task implementations from that commit. Both PRs initially show that same backend change; after the first merges, refresh the second branch/base so the change disappears from its diff. Do not introduce a dependency from 6 to the fixed-bed task or create another standalone PR.

## Isaac Lab file plan

| No. | Category | Plan | Files to add | Files to modify |
| --- | --- | --- | --- | --- |
| 1.1 | **MPM material getter/setter API** | Validated material getters/setters and density-to-mass updates. No randomization events or task code. | `N/test/physics/test_mpm_material_parameters.py` | `N/isaaclab_newton/physics/mpm_manager.py` |
| 3 | **W&B checkpoint loading** | Independent checkpoint selection, download, playback, and fine-tuning. Exclude OSMO/distributed-training changes. | No new runtime or test module planned. | `R/isaaclab_rl/entrypoints/common.py`<br>`R/isaaclab_rl/entrypoints/backends/{play_rsl_rl.py,train_rsl_rl.py}`<br>`R/test/{test_entrypoints.py,test_entrypoints_common.py}` — checkpoint-specific changes only. |
| 4 | **Shared symmetry augmentation** | Reusable implementation and tests only. Put a configuration example in the PR description later. No task term configs, G1 mappings, or RSL-RL task config changes. | `C/isaaclab_contrib/mdp/symmetry/{__init__.py,__init__.pyi,symmetry.py,symmetry_cfg.py}`<br>`C/test/mdp/test_symmetry.py` — **new implementation**. | `C/isaaclab_contrib/mdp/__init__.pyi` |
| 5 | **Contributed G1 assets** | Add the asset directly to `isaaclab_contrib`. No temporary task-local asset and no task scene changes. Keep upstream Unitree definitions intact. | `C/isaaclab_contrib/assets/unitree.py`<br>Required USD payloads under `C/isaaclab_contrib/assets/data/unitree/`.<br>`C/test/assets/test_unitree_g1.py` — **new implementation**, asset composition/packaging coverage. | `C/isaaclab_contrib/assets/__init__.pyi`<br>`C/pyproject.toml` — package the USD data. |
| 2 | **Fixed-bed G1 MPM example and generic randomization** | Use all four standalone PRs. Keep the rigid approach platform. Include symmetry wiring, local G1 mappings/helpers, and material randomization from the outset. | Fixed-bed task files below.<br>`T/test/contrib/test_g1_mpm.py` — **new implementation**. | Shared backend-event commit: `N/isaaclab_newton/envs/mdp/{events.py,__init__.pyi}` and `N/test/physics/test_mpm_material_parameters.py`.<br>Generate the environment listing for task registration. |
| 6 | **Moving-patch task and material randomization** | Combine former 6 and 1.2. Include recycling, terrain queries, graph-compatible updates, GL/RTX particle rendering, material event wrappers, density-consistent recycling, and symmetry wiring. Reuse the contributed G1 asset. Proxy coupling initially; no direct-coupler or Kit dependency. | Moving-patch task files below.<br>`T/test/contrib/{test_mpm_moving_patch_reset.py,test_mpm_moving_patch_rewards.py}`. | Same shared backend-event commit as PR 2 if not yet merged.<br>Generate the environment listing. Include any other infrastructure change only if isolated validation proves it necessary. |
| 7 | **Mixed rigid/MPM terrain** | Stack on combined 6. Add terrain assignment, curriculum, contact handling, and MPM-only randomization. Use fixed `rigid_contact_margin` and `mpm_contact_margin`. | `M/{mixed_env.py,mixed_env_cfg.py}`<br>`M/env_cfg/mixed_scene_cfg.py` | `M/{__init__.py,mpm_env.py}`<br>`M/env_cfg/__init__.py`<br>`M/util/terrain.py`<br>`M/mdp/curriculums.py`<br>`T/test/contrib/{test_mpm_moving_patch_reset.py,test_mpm_moving_patch_rewards.py}`<br>Generated environment listing. |
| 8 | **Direct MJWarp–MPM coupler — on hold** | No implementation, branch, tests, or fragments in the active series. | None while on hold. | None while on hold. |

## Task files to add

Both task directories are absent from the checked `develop`. These are selective additions, not a directory copy including local reports and experiments.

| PR | Directory | Files to add |
| --- | --- | --- |
| 2 | `F/` | `__init__.py`<br>`g1_mpm_env.py`<br>`g1_mpm_env_cfg.py`<br>`README.md` — concise example usage. |
| 2 | `F/agents/` | `__init__.py`<br>`rsl_rl_ppo_cfg.py` |
| 2 | `F/env_cfg/` | `__init__.py`<br>`action_cfg.py`<br>`commands_cfg.py`<br>`curriculum_cfg.py`<br>`event_cfg.py`<br>`observation_cfg.py`<br>`physics_cfg.py`<br>`reward_cfg.py`<br>`scene_cfg.py`<br>`termination_cfg.py` |
| 2 | `F/mdp/` | Existing task files: `__init__.py`, `__init__.pyi`, `events.py`, `observations.py`, `rewards.py`, `terminations.py`.<br>`symmetry.py` — G1 mappings extracted from the current rigid-task helper.<br>**New implementation:** `curriculums.py`, plus `commands/{__init__.py,__init__.pyi,commands_cfg.py,velocity_command.py}` for the required velocity/yaw command and curriculum helpers currently imported from the rigid task. |
| 6 | `M/` | `__init__.py`<br>`mpm_env.py`<br>`mpm_env_cfg.py`<br>`README.md` — concise example usage. |
| 6 | `M/agents/` | `__init__.py`<br>`rsl_rl_ppo_cfg.py` |
| 6 | `M/env_cfg/` | `__init__.py`<br>`action_cfg.py`<br>`commands_cfg.py`<br>`curriculum_cfg.py`<br>`event_cfg.py`<br>`observation_cfg.py`<br>`physics_cfg.py`<br>`reward_cfg.py`<br>`scene_cfg.py`<br>`termination_cfg.py`<br>`terrain_cfg.py` |
| 6 | `M/mdp/` | `__init__.py`<br>`__init__.pyi`<br>`curriculums.py`<br>`events.py`<br>`observations.py`<br>`rewards.py`<br>`terminations.py`<br>`symmetry.py` — G1 mappings.<br>**New implementation:** `commands/{__init__.py,__init__.pyi,commands_cfg.py,velocity_command.py}` for required G1 command behavior. |
| 6 | `M/util/` | `__init__.py`<br>`kernel.py`<br>`particles.py`<br>`solver_setting.py`<br>`terrain.py`<br>`visualizer.py` |

For PR 2, reuse core MDP implementations where behavior matches. Put the remaining required rewards, observations, and terminations into the existing task-local modules. Do not introduce a new general-purpose shared module for these terms. PR 6 must likewise include the required helpers in its own task-local modules; it must not import PR 2 or the excluded rigid/soft tasks. Reuse existing core helpers where they match the required behavior.

Keep `env_cfg/physics_cfg.py` in both tasks: it contains their solver configuration. Excluding it from the reusable direct-coupler PR does not mean removing task physics configuration.

## G1 asset files to add in standalone PR 5

Destination: `C/isaaclab_contrib/assets/data/unitree/g1_29dof_rev_1_0_box_foot_improved_collision/`.

PRs 2 and 6 import this contributed asset directly. There is no temporary `F/asset/` directory or later relocation.

Copy only the variant used by the example. Its current source is `source/isaaclab_assets/data/unitree/g1_29dof_rev_1_0_box_foot_improved_collision/`.

| File | Purpose |
| --- | --- |
| `g1_29dof_rev_1_0_box_foot_improved_collision.usda` | Robot entry point |
| `payloads/base.usda` | Base payload composition |
| `payloads/geometries.usd` | Robot geometry |
| `payloads/instances.usda` | Geometry instances |
| `payloads/materials.usda` | Robot visual materials |
| `payloads/robot.usda` | Robot hierarchy |
| `payloads/Physics/physics.usda` | Common physics properties |
| `payloads/Physics/mujoco.usda` | MuJoCo physics variant |
| `payloads/Physics/physx.usda` | PhysX physics variant |

Check USD references and package installation before committing. Ground textures are excluded; robot geometry and robot materials are retained. Use `visual_material=None` for the optional textured-terrain material.

## Dependency and scope checks

| Issue in the current branch | Planned handling |
| --- | --- |
| Task code imports excluded rigid-task helpers. | Include only required G1 helpers locally in each task PR. Wire the shared symmetry API from PR 4 in PRs 2 and 6. Preserve observation/action order and command dimensions. |
| Moving-patch code imports direct-coupler and Kit classes. | Remove those imports and associated branches from the initial PR 6 extraction. Retain GL/RTX particle rendering. Direct coupling remains PR 8; Kit-specific fixes remain excluded. |
| Mixed terrain requires distinct margins. | PR 7 uses fixed margins for rigid and MPM environments, not margin randomization. |
| Current lockfile includes fork-specific Newton and nightly Warp changes. | Do not copy the whole dependency diff. Verify the required Newton synchronization API and Warp fix against the target branch's dependencies. Propose only a necessary upstream dependency update before dependent PRs. |
| Backend files contain changes outside the material API, such as the solver-sequence annotation. | Exclude unrelated changes from PR 1.1. Reassess only if an isolated task requires them. |
| The full branch includes unrelated rigid/soft tasks, OSMO, launch fixes, and experiments. | Exclude them. Exclude `dev/`, sweep data, videos, checkpoints, terrain textures, `PR_major.md`, and this plan from the submitted source changes. |

## Test code to include

Test code is part of each implementation PR, not a separate follow-up. Extend the existing test owner where possible. New tests must protect observable behavior; do not add source-text, import-list, or configuration-value tests. Simulation-dependent checks are identified separately from unit tests.

| PR | Test files to add or extend | Required behavior and regression coverage |
| --- | --- | --- |
| Newton 1 | Extend `newton/tests/test_coupled_solver.py` in the Newton repository. | Verify notified particle material and mass updates reach the appropriate child solver models. The regression must fail without the synchronization fix. |
| 1.1 | Add `N/test/physics/test_mpm_material_parameters.py` relative to `develop`; reuse the current branch's relevant tests. | Backend/API tests: validate material ranges, update selected environments, preserve unselected particles, and verify density-to-mass consistency. Keep event sampling tests in the shared backend-event commit used by PRs 2 and 6; do not duplicate Newton's internal synchronization tests. |
| 2 | Extend `N/test/physics/test_mpm_material_parameters.py` from PR 1.1; add `T/test/contrib/test_g1_mpm.py` — new implementation. | Shared backend event tests (also included in PR 6): cover uniform/log-uniform bounds, invalid log ranges, and selected-environment updates through the real material API. Simulation integration test: reuse one fixed-bed scene for full/partial resets and rigid-platform/MPM contact checks. Assert observable reset state, finite observations, and supporting contact forces; stepping without assertions is insufficient. |
| 3 | Extend `R/test/test_entrypoints_common.py` and `R/test/test_entrypoints.py`. | Unit tests: select the requested/latest checkpoint, resolve optional entity information, report missing checkpoints, and forward checkpoint options for train/play. Mock W&B transport only; run the real selection and argument-handling code. |
| 4 | Add `C/test/mdp/test_symmetry.py` — new implementation. | Unit tests: compare mirrored values with independently specified signs/order, verify reflection twice restores input, and cover observation histories, nested groups, action mapping, and input non-mutation. Do not validate correctness through involution alone. |
| 5 | Add `C/test/assets/test_unitree_g1.py` — new implementation. | Standalone asset test: compose the USD, resolve payloads, and check required articulation/collision structure. Check installed-package payload availability without creating a G1 task scene. |
| 6 | Add `T/test/contrib/test_mpm_moving_patch_rewards.py` and `T/test/contrib/test_mpm_moving_patch_reset.py` relative to `develop`. | Reward unit tests: explicit expected values for slope/stance handling and invalid scanner data. Simulation integration test: graph-captured stepping, recycling, partial resets, and terrain-query failure reporting. Also verify selected-environment material updates and density-consistent recycling after reset/patch motion. Keep mixed-terrain cases for PR 7. |
| 7 | Extend `T/test/contrib/test_mpm_moving_patch_reset.py` and `T/test/contrib/test_mpm_moving_patch_rewards.py`. | Mixed-scene integration test: MPM-only randomization, rigid/MPM force routing, fixed margins, and partial resets. Reward unit test: apply the sand-depth clearance offset only to MPM environments. |

Use the contribution guide's test-audit rules before landing these tests. Regression tests must fail on the pre-fix implementation for the intended reason. Avoid redundant device matrices and repeated scene construction.

## Changelog files to add

Follow [Maintaining package changelogs and versions](../../../../../docs/source/refs/contributing.rst#maintaining-package-changelogs-and-versions). These files are part of the corresponding PR's added-file list, in addition to the source and test files above.

- Add an entry under each affected package's `changelog.d/`.
- Use `<slug>.<type>.rst`, where the type is `added`, `changed`, `deprecated`, `removed`, or `fixed`.
- Write concise, past-tense `*` bullets in reStructuredText. Indent continuation lines.
- Add a separate **empty** `<slug>.minor` marker for a new public API. Patch is the default; do not create `.patch` or `.minor.rst` files.
- Use an empty `.major` marker for a breaking change, and prefix the entry with `**Breaking:**` plus migration guidance. No breaking change is intended in this plan.
- Use `.skip` only for packages with exclusively documentation, CI, or test changes and no release entry. It does not replace the feature fragments below.
- Do not edit compiled `docs/CHANGELOG.rst` files or package version numbers; CI owns those updates.

| PR | Fragment files to add | Proposed entry content |
| --- | --- | --- |
| 1.1 | `N/changelog.d/mpm-material-api.added.rst`<br>`N/changelog.d/mpm-material-api.minor` — empty | `* Added MPM particle material getters and setters with density-consistent mass updates.` |
| 2 | `N/changelog.d/mpm-material-randomization.added.rst`<br>`N/changelog.d/mpm-material-randomization.minor` — empty<br>`T/changelog.d/g1-fixed-bed-mpm.added.rst`<br>`T/changelog.d/g1-fixed-bed-mpm.minor` — empty | Newton backend: `* Added uniform/log-uniform MPM material randomization events using the material setter API.`<br>Tasks: `* Added a fixed-bed G1 MPM locomotion task with a rigid approach platform and material randomization.` |
| 3 | `R/changelog.d/wandb-checkpoint-loading.added.rst`<br>`R/changelog.d/wandb-checkpoint-loading.minor` — empty | `* Added W&B checkpoint selection and loading for RSL-RL fine-tuning and playback.` |
| 4 | `C/changelog.d/shared-symmetry.added.rst`<br>`C/changelog.d/shared-symmetry.minor` — empty | `* Added reusable observation and action symmetry augmentation with separate configuration classes.` |
| 5 | `C/changelog.d/unitree-g1-asset.added.rst`<br>`C/changelog.d/unitree-g1-asset.minor` — empty | `* Added the Unitree G1 box-foot asset and configuration for reuse across contributed tasks.` |
| 6 | `T/changelog.d/g1-moving-patch-mpm.added.rst`<br>`T/changelog.d/g1-moving-patch-mpm.minor` — empty<br>Include the same `N/changelog.d/mpm-material-randomization.{added.rst,minor}` files as PR 2 until that shared commit lands. | `* Added G1 moving-patch MPM locomotion with material randomization, density-consistent particle recycling, and OpenGL/RTX particle visualization.` |
| 7 | `T/changelog.d/g1-mixed-rigid-mpm.added.rst`<br>`T/changelog.d/g1-mixed-rigid-mpm.minor` — empty | `* Added mixed rigid/MPM terrain training with terrain-specific contact handling and fixed robot foot collider margins.` |

If the final diff touches another source package, add that package's appropriate fragment too. PRs 4 and 5 touch only the contributed package and have no task-package fragments. PR 8 remains on hold. Newton release notes must follow Newton's own contribution rules, checked when preparing Newton 1; the Isaac Lab fragment format is not assumed to apply there.


## Validation before submission

| PR | Required focused checks before submission |
| --- | --- |
| Newton 1 / Isaac Lab 1.1 | Material updates reach parent and child solver models; density updates particle mass consistently; invalid ranges fail clearly; selected-environment updates leave other environments unchanged. |
| 2 | Validate generic uniform/log-uniform randomization and selected-environment updates; import without the excluded velocity tasks; construct and step on the rigid approach platform and sand; verify full/partial resets, finite observations, material updates, and the packaged robot asset. |
| 3 | Check checkpoint selection/download and both train/play argument handling using existing entrypoint tests. |
| 4 | Check reflection applied twice restores input; observation histories, nested groups, and action mappings are correct; generic augmentation preserves the documented tensor layouts. Task wiring is validated in PRs 2 and 6. |
| 5 | Load the installed contributed asset with all USD references resolved, independently of either task. |
| 6 | Check graph capture, particle recycling, mass/density consistency, terrain query failures, and partial resets. Run GL/RTX particle-rendering smoke checks on the extracted branch. |
| 7 | Check selected-environment randomization and recycled masses; rigid environments receive no MPM updates; verify MPM reaction forces and fixed margins by environment assignment. |

Run formatting, lint, and the contribution guide's relevant checks for each extracted PR. The tables above describe the intended coverage; the published status records completed validation and remaining checks.

Create the test code and changelog files listed above when implementing each PR. Validate fragments against its actual base so prerequisite changes are excluded. For standalone PRs:

```bash
uv run python tools/changelog/cli.py check refs/heads/develop --include-worktree
```

For PRs 2 and 6, use `refs/heads/jnskkmhr/PR_mpm_prerequisites`; for PR 7, use `refs/heads/jnskkmhr/PR_g1_moving_patch_mpm`. Set `ISAACLAB_CHANGELOG_BASE_REF` to the same base used for that PR's validation. After retargeting, use the new base. Refresh the intended base before final validation. Generate task-browser documentation for the final registrations rather than copying the branch's complete documentation diff.
