# Term-driven mirror augmentation

`vel_mdp.symmetry` reflects locomotion observations and raw actions across the XZ
plane (left/right). It reads active term names, dimensions, concatenation settings,
and history lengths from initialized managers. It does not assume a `policy` group,
a fixed joint count, a clock term, or a particular observation order.

## Configure observation terms

Use `MirrorObservationTermCfg` wherever you would use `ObservationTermCfg`:

```python
from isaaclab.envs import mdp
from isaaclab_tasks.contrib.velocity.config import vel_mdp

base_velocity = vel_mdp.MirrorObservationTermCfg(
    func=mdp.base_lin_vel,
    mirror=vel_mdp.mirror_vec3,
)
angular_velocity = vel_mdp.MirrorObservationTermCfg(
    func=mdp.base_ang_vel,
    mirror=vel_mdp.mirror_vec3,
    mirror_params={"axial": True},
)
orientation = vel_mdp.MirrorObservationTermCfg(
    func=mdp.root_quat_w,
    mirror=vel_mdp.mirror_quat,
)
```

Each `mirror(data, **mirror_params)` callback receives the stored observation
**after its observation scale, clipping, and modifiers**. It must return a tensor
with the same shape, dtype, and device. It must handle arbitrary leading batch
axes. Flattened histories are temporarily reshaped to
`(*batch, history_length, features)`; unflattened terms retain their shape.
For flattened images or grids, pass their spatial shape as a mirror parameter
and restore that shape inside the callback. Do not read live environment state:
RSL-RL also calls these functions on historical rollout minibatches.

Built-in rules:

| Function | Reflection |
| --- | --- |
| `mirror_vec3` | Polar vectors such as velocity or gravity: `[x, -y, z]` |
| `mirror_vec3(..., axial=True)` | Angular velocity or torque: `[-x, y, -z]` |
| `mirror_quat` | XYZW orientation: `[-x, y, -z, w]` |
| `mirror_joints` | Explicit joint permutation and optional signs |
| `mirror_identity` | An explicitly invariant observation or action |

Supply a custom function for compound commands, contacts, images, or other
specialized terms. Use `mirror_identity` only when leaving a term unchanged is
appropriate for your task. A missing mirror rule raises an error when that term
is augmented, rather than silently producing an inconsistent sample.

## Configure action terms

For joint position actions, use `MirrorJointPositionActionCfg`:

```python
joint_pos = vel_mdp.MirrorJointPositionActionCfg(
    asset_name="robot",
    joint_names=["left_joint", "right_joint"],
    preserve_order=True,
    scale=0.25,
    mirror=vel_mdp.mirror_joints,
    mirror_params={"permutation": [1, 0], "signs": [-1, -1]},
)
```

The signs depend on the robot's joint axes. Indices must describe the **actual
term order**, not an assumed PhysX or Newton articulation order. Both the
permutation and its signs must restore the original data when applied twice.

For other action types, combine the concrete action config with
`MirrorActionTermCfg`, following the same pattern as `MirrorJointPositionActionCfg`:

```python
from isaaclab.utils.configclass import configclass

@configclass
class MirroredCustomActionCfg(CustomActionCfg, vel_mdp.MirrorActionTermCfg):
    pass
```

Rules operate on raw policy actions, before the action manager applies scales,
offsets, or clipping. Those settings, nominal joint poses, and observation
preprocessing must themselves respect the intended physical symmetry. A custom
mirror must account for asymmetric normalization or offsets when necessary.

## Connect RSL-RL

```python
from isaaclab_rl.rsl_rl import RslRlSymmetryCfg

symmetry_cfg = RslRlSymmetryCfg(
    use_data_augmentation=True,
    data_augmentation_func=vel_mdp.compute_mirrored_states,
)
```

Assign this config to your PPO algorithm's `symmetry_cfg` field. The callback
returns originals followed by mirrored samples along the first batch dimension.
It transforms **every supplied observation group**, including critic inputs,
and supports actions-only or observations-only calls. Concatenated groups and
nested TensorDict groups are supported. Input tensors are not modified.

`MirrorAugmentation(env)` is also available for direct use with initialized
managers. The RSL-RL callback caches one instance per environment; rebuild the
environment after changing the manager configuration.

## G1 integration and migration

The G1 rigid task declares reflection rules on its policy, critic, privileged,
logging, and action terms. Its existing `compute_symmetric_states` callback now
delegates to the generic implementation, so the existing `WithSymmetry` runner
configs retain their entry point. This change does not enable symmetry on the
default runner.

G1 rules swap left/right joints by their explicitly configured names and negate
roll/yaw axes. They also reflect the seven-channel velocity/heading command,
swap paired foot signals, and reflect height scans using the supplied grid
pattern. Keep `mirror_params` synchronized when changing joint selectors, action
order, or scan patterns. The current joint selectors use `preserve_order=True`.

For custom observation/action terms added to these tasks, declare a mirror
callback (or explicitly use `mirror_identity`) before enabling augmentation.
Existing checkpoints retain their input dimensions, but resumed symmetry-enabled
training will use corrected augmented targets: the previous G1 implementation
produced zero mirrored joint actions and left critic/privileged groups unchanged.
This reflection support does not change world-frame heading observations into
relative heading observations.
