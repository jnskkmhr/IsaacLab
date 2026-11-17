# Unitree G1 with articulated hands

Converted from Unitree's `robots/g1_description/g1_29dof_with_hand_rev_1_0.xml` at
revision `41dec58c7a3ccb3050243ef16ac2ca74b2b0f9ca`:
https://github.com/unitreerobotics/unitree_ros/blob/41dec58c7a3ccb3050243ef16ac2ca74b2b0f9ca/robots/g1_description/g1_29dof_with_hand_rev_1_0.xml

Upstream BSD-3-Clause license is included. `source.json` records source provenance.
The asset contains 44 rigid bodies and 43 revolute joints: 29 body joints and 14 finger
joints. Source meshes are embedded in `payloads/geometries.usd`; no external STL paths
are required. Each foot uses the box collider from `g1_29dof_rev_1_0_box_foot_improved_collision`: dimensions
0.17 × 0.06 × 0.01 m, centered at (0.035, 0, -0.0304) m in the ankle-roll link frame.
The four source sphere colliders per foot are deactivated in the root USDA. The XML's demo floor is deactivated
in the root USDA: terrain belongs to the task and must not be cloned with each robot.

## Conversion

Download the XML and its referenced `meshes/` directory from the pinned revision.
Run `MjcfConverter(MjcfConverterCfg(asset_path=..., usd_dir=...,
force_usd_conversion=True, self_collision=True, fix_base=False,
robot_type="Humanoid", run_asset_transformer=True, run_multi_physics_conversion=True))`
using the Isaac Lab uv environment. Copy the generated asset directory here.

The importer emitted 34 physics-material bindings outside their referenced instance
scope. Those invalid relationships were removed from `payloads/instances.usda` and
the shared root PhysicsMaterial was bound to the composed Geometry ancestor in the
root USDA. All source geometries use the same default material; this repair preserves
that material while making it resolve through USD composition. Masses, inertias,
and joint limits were not edited. The foot-collider substitution is authored separately
in the root USDA, leaving the converted payloads intact.

## Closed-hand configuration

`UNITREE_G1_29DOF_WITH_HAND_CFG` initializes thumb joints 0/1/2 to 0/0.4/0.8 rad
on the left and 0/-0.4/-0.8 rad on the right. Index and middle joints are -1.3 rad
on the left and +1.3 rad on the right. These positions are inside source joint limits.
Finger drives use simulation holding gains of stiffness 5 and damping 0.1; these are
not measured hardware gains. Effort limits match the XML: 2.45 Nm for thumb joint 0,
1.4 Nm for the other finger joints.

Tasks without finger actions must explicitly set these finger position targets on
reset. The object-push task does this with its `hold_closed_hands` event.
The policy retains 29 body actions. The foot boxes match the previous box-foot asset, but added finger inertia and hand
contacts mean the complete robot is not dynamically identical to the no-hand asset.
