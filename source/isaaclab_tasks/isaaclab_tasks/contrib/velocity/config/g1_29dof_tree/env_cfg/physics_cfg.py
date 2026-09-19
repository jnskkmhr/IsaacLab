# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Coupled MJWarp + VBD physics for the G1 locomotion tasks among trees.

The rigid entry integrates the articulation with MJWarp, the cable entry integrates the logs or
rods with VBD, and a lagged proxy mapping exposes the robot's links to the cable solver as
colliders. The whole robot is proxied rather than a selected set of links: a humanoid walking into
an obstacle meets it with its feet, but it meets the ground with a knee or a hand as soon as the
obstacle trips it.

Static shapes -- the terrain plane, and the bar task's pillars -- are owned by the rigid entry,
since a shape belongs to at most one entry. The pile task's logs therefore need a floor of their
own: the kinematic ``PileGround`` box of :mod:`.scene_cfg` is owned by the cable entry and sunk
just below the terrain, which keeps the robot's proxies from resting on it and being carried by
two floors at once. The bar task's rods hang from world anchors and need no floor.
"""

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonCollisionPipelineCfg, NewtonShapeCfg, VBDSolverCfg

from isaaclab.utils.configclass import configclass

from isaaclab_contrib.coupling import CouplerEntryCfg, CouplerProxyCfg, CouplerProxyMappingCfg

from isaaclab_tasks.utils import PresetCfg

RIGID_ENTRY = "robot"
"""Name of the MJWarp coupler entry."""

CABLE_ENTRY = "cable"
"""Name of the VBD coupler entry."""

ROBOT_BODIES = [r"/World/envs/env_.*/Robot"]
"""Bodies owned by the rigid entry; the selector covers every descendant link."""

PILE_BODIES = [r"/World/envs/env_.*/Log_.*", r"/World/envs/env_.*/PileGround"]
"""Bodies owned by the cable entry of the pile task: every log segment and the floor of the piles."""

BAR_BODIES = [r"/World/envs/env_.*/Bar_.*"]
"""Bodies owned by the cable entry of the bar task: every rod segment."""


def _coupled_cfg(cable_bodies: list[str]) -> NewtonCfg:
    """Build the coupled backend for a scene whose cables are selected by ``cable_bodies``.

    Args:
        cable_bodies: Body selectors owned by the VBD entry.

    Returns:
        The Newton configuration.
    """
    return NewtonCfg(
        solver_cfg=CouplerProxyCfg(
            entries=[
                CouplerEntryCfg(
                    name=RIGID_ENTRY,
                    solver_cfg=MJWarpSolverCfg(
                        njmax=1000,
                        nconmax=300,
                        cone="pyramidal",
                        impratio=1.0,
                        integrator="implicitfast",
                        use_mujoco_contacts=False,
                    ),
                    bodies=ROBOT_BODIES,
                    # picks up the ground plane and any other static collider, which have no rigid
                    # body to name
                    include_static_shapes=True,
                ),
                CouplerEntryCfg(
                    name=CABLE_ENTRY,
                    solver_cfg=VBDSolverCfg(iterations=10),
                    bodies=cable_bodies,
                    include_static_shapes=False,
                    # Refine the cable integration between coupled exchanges: the cables are two
                    # orders stiffer than the contacts that drive them, and a pile is resolved
                    # through cable-on-cable contact.
                    substeps=2,
                ),
            ],
            proxies=[
                CouplerProxyMappingCfg(
                    source=RIGID_ENTRY,
                    destination=CABLE_ENTRY,
                    bodies=ROBOT_BODIES,  # type: ignore
                    mode="lagged",
                    collide_interval=1,
                )
            ],
            iterations=1,
        ),
        collision_cfg=NewtonCollisionPipelineCfg(max_triangle_pairs=2_500_000),
        # Newton's default shape stiffness (ke 2.5e3, kd 1e2) is two orders of magnitude too
        # compliant for a 32 kg humanoid: `use_mujoco_contacts=False` routes it through
        # `convert_solref`, so the ground answers a footfall like a mattress. These are the values
        # the rigid and soft-contact G1 tasks walk on.
        default_shape_cfg=NewtonShapeCfg(margin=0.0, ke=160000.0, kd=1100.0),
        num_substeps=4,
    )


@configclass
class G1TreePhysicsCfg(PresetCfg):
    """Backend presets for the G1 29-DoF locomotion task among log piles.

    Only the coupled Newton backend is offered: the logs are Newton cables, which no other
    backend simulates.
    """

    newton_mjwarp_vbd_proxy = _coupled_cfg(PILE_BODIES)

    default = newton_mjwarp_vbd_proxy


@configclass
class G1BarPhysicsCfg(PresetCfg):
    """Backend presets for the G1 29-DoF locomotion task among bars.

    Only the coupled Newton backend is offered: the rods are Newton cables, which no other
    backend simulates.
    """

    newton_mjwarp_vbd_proxy = _coupled_cfg(BAR_BODIES)

    default = newton_mjwarp_vbd_proxy
