# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""USD models of simple towed objects, written to a cache directory on demand.

The repository does not track USD files, so objects made of a few primitives are described here and written
out when a task configuration asks for them.
"""

import hashlib
import os
import pathlib
import tempfile


def _cylinder_y(name: str, y: float, radius: float, length: float) -> str:
    """One collider cylinder with its axis along y, centered at ``(0, y, 0)`` [m], bound to the body's material."""
    return (
        f'\n    def Cylinder "{name}" (\n'
        '        prepend apiSchemas = ["PhysicsCollisionAPI", "MaterialBindingAPI"]\n    )\n    {\n'
        "        rel material:binding:physics = </Wheelset/PhysicsMaterial>\n"
        '        uniform token axis = "Y"\n'
        f"        double radius = {radius:.6f}\n        double height = {length:.6f}\n"
        f"        double3 xformOp:translate = (0, {y:.6f}, 0)\n"
        '        uniform token[] xformOpOrder = ["xformOp:translate"]\n    }\n'
    )


def wheelset_usd(wheel_radius: float, wheel_width: float, track: float, friction: float) -> str:
    """Path of a wheelset model: two wheels joined by an axle, one rigid body that rolls as a whole.

    It stands in for a cart whose load sits on the axle: the wheels' sinkage sets the rolling resistance. The
    origin is the middle of the axle and the wheels roll about the body's y axis.

    Args:
        wheel_radius: Radius of each wheel [m].
        wheel_width: Width of each wheel [m].
        track: Distance between the wheel centers [m].
        friction: Friction coefficient of every shape [-].
    """
    text = (
        '#usda 1.0\n(\n    defaultPrim = "Wheelset"\n    metersPerUnit = 1\n    upAxis = "Z"\n)\n\n'
        'def Xform "Wheelset" (\n    prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]\n)\n{\n'
        "    float physics:mass = 1\n"
        '\n    def Material "PhysicsMaterial" (\n        prepend apiSchemas = ["PhysicsMaterialAPI"]\n    )\n    {\n'
        f"        float physics:staticFriction = {friction:.3f}\n"
        f"        float physics:dynamicFriction = {friction:.3f}\n"
        "        float physics:restitution = 0\n    }\n"
    )
    text += _cylinder_y("wheel_left", 0.5 * track, wheel_radius, wheel_width)
    text += _cylinder_y("wheel_right", -0.5 * track, wheel_radius, wheel_width)
    text += _cylinder_y("axle", 0.0, 0.2 * wheel_radius, track - wheel_width)
    text += "}\n"
    directory = pathlib.Path(tempfile.gettempdir()) / "isaaclab_push_assets"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"wheelset_{hashlib.sha1(text.encode()).hexdigest()[:10]}.usda"
    if not path.exists():
        temporary = path.with_suffix(f".{os.getpid()}.tmp")
        temporary.write_text(text)
        os.replace(temporary, path)
    return str(path)
