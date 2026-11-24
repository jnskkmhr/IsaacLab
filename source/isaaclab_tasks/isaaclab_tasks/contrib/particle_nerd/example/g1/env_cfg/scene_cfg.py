# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Scene configuration for the `g1` humanoid granular locomotion task (`3.6 m x 1.6 m x 0.30 m` sand bed)."""

from copy import deepcopy

import numpy as np
from isaaclab_newton.assets import MPMObjectCfg
from isaaclab_newton.sim.schemas import NewtonCollisionCfg
from isaaclab_newton.sim.spawners.mpm import MPMParticleMaterialCfg, MPMPointsCfg

import isaaclab.sim as sim
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass, replace
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_tasks.contrib.g1_29dof_mpm.env_cfg.scene_cfg import MPM_VISUAL_COLOR

from isaaclab_assets import UNITREE_G1_29DOF_BOX_FOOT_CFG

VOXEL_SIZE = 0.04
FOOT_MARGIN = 0.02
SPACING = VOXEL_SIZE / 1.25
SAND_DENSITY = 1600.0
PARTICLE_JITTER = 0.20 * SPACING
PARTICLE_SEED = 42
BED_SIZE = (3.6, 1.6, 0.30)


def sand_grid_positions(size: tuple[float, float, float] = BED_SIZE, spacing: float = SPACING) -> list[list[float]]:
    """Build jittered particle positions of the sand bed (`N=50,400`)."""
    length, width, depth = size
    axes = [
        np.arange(-length / 2 + spacing / 2, length / 2, spacing),
        np.arange(-width / 2 + spacing / 2, width / 2, spacing),
        np.arange(-depth + spacing / 2, 0.2, spacing),
    ]
    points = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    points = points[points[:, 2] < 0.0]
    rng = np.random.default_rng(PARTICLE_SEED)
    lower = np.maximum(points - PARTICLE_JITTER, (-length / 2, -width / 2, -depth))
    upper = np.minimum(points + PARTICLE_JITTER, (length / 2, width / 2, 0.0))
    points[:, :2] = rng.uniform(lower[:, :2], upper[:, :2])
    points[:, 2] = rng.uniform(lower[:, 2], upper[:, 2])
    return points.astype(np.float32).tolist()


def _tray_wall_cfg(name: str, size: tuple[float, float, float], pos: tuple[float, float, float]) -> AssetBaseCfg:
    return AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/" + name,
        spawn=sim.CuboidCfg(
            size=size,
            collision_props=NewtonCollisionCfg(contact_margin=0.016),
            visual_material=sim.PreviewSurfaceCfg(diffuse_color=(0.25, 0.30, 0.33)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=pos),
    )


@configclass
class G1SceneCfg(InteractiveSceneCfg):
    """Scene for the `g1` humanoid granular locomotion task."""

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    robot = replace(UNITREE_G1_29DOF_BOX_FOOT_CFG, prim_path="{ENV_REGEX_NS}/Robot")
    robot.spawn.collision_props = {
        r"/.*_ankle_roll_link/.*": [NewtonCollisionCfg(contact_margin=FOOT_MARGIN, contact_gap=0.0)]
    }
    robot.init_state.pos = (-1.2, 0.0, 0.76)

    sand = MPMObjectCfg(
        prim_path="{ENV_REGEX_NS}/Sand",
        spawn=MPMPointsCfg(
            positions=sand_grid_positions(),
            radius=0.5 * SPACING,
            mass=SAND_DENSITY * (SPACING**3),
            visual_color=MPM_VISUAL_COLOR,
            material=MPMParticleMaterialCfg(
                density=SAND_DENSITY,
                young_modulus=1.0e15,
                poisson_ratio=0.30,
                friction=0.48,
                yield_pressure=1.0e15,
                tensile_yield_ratio=0.0,
                yield_stress=0.0,
                hardening=0.0,
                dilatancy=0.0,
                viscosity=0.0,
            ),
        ),
    )

    def __post_init__(self) -> None:
        length, width, depth = BED_SIZE
        wall_top = 0.20
        wall_height = depth + wall_top
        wall_center = (wall_top - depth) / 2.0

        self.floor = _tray_wall_cfg("Floor", (length + 0.2, width + 0.2, 0.20), (0.0, 0.0, -depth - 0.10))
        self.wall_x0 = _tray_wall_cfg(
            "WallX0", (0.10, width + 0.2, wall_height), (-length / 2 - 0.05, 0.0, wall_center)
        )
        self.wall_x1 = _tray_wall_cfg("WallX1", (0.10, width + 0.2, wall_height), (length / 2 + 0.05, 0.0, wall_center))
        self.wall_y0 = _tray_wall_cfg("WallY0", (length, 0.10, wall_height), (0.0, -width / 2 - 0.05, wall_center))
        self.wall_y1 = _tray_wall_cfg("WallY1", (length, 0.10, wall_height), (0.0, width / 2 + 0.05, wall_center))

        for name in ("floor", "wall_x0", "wall_x1", "wall_y0", "wall_y1"):
            mirror = deepcopy(getattr(self, name))
            mirror.prim_path = mirror.prim_path.replace("}/", "}/Rigid")
            mirror.spawn.visible = False
            setattr(self, "rigid_" + name, mirror)
