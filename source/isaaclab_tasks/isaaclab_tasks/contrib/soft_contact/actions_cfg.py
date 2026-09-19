# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING
from typing import Literal

import isaaclab.sim as sim_utils
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.utils.configclass import configclass

from isaaclab_assets import ISAACLAB_ASSETS_DATA_DIR

from . import physics_callback_actions
from ._impl.collider import ColliderCfg

CONTACT_MARKER_CFG = VisualizationMarkersCfg(
    markers={
        "contact": sim_utils.SphereCfg(
            radius=0.02,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        ),
    },
)

BLUE_ARROW_Z_MARKER_CFG = VisualizationMarkersCfg(
    markers={
        "arrow": sim_utils.UsdFileCfg(
            usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/prop/arrow_z.usd",
            scale=(0.1, 0.1, 1.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
        )
    }
)


@configclass
class PhysicsCallbackActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = physics_callback_actions.PhysicsCallbackAction
    body_names: list[str] = MISSING
    """List of joint names or regex expressions that the action will be mapped to."""
    backend: Literal["2D", "3D", "2D-warp", "3D-warp", "spring-damper", "cone-drft", "cone-drft-multipoint"] = "3D-warp"
    """The RFT backend to use.

    The ``-warp`` variants are the Warp implementations; ``2D`` and ``3D`` are the reference
    PyTorch implementations.
    """
    disable: bool = False
    """Whether to disable this action term."""
    enable_ema_filter: bool = True
    """Whether to enable an exponential moving average filter on the input actions."""
    contact_threshold: float = 10.0
    """Threshold for contact detection (N)."""
    intruder_geometry_cfg: ColliderCfg = MISSING
    """Configuration for the intruder geometry used in soft contact modeling."""
    contact_data_history_length: int = 3
    """Length of the contact data history."""
    history_logging_decimation: int = 1
    """Decimation factor for logging contact data."""

    contact_visualizer_cfg: VisualizationMarkersCfg = CONTACT_MARKER_CFG.replace(
        prim_path="/Visuals/Contact/contact",
    )
    """Configuration for the contact visualization markers."""

    contact_force_visualizer_cfg: VisualizationMarkersCfg = BLUE_ARROW_Z_MARKER_CFG.replace(
        prim_path="/Visuals/Contact/force",
    )
    """Configuration for the contact force visualization markers."""
    contact_vis_scale: float = 100.0
    """Maximum force to visualize (N)."""
    contact_vis_force_threshold: float = 40.0
    """Threshold for contact force visualization (N)."""
    contact_force_visualizer_cfg.markers["arrow"].scale = (0.3, 0.3, 0.3)
    """Scale of the contact force visualization marker."""
