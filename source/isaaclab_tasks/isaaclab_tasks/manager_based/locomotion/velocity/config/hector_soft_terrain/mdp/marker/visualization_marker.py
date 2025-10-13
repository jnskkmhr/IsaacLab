# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg


class FootPlacementVisualizer:
    def __init__(self, prim_path):
        self.prim_path = prim_path
        self.foot_size_x = 0.145
        self.foot_size_y = 0.073
        self.markers_cfg = VisualizationMarkersCfg(
            prim_path=prim_path,
            markers={
                "left_fps": sim_utils.SphereCfg(
                radius=0.02,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.17212, 0.96911)),
                ),
                "right_fps": sim_utils.SphereCfg(
                radius=0.02,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.88305, 0.0, 0.96911)),
                ),
            }
        )
        self.marker = VisualizationMarkers(self.markers_cfg)

    def visualize(self, fps:torch.Tensor):
        """_summary_

        Args:
            fps (torch.Tensor): (num_envs, 2, 3)
        """
        positions = fps.clone()
        num_envs = positions.shape[0]
        indices = torch.arange(self.marker.num_prototypes, device=positions.device).reshape(1, -1).repeat(num_envs, 1) # (num_envs, 4)
        positions = positions.reshape(-1, 3) # (num_envs*2, 3)
        indices = indices.reshape(-1)
        self.marker.visualize(translations=positions, marker_indices=indices)


class SwingFootVisualizer:
    def __init__(self, prim_path):
        self.prim_path = prim_path
        self.markers_cfg = VisualizationMarkersCfg(
            prim_path=prim_path,
            markers={
                "left": sim_utils.SphereCfg(
                radius=0.02,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                ),
                "right": sim_utils.SphereCfg(
                radius=0.02,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                ),
            }
        )
        self.marker = VisualizationMarkers(self.markers_cfg)
    
    def visualize(self, swing_foot_pos:torch.Tensor):
        """_summary_

        Args:
            swing_foot_pos (torch.Tensor): (num_envs, 2, 3)
        """
        num_envs = swing_foot_pos.shape[0]
        indices = torch.arange(self.marker.num_prototypes, device=swing_foot_pos.device).reshape(1, -1).repeat(num_envs, 1) # (num_envs, 2)
        positions = swing_foot_pos.reshape(-1, 3) # (num_envs*2, 3)
        indices = indices.reshape(-1)
        self.marker.visualize(translations=positions, marker_indices=indices)


class PositionTrajectoryVisualizer:
    def __init__(self, prim_path:str, color:tuple=(1.0, 0.0, 0.0)):
        self.prim_path = prim_path
        self.markers_cfg = VisualizationMarkersCfg(
            prim_path=prim_path,
            markers={
                "pos": sim_utils.SphereCfg(
                radius=0.01,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
                ),
            }
        )
        self.marker = VisualizationMarkers(self.markers_cfg)
    
    def visualize(self, position_trajectory:torch.Tensor):
        """_summary_

        Args:
            position_trajectory (torch.Tensor): (num_envs, 10, 3)
        """
        num_envs = position_trajectory.shape[0]
        indices = torch.arange(self.marker.num_prototypes, device=position_trajectory.device).reshape(1, -1).repeat(num_envs, 10) # (num_envs, 10)
        positions = position_trajectory.reshape(-1, 3) # (num_envs*10, 3)
        indices = indices.reshape(-1)
        self.marker.visualize(translations=positions, marker_indices=indices)