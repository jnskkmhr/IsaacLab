from dataclasses import MISSING
import math
import torch
from isaaclab.utils import configclass
from isaaclab.utils.math import matrix_from_quat, euler_xyz_from_quat, matrix_from_euler
from typing import Tuple

from .soft_contact_model_data import SoftContactData

"""
2D RFT material parameters.
"""

@configclass
class MaterialCfg:
    A00: float = MISSING # type: ignore
    A10: float = MISSING # type: ignore
    B11: float = MISSING # type: ignore
    B01: float = MISSING # type: ignore
    B_11: float = MISSING # type: ignore
    C11: float = MISSING # type: ignore
    C01: float = MISSING # type: ignore
    C_11: float = MISSING # type: ignore
    D10: float = MISSING # type: ignore
    stiffness: float = MISSING # type: ignore # scaling factor applied to resulting force per area (alpha)
    static_friction_coef: float = MISSING # type: ignore
    dynamic_friction_coef: float = MISSING # type: ignore

@configclass
class PoppySeedLPCfg(MaterialCfg):
    A00: float = 0.051
    A10: float = 0.047
    B11: float = 0.053
    B01: float = 0.083
    B_11: float = 0.020
    C11: float = -0.026
    C01: float = 0.057
    C_11: float = 0.0
    D10: float = 0.025
    stiffness: float = 1.0
    static_friction_coef: float = 1.0
    dynamic_friction_coef: float = 0.5

@configclass
class PoppySeedCPCfg(MaterialCfg):
    A00: float = 0.094
    A10: float = 0.092
    B11: float = 0.092
    B01: float = 0.151
    B_11: float = 0.035
    C11: float = -0.039
    C01: float = 0.086
    C_11: float = 0.018
    D10: float = 0.046
    stiffness: float = 10.0
    static_friction_coef: float = 1.0
    dynamic_friction_coef: float = 0.5


"""
Intruder geometry configuration.
"""

@configclass
class IntruderGeometryCfg:
    """
    Intruder surface is approximated as rectangle.
    contact_edge_*: tuple of lower and upper bounds of corner from intruder link origin.
    """
    # contact_edge_x: Tuple[float, float] = (-0.07, 0.14) # length in x direction (m)
    # contact_edge_x: Tuple[float, float] = (-0.065, 0.141) # length in x direction (m)
    contact_edge_x: Tuple[float, float] = (-1.3*0.065, 1.3*0.141) # length in x direction (m)
    contact_edge_y: Tuple[float, float] = (-0.0368, 0.0368) # length in y direction (m)
    contact_edge_z: Tuple[float, float] = (-0.039, 0.0) # length in z direction (m)
    num_contact_points:int = 20*20

"""
2D RFT with dynamic inertial modification (DRFT) + exponential moving average filtering.
"""

class RFT_EMF:
    def __init__(self, 
                 num_envs: int,
                 num_bodies: int, 
                 device: torch.device,
                 dt:float, 
                 history_length: int = 3,
                 material_cfg: MaterialCfg=PoppySeedCPCfg(), 
                 intruder_cfg: IntruderGeometryCfg=IntruderGeometryCfg(),
                 ):
        """
        Resistive Force Theory based soft terrain contact solver.
        https://www.science.org/doi/10.1126/science.1229163
        
        Args: 
            num_envs: number of parallel environments
            num_bodies: number of bodies using soft contact model per env
            device: torch device
            dt: simulation time step
            history_length: length of history for force tracking
            material_cfg: material configuration
            intruder_cfg: intruder geometry configuration
        """

        self.cfg = material_cfg
        self.num_envs = num_envs
        self.num_bodies = num_bodies
        self.num_contact_points = intruder_cfg.num_contact_points
        self.device = device
        self.dt = dt
        
        self.contact_edge_x = intruder_cfg.contact_edge_x
        self.contact_edge_y = intruder_cfg.contact_edge_y
        self.contact_edge_z = intruder_cfg.contact_edge_z
        self.foot_depth = self.contact_edge_z[1] - self.contact_edge_z[0]
        self.surface_area = (self.contact_edge_x[1]-self.contact_edge_x[0])*(self.contact_edge_y[1]-self.contact_edge_y[0])
        
        self.c_r = 0.01 # 100/f (e.g. f=2000hz -> 0.05)
        self.static_friction_coef = material_cfg.static_friction_coef
        self.dynamic_friction_coef = material_cfg.dynamic_friction_coef
        self.history_length = history_length

        self.create_internal_tensors()
        self.create_contact_points()
        
        self._data: SoftContactData = SoftContactData()
        self.initialize_data()
    
    """
    Initialization.
    """
    
    def create_contact_points(self)->None:
        """
        Create contact points in contact surface in root body frame (e.g. foot link frame). 
        """
        num_contact_point_side = int(math.sqrt(self.num_contact_points))
        assert num_contact_point_side**2 == self.num_contact_points, "num_contact_points must be a perfect square"
        contact_point_offset_x = torch.linspace(self.contact_edge_x[0], self.contact_edge_x[1], num_contact_point_side, device=self.device)
        contact_point_offset_y = torch.linspace(self.contact_edge_y[0], self.contact_edge_y[1], num_contact_point_side, device=self.device)
        contact_point_offset_y, contact_point_offset_x = torch.meshgrid(contact_point_offset_y, contact_point_offset_x, indexing='ij')
        contact_point_offset = torch.stack((
            contact_point_offset_x.flatten(), 
            contact_point_offset_y.flatten(), 
            -self.foot_depth * torch.ones_like(contact_point_offset_x).flatten()
            ), dim=-1)
        contact_point_offset = contact_point_offset.unsqueeze(0).unsqueeze(1).repeat(self.num_envs, self.num_bodies, 1, 1) # (num_envs, num_bodies, num_contact_points, 3)
        self.contact_point_offset[:, :, :, :] = contact_point_offset
        
    def create_internal_tensors(self):
        """
        Create buffer tensors.
        """
        self.body_pos = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device)
        self.body_quat = torch.zeros((self.num_envs, self.num_bodies, 4), device=self.device)
        self.body_rot_mat = torch.zeros((self.num_envs, self.num_bodies, 3, 3), device=self.device)
        self.body_rot_mat_roll_yaw = torch.zeros((self.num_envs, self.num_bodies, 3, 3), device=self.device)
        self.body_lin_vel = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device)
        self.body_ang_vel = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device)
        
        self.is_contact = torch.zeros((self.num_envs, self.num_bodies, self.num_contact_points), dtype=torch.bool, device=self.device)

        self.contact_point_offset = torch.zeros((self.num_envs, self.num_bodies, self.num_contact_points, 3), device=self.device)
        self.contact_point_pos = torch.zeros((self.num_envs, self.num_bodies, self.num_contact_points, 3), device=self.device)
        self.contact_point_euler = torch.zeros((self.num_envs, self.num_bodies, self.num_contact_points, 3), device=self.device)
        self.contact_point_lin_vel = torch.zeros((self.num_envs, self.num_bodies, self.num_contact_points, 3), device=self.device)
        self.contact_point_lin_vel_b = torch.zeros((self.num_envs, self.num_bodies, self.num_contact_points, 3), device=self.device)
        self.contact_point_intrusion_angle = torch.zeros((self.num_envs, self.num_bodies, self.num_contact_points), device=self.device)
        self.contact_point_lin_vel_prev = torch.zeros((self.num_envs, self.num_bodies, self.num_contact_points, 3), device=self.device)
        
        self.contact_point_force = torch.zeros((self.num_envs, self.num_bodies, self.num_contact_points, 3), device=self.device)
        self.contact_point_torque = torch.zeros((self.num_envs, self.num_bodies, self.num_contact_points, 3), device=self.device)
        self.contact_point_filtered_force = torch.zeros((self.num_envs, self.num_bodies, self.num_contact_points, 3), device=self.device)
        
        self.vt_filtered = torch.zeros((self.num_envs, self.num_bodies*self.num_contact_points), device=self.device)
        self.vt_unit_filtered = torch.zeros((self.num_envs, self.num_bodies*self.num_contact_points, 2), device=self.device)
        
        # lumped force and torque
        self.contact_force = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device)
        self.contact_torque = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device)
        
        self.force_gm = torch.zeros((self.num_envs, self.num_bodies*self.num_contact_points), device=self.device)
        self.force_ema = torch.zeros((self.num_envs, self.num_bodies*self.num_contact_points), device=self.device)
        self.tau_r = torch.zeros((self.num_envs, self.num_bodies*self.num_contact_points), device=self.device)
        
        self.stiffness = self.cfg.stiffness * torch.ones(self.num_envs, device=self.device)
        self.soft_level = torch.zeros(self.num_envs, device=self.device)
        self.max_level = 10
        
        self._timestamp = torch.zeros(self.num_envs, device=self.device)
        self._timestamp_last_update = torch.zeros(self.num_envs, device=self.device)
        
    def initialize_data(self):
        """
        Initialize soft contact data.
        """
        self._data.net_forces_w = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device)
        self._data.net_forces_w_history = torch.zeros((self.num_envs, self.history_length, self.num_bodies, 3), device=self.device)
        self._data.force_matrix_w = torch.zeros((self.num_envs, self.num_bodies, self.num_contact_points, 3), device=self.device)
        self._data.force_matrix_w_history = torch.zeros((self.num_envs, self.history_length, self.num_bodies, self.num_contact_points, 3), device=self.device)
        self._data.last_air_time = torch.zeros((self.num_envs, self.num_bodies), device=self.device)
        self._data.current_air_time = torch.zeros((self.num_envs, self.num_bodies), device=self.device)
        self._data.last_contact_time = torch.zeros((self.num_envs, self.num_bodies), device=self.device)
        self._data.current_contact_time = torch.zeros((self.num_envs, self.num_bodies), device=self.device)
        
        
    """
    operations.
    """
    
    def update(self, body_pos:torch.Tensor, body_quat:torch.Tensor, body_lin_vel:torch.Tensor, body_ang_vel:torch.Tensor):
        """
        Update soft contact model.
        
        Args:
            body_pos: intruder position. (num_envs, num_bodies, 3)
            body_quat: intruder orientation in quaternion form. (num_envs, num_bodies, 4)
            body_lin_vel: intruder linear velocity wrt global frame. (num_envs, num_bodies, 3)
            body_ang_vel: intruder angular velocity wrt global frame. (num_envs, num_bodies, 3)
        """
        
        self._timestamp += self.dt
        self.body_pos[:, :, :] = body_pos
        self.body_quat[:, :, :] = body_quat
        self.body_rot_mat[:, :, :, :] = matrix_from_quat(body_quat.view(-1, 4)).view(self.num_envs, self.num_bodies, 3, 3)
        
        roll, pitch, yaw = euler_xyz_from_quat(body_quat.view(-1, 4))
        self.body_rot_mat_roll_yaw[:, :, :, :] = \
            matrix_from_euler(
                torch.stack(
                (roll, torch.zeros_like(pitch), yaw), dim=-1),
                # (torch.zeros_like(roll), torch.zeros_like(pitch), yaw), dim=-1),
                convention="XYZ").view(self.num_envs, self.num_bodies, 3, 3)

        self.body_lin_vel[:, :, :] = body_lin_vel
        self.body_ang_vel[:, :, :] = body_ang_vel

        self._process_contact_points()
        self._compute_force()
        self._update_data(torch.arange(self.num_envs, device=self.device))
        self._timestamp_last_update[:] = self._timestamp[:]
        
        # print("contact force:", self.contact_force)
        # print("current air time:")
        # print(self.data.current_air_time)
        # print("current contact time:")
        # print(self.data.current_contact_time)
        
    def update_ground_stiffness(self, env_ids:torch.Tensor, move_up: torch.Tensor, move_down:torch.Tensor):
        """
        Update ground stiffness (N/m) for each env.
        Implementation is similar to terrain curriculum used in terrain importer class.
        
        Args:
            env_ids: tensor of env ids to update
            move_up: tensor of env ids to increase softness level (len(env_ids),)
            move_down: tensor of env ids to decrease softness level (len(env_ids),)
        """
        self.soft_level[env_ids] += 1 * move_up - 1 * move_down
        self.soft_level[env_ids] = torch.where(
            self.soft_level[env_ids] >= self.max_level,
            torch.randint_like(self.soft_level[env_ids], self.max_level),
            torch.clip(self.soft_level[env_ids], 0),
        )
        self.stiffness = self.max_level - self.soft_level

    
    """
    properties.
    """
    
    @property
    def data(self)-> SoftContactData:
        return self._data
    
    @property
    def contact_wrench(self)->torch.Tensor:
        return torch.cat((self.contact_force, self.contact_torque), dim=-1) # (num_envs, num_bodies, 6)
    
    @property
    def contact_wrench_b(self)->torch.Tensor:
        contact_force_b = (self.body_rot_mat.transpose(-1, -2) @ self.contact_force.unsqueeze(-1)).squeeze(-1) # (num_envs, num_bodies, 3)
        contact_torque_b = (self.body_rot_mat.transpose(-1, -2) @ self.contact_torque.unsqueeze(-1)).squeeze(-1) # (num_envs, num_bodies, 3)
        return torch.cat((contact_force_b, contact_torque_b), dim=-1) # (num_envs, num_bodies, 6)
    
    @property
    def contact_point_wrench(self)->torch.Tensor:
        return torch.cat((self.contact_point_force, self.contact_point_torque), dim=-1) # (num_envs, num_bodies, num_contact_points, 6)
        

    
    """
    helper functions.
    """
    
    def _update_data(self, env_ids:torch.Tensor)->None:
        """
        Update soft contact data.
        
        Args:
            env_ids: tensor of env ids to update
        """
        self._data.net_forces_w[env_ids, :, :] = self.contact_force[env_ids, :, :] # type: ignore
        self._data.force_matrix_w[env_ids, :, :, :] = self.contact_point_force[env_ids, :, :, :] # type: ignore
        if self.history_length > 0:
            self._data.net_forces_w_history[env_ids] = self._data.net_forces_w_history[env_ids].roll(shifts=1, dims=1) # type: ignore
            self._data.net_forces_w_history[env_ids, 0] = self._data.net_forces_w[env_ids] # type: ignore
            
            self._data.force_matrix_w_history[env_ids] = self._data.force_matrix_w_history[env_ids].roll(shifts=1, dims=1) # type: ignore
            self._data.force_matrix_w_history[env_ids, 0] = self._data.force_matrix_w[env_ids] # type: ignore
            
        # track air time (see contact sensor class)
        elapsed_time = self._timestamp[env_ids] - self._timestamp_last_update[env_ids]
        is_contact = torch.norm(self._data.net_forces_w[env_ids, :, :], dim=-1) > 5.0 # type: ignore
        is_first_contact = (self._data.current_air_time[env_ids] > 0) * is_contact # type: ignore
        is_first_detached = (self._data.current_contact_time[env_ids] > 0) * ~is_contact # type: ignore
        # -- update the last contact time if body has just become in contact
        self._data.last_air_time[env_ids] = torch.where( # type: ignore
            is_first_contact,
            self._data.current_air_time[env_ids] + elapsed_time.unsqueeze(-1), # type: ignore
            self._data.last_air_time[env_ids], # type: ignore
        )
        # -- increment time for bodies that are not in contact
        self._data.current_air_time[env_ids] = torch.where( # type: ignore
            ~is_contact, self._data.current_air_time[env_ids] + elapsed_time.unsqueeze(-1), 0.0 # type: ignore
        )
        # -- update the last contact time if body has just detached
        self._data.last_contact_time[env_ids] = torch.where( # type: ignore
            is_first_detached,
            self._data.current_contact_time[env_ids] + elapsed_time.unsqueeze(-1), # type: ignore
            self._data.last_contact_time[env_ids], # type: ignore
        )
        # -- increment time for bodies that are in contact
        self._data.current_contact_time[env_ids] = torch.where( # type: ignore
            is_contact, self._data.current_contact_time[env_ids] + elapsed_time.unsqueeze(-1), 0.0 # type: ignore
        )
        
        
    def _process_contact_points(self)->None:
        """
        Update contact points' kinematic states.
        """
        R = self.body_rot_mat.unsqueeze(2)
        p = self.contact_point_offset.unsqueeze(-1)
        self.contact_point_pos = self.body_pos.unsqueeze(2) + (R @ p).squeeze(-1)
        
        roll, pitch, yaw = euler_xyz_from_quat(self.body_quat.view(-1, 4))
        self.contact_point_euler = torch.stack((roll, pitch, yaw), dim=-1).view(self.num_envs, self.num_bodies, 1, 3).repeat(1, 1, self.num_contact_points, 1)
        self.contact_point_lin_vel = self.body_lin_vel.unsqueeze(2) + \
            torch.cross(self.body_ang_vel.unsqueeze(2), self.contact_point_pos - self.body_pos.unsqueeze(2), dim=-1)
        
        # pre-rotate roll and yaw
        Rt = self.body_rot_mat_roll_yaw.transpose(-1, -2).unsqueeze(2)
        v = self.contact_point_lin_vel.clone().unsqueeze(-1)
        self.contact_point_lin_vel_b = (Rt @ v).squeeze(-1)
        
        # compute velocity angle
        self.contact_point_intrusion_angle = torch.atan2(-self.contact_point_lin_vel_b[..., 2], self.contact_point_lin_vel_b[..., 0])
        self.contact_point_intrusion_angle = torch.where(
            self.contact_point_intrusion_angle > torch.pi/2, 
            torch.pi - self.contact_point_intrusion_angle, 
            self.contact_point_intrusion_angle
            )
        self.contact_point_intrusion_angle = torch.where(
            self.contact_point_intrusion_angle < -torch.pi/2,
            -torch.pi - self.contact_point_intrusion_angle,
            self.contact_point_intrusion_angle
            )
        
        # TODO: later replace simple height based contact detection with actual contact pair checking to handle rough terrain
        self.is_contact = -self.contact_point_pos[..., 2] > 0.0
        
    def _compute_force(self)->None:
        """
        Compute contact forces and torques.
        """
        f_normal = self._get_resistive_force(
            self.contact_point_pos.reshape(self.num_envs, -1, 3),
            self.contact_point_lin_vel.reshape(self.num_envs, -1, 3),
            self.contact_point_lin_vel_prev.reshape(self.num_envs, -1, 3),
            -self.contact_point_euler.reshape(self.num_envs, -1, 3)[..., 1], # beta = - pitch (physics ppl's coordinate is different from robotics)
            self.contact_point_intrusion_angle.reshape(self.num_envs, -1), # gamma
        )
        
        f_tangential = self._get_coulomb_friction_force(
            self.contact_point_lin_vel.reshape(self.num_envs, -1, 3),
            f_normal[:, :, 2]
        )
        
        # TODO: verify coulomb friction should be in each body's local frame or global frame?
        # p = f_normal.clone().reshape(self.num_envs, self.num_bodies, self.num_contact_points, 3).unsqueeze(-1)
        # R = self.body_rot_mat.unsqueeze(2)
        # contact_point_lin_vel_local = (R.transpose(-1, -2) @ self.contact_point_lin_vel.unsqueeze(-1)).squeeze(-1).reshape(self.num_envs, -1, 3)
        # force_normal_local = (R.transpose(-1, -2) @ p).squeeze(-1).reshape(self.num_envs, -1, 3)
        # f_tangential = self._get_coulomb_friction_force(
        #     contact_point_lin_vel_local,
        #     force_normal_local[:, :, 2]
        # )
        

        force = (f_normal+f_tangential).reshape(self.num_envs, self.num_bodies, self.num_contact_points, 3)
        torque = torch.cross((self.contact_point_pos - self.body_pos.unsqueeze(2)), force, dim=-1)
        
        self.contact_point_force[:, :, :, :] = force
        self.contact_point_torque[:, :, :, :] = torque
        
        # lumped force and torque
        self.contact_force[:, :, :] = torch.sum(force, dim=2) # (num_envs, num_bodies, 3)
        self.contact_torque[:, :, :] = torch.sum(torque, dim=2) # (num_envs, num_bodies, 3)
        
        # update velocity history
        self.contact_point_lin_vel_prev[:, :, :, :] = self.contact_point_lin_vel[:, :, :, :]
    
    def _get_resistive_force(
        self, 
        foot_pos:torch.Tensor, 
        foot_velocity:torch.Tensor, 
        foot_velocity_prev:torch.Tensor, 
        beta:torch.Tensor, 
        gamma:torch.Tensor)->torch.Tensor:
        """
        Compute normal force wrt global frame using RFT z component.
        
        Args: 
            foot_pos: contact point position in global frame. (num_envs, num_bodies*num_contact_points, 3)
            foot_velocity: contact point velocity in global frame. (num_envs, num_bodies*num_contact_points, 3)
            foot_velocity_prev: contact point velocity at previous time step in global frame. (num_envs, num_bodies*num_contact_points, 3)
            beta: intrusion angle (pitch) of contact point. (num_envs, num_bodies*num_contact_points)
            gamma: intrusion direction angle of contact point. (num_envs, num_bodies*num_contact_points)
        Returns:
            force_normal: normal force in global frame. (num_envs, num_bodies*num_contact_points, 3)
        """
        dA = self.surface_area/self.num_contact_points
        depth = -foot_pos[:, :, -1]
        is_contact = depth > 0 # apply resistive force only when foot is penetrating
        
        alpha_x, alpha_z = self._compute_elementary_force(beta, gamma) # get RFT force
        self.force_gm = self.stiffness[:, None] * alpha_z * depth * dA * is_contact * (1e6) #m^3 to cm^3 since alpha is N/cm^3
        self._ema_filtering(foot_velocity, foot_velocity_prev, depth)
        
        force_normal = torch.zeros((self.num_envs, self.num_bodies*self.num_contact_points, 3), device=self.device)
        force_normal[:, :, -1] = self.force_ema
        
        # rotate normal force back to simulation global frame
        force_normal = force_normal.reshape(self.num_envs, self.num_bodies, self.num_contact_points, 3)
        R = self.body_rot_mat_roll_yaw.unsqueeze(2)
        p = force_normal.unsqueeze(-1)
        force_normal = (R @ p).squeeze(-1).reshape(self.num_envs, -1, 3)
        
        # add dynamic inertial term from DRFT
        lam = 1.0
        rho = 638.0 * (1e-6) # kg/mm^3 to kg/cm^3
        force_normal[:, :, 2] = force_normal[:, :, 2] + lam * rho * foot_velocity[:, :, 2]**2

        return force_normal
    
    def _compute_elementary_force(self, beta:torch.Tensor, gamma:torch.Tensor)->Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute elementary force per foot using Fourier series expansion. 
        See the original paper. 
        
        Args: 
            beta: intrusion angle (pitch) of contact point. (num_envs, num_bodies*num_contact_points)
            gamma: intrusion direction angle of contact point. (num_envs, num_bodies*num_contact_points)
        Returns:
            alpha_x: elementary force coefficient in x direction. (num_envs, num_bodies*num_contact_points)
            alpha_z: elementary force coefficient in z direction. (num_envs, num_bodies*num_contact_points)
        """
        alpha_z = torch.zeros_like(beta) # A00
        alpha_z += self.cfg.A00*torch.cos(2*torch.pi*(0*beta/torch.pi)) #A00
        alpha_z += self.cfg.A10*torch.cos(2*torch.pi*(1*beta/torch.pi)) # A10
        alpha_z += self.cfg.B01*torch.sin(2*torch.pi*(1*gamma/(2*torch.pi))) # B01
        alpha_z += self.cfg.B11*torch.sin(2*torch.pi*(1*beta/torch.pi + 1*gamma/(2*torch.pi))) # B11
        alpha_z += self.cfg.B_11*torch.sin(2*torch.pi*(-1*beta/torch.pi + 1*gamma/(2*torch.pi))) # B-11
        
        # calculate alpha_x
        alpha_x = torch.zeros_like(beta)
        alpha_x += self.cfg.C01*torch.sin(2*torch.pi*(1*gamma/(2*torch.pi))) # C01
        alpha_x += self.cfg.C11*torch.sin(2*torch.pi*(1*beta/(torch.pi) + 1*gamma/(2*torch.pi))) # C11
        alpha_x += self.cfg.C_11*torch.sin(2*torch.pi*(-1*beta/(torch.pi) + 1*gamma/(2*torch.pi))) # C-11
        alpha_x += self.cfg.D10*torch.cos(2*torch.pi*(1*beta/(torch.pi))) # D10
        
        return alpha_x, alpha_z
    
    def _ema_filtering(self, velocity:torch.Tensor, velocity_prev:torch.Tensor, depth:torch.Tensor):
        """
        Exponential moving filtering
        See KAIST science robotics supplementary material S12.
        
        Args:
            velocity: contact point velocity in global frame. (num_envs, num_bodies*num_contact_points, 3)
            velocity_prev: contact point velocity at previous time step in global frame. (num_envs, num_bodies*num_contact_points, 3)
            depth: contact point penetration depth. (num_envs, num_bodies*num_contact_points)
        """
        coef = 0.8
        increment_mask = velocity[:,:, -1]*velocity_prev[:, :, -1] < 0
        tau_r_boundary = self.tau_r < 1
        depth_mask = depth > 0
        mask = increment_mask & tau_r_boundary
        self.tau_r[mask] += self.c_r
        self.tau_r[~depth_mask] = 0.0
        
        self.force_ema[depth_mask] = (1-coef*self.tau_r[depth_mask])*self.force_gm[depth_mask] + coef*self.tau_r[depth_mask]*self.force_ema[depth_mask]
        self.force_ema[~depth_mask] = 0.0
    
    def _get_coulomb_friction_force(self, foot_velocity:torch.Tensor, fz:torch.Tensor)->torch.Tensor:
        """
        Get tangential force using Coulomb friction model. 
        Computed force is in simulation global frame.
        Idea is from https://iscicra25.github.io/papers/2025-Lee-4_Soft_Contact_Model_for_Robus.pdf
        
        Args:
            foot_velocity: contact point velocity in global frame. (num_envs, num_bodies*num_contact_points, 3)
            fz: normal force in z direction in global frame. (num_envs, num_bodies*num_contact_points)
        Returns:
            tangential_force: tangential force in global frame. (num_envs, num_bodies*num_contact_points, 3)
        """
        
        vt = torch.sqrt(foot_velocity[:, :, 0]**2 + foot_velocity[:, :, 1]**2)
        vt_dir = foot_velocity[:, :, :2]/(vt.unsqueeze(2) + 1e-6)
        
        # filter tangential velocity
        # self.filter_tangential_velocity(vt, vt_dir, dt=1/200)
        # vt = self.vt_filtered
        # vt_dir = self.vt_unit_filtered
        
        # # combines Coulomb friction and Stribeck friction model
        # v_cf = 0.2 # Coulomb friction velocity threshold
        # v_st = 0.01 # Stribeck friction velocity threshold
        # friction_force = (self.dynamic_friction_coef * torch.tanh(vt/v_cf) + \
        #     math.sqrt(2*math.e) * (self.static_friction_coef - self.dynamic_friction_coef) * torch.exp(-(vt/v_st)**2) * (vt/v_st)) * torch.abs(fz)
        # friction_force_vec = -friction_force.unsqueeze(2) * vt_dir
        
        # Coulomb friction
        kf = 1.0
        friction_force = torch.minimum(-self.dynamic_friction_coef * fz, kf * vt)
        friction_force_vec = friction_force.unsqueeze(2) * vt_dir
        
        tangential_force = torch.zeros((self.num_envs, self.num_bodies*self.num_contact_points, 3), device=self.device)
        tangential_force[:, :, :2] = friction_force_vec
        
        # TODO: verify coulomb friction should be in each body's local frame or global frame?
        # # rotate back to global frame
        # tangential_force = tangential_force.reshape(self.num_envs, self.num_bodies, self.num_contact_points, 3)
        # R = self.body_rot_mat.unsqueeze(2)
        # tangential_force = (R @ tangential_force.unsqueeze(-1)).squeeze(-1).reshape(self.num_envs, -1, 3)
        
        return tangential_force
    
    def filter_tangential_velocity(self, vt, vt_unit_vec, dt, tau=0.02):
        """
        Filter tangential velocity using exponential moving average.
        Args:
            vt: tangential velocity magnitude. (num_envs, num_bodies*num_contact_points)
            vt_unit_vec: tangential velocity unit vector. (num_envs, num_bodies*num_contact_points, 2)
            dt: time step
            tau: time constant
        """
        # alpha = dt / (tau + dt)
        alpha = 0.95
        self.vt_filtered = (1 - alpha) * self.vt_filtered + alpha * vt

        # For direction, normalize after filtering
        self.vt_unit_filtered = (1 - alpha) * self.vt_unit_filtered + alpha * vt_unit_vec
        self.vt_unit_filtered = self.vt_unit_filtered / (
            torch.norm(self.vt_unit_filtered, dim=-1, keepdim=True) + 1e-6
        )
    
    def reset(self, env_ids:torch.Tensor):
        """
        Reset internal states for given env ids.
        Args:
            env_ids: tensor of env ids to reset
        """
        self.force_gm[env_ids] = 0.0
        self.force_ema[env_ids] = 0.0
        self.tau_r[env_ids] = 0.0
        self.contact_point_lin_vel_prev[env_ids] = 0.0


"""
3D RFT material parameters.
"""
        
@configclass
class Material3DRFTCfg:
    """
    See https://www.pnas.org/doi/10.1073/pnas.2214017120 supplementary material S3. 
    """
    # c1^k
    c1_1: float = 0.00212
    c1_2: float = -0.02320
    c1_3: float = -0.20890
    c1_4: float = -0.43083
    c1_5: float = -0.00259
    c1_6: float = 0.48872
    c1_7: float = -0.00415
    c1_8: float = 0.07204
    c1_9: float = -0.02750
    c1_10: float = -0.08772
    c1_11: float = 0.01992
    c1_12: float = -0.45961
    c1_13: float = 0.40799
    c1_14: float = -0.10107
    c1_15: float = -0.06576
    c1_16: float = 0.05664
    c1_17: float = -0.09269
    c1_18: float = 0.01892
    c1_19: float = 0.01033
    c1_20: float = 0.15120

    # c2^k
    c2_1: float = -0.06796
    c2_2: float = -0.10941
    c2_3: float = 0.04725
    c2_4: float = -0.06914
    c2_5: float = -0.05835
    c2_6: float = -0.65880
    c2_7: float = -0.11985
    c2_8: float = 0.25739
    c2_9: float = -0.26834
    c2_10: float = 0.02692
    c2_11: float = -0.00736
    c2_12: float = 0.63758
    c2_13: float = 0.08997
    c2_14: float = 0.21069
    c2_15: float = 0.04748
    c2_16: float = 0.27066
    c2_17: float = 0.18519
    c2_18: float = 0.04934
    c2_19: float = 0.13527
    c2_20: float = -0.33207

    # c3^k
    c3_1: float = -0.02634
    c3_2: float = -0.03436
    c3_3: float = 0.45256
    c3_4: float = 0.00835
    c3_5: float = 0.02553
    c3_6: float = -1.31290
    c3_7: float = -0.05532
    c3_8: float = 0.06790
    c3_9: float = -0.61404
    c3_10: float = 0.02287
    c3_11: float = 0.02927
    c3_12: float = 0.95406
    c3_13: float = 0.09701
    c3_14: float = -0.11028
    c3_15: float = 0.01487
    c3_16: float = 0.20770
    c3_17: float = 0.10911
    c3_18: float = -0.04097
    c3_19: float = 0.07881
    c3_20: float = -0.27519

    hardness: float = 1.0
        
class RFT_3D:
    def __init__(self, 
                 cfg: Material3DRFTCfg, 
                 num_envs: int,
                 num_bodies: int, 
                 num_contact_points: int,
                 device: torch.device,
                 static_friction_coef: float=1.0, 
                 dynamic_friction_coef: float=0.3, 
                 ):
        """
        3D Resistive Force Theory based soft terrain contact solver.
        https://www.pnas.org/doi/10.1073/pnas.2214017120
        """
        
        self.cfg = cfg
        self.num_envs = num_envs
        self.num_bodies = num_bodies
        self.num_contact_points = num_contact_points
        self.device = device
        
        self.contact_edge_x = (-0.06, 0.14)
        self.contact_edge_y = (-0.03, 0.03)
        self.foot_depth = 0.035
        self.surface_area = (self.contact_edge_x[1]-self.contact_edge_x[0])*(self.contact_edge_y[1]-self.contact_edge_y[0])
        
        self.c_r = 0.05 # 100/f (e.g. f=2000hz -> 0.05)
        self.static_friction_coef = static_friction_coef
        self.dynamic_friction_coef = dynamic_friction_coef
        
        self.create_internal_tensors()
        self.create_contact_points()
    
    """
    Initializations.
    """
    def create_internal_tensors(self):
        self.body_pos = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device)
        self.body_quat = torch.zeros((self.num_envs, self.num_bodies, 4), device=self.device)
        self.body_rot_mat = torch.zeros((self.num_envs, self.num_bodies, 3, 3), device=self.device)
        self.body_rot_mat_roll_yaw = torch.zeros((self.num_envs, self.num_bodies, 3, 3), device=self.device)
        self.body_lin_vel = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device)
        self.body_ang_vel = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device)

        self.contact_point_offset = torch.zeros((self.num_envs, self.num_bodies, self.num_contact_points, 3), device=self.device)
        self.contact_point_pos = torch.zeros((self.num_envs, self.num_bodies, self.num_contact_points, 3), device=self.device)
        self.contact_point_euler = torch.zeros((self.num_envs, self.num_bodies, self.num_contact_points, 3), device=self.device)
        self.contact_point_lin_vel = torch.zeros((self.num_envs, self.num_bodies, self.num_contact_points, 3), device=self.device)
        self.contact_point_lin_vel_b = torch.zeros((self.num_envs, self.num_bodies, self.num_contact_points, 3), device=self.device)
        self.contact_point_intrusion_angle = torch.zeros((self.num_envs, self.num_bodies, self.num_contact_points), device=self.device)
        self.contact_point_lin_vel_prev = torch.zeros((self.num_envs, self.num_bodies, self.num_contact_points, 3), device=self.device)
        
        self.contact_point_force = torch.zeros((self.num_envs, self.num_bodies, self.num_contact_points, 3), device=self.device)
        self.contact_point_torque = torch.zeros((self.num_envs, self.num_bodies, self.num_contact_points, 3), device=self.device)
        self.contact_point_filtered_force = torch.zeros((self.num_envs, self.num_bodies, self.num_contact_points, 3), device=self.device)
        
        # lumped force and torque
        self.contact_force = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device)
        self.contact_torque = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device)
        
        self.force_gm = torch.zeros((self.num_envs, self.num_bodies*self.num_contact_points), device=self.device)
        self.force_ema = torch.zeros((self.num_envs, self.num_bodies*self.num_contact_points), device=self.device)
        self.tau_r = torch.zeros((self.num_envs, self.num_bodies*self.num_contact_points), device=self.device)
        
    def create_contact_points(self)->None:
        """
        Create contact points in contact surface in root body frame (e.g. foot link frame). 
        """
        num_contact_point_side = int(math.sqrt(self.num_contact_points))
        assert num_contact_point_side**2 == self.num_contact_points, "num_contact_points must be a perfect square"
        contact_point_offset_x = torch.linspace(self.contact_edge_x[0], self.contact_edge_x[1], num_contact_point_side, device=self.device)
        contact_point_offset_y = torch.linspace(self.contact_edge_y[0], self.contact_edge_y[1], num_contact_point_side, device=self.device)
        contact_point_offset_y, contact_point_offset_x = torch.meshgrid(contact_point_offset_y, contact_point_offset_x, indexing='ij')
        contact_point_offset = torch.stack((
            contact_point_offset_x.flatten(), 
            contact_point_offset_y.flatten(), 
            -self.foot_depth * torch.ones_like(contact_point_offset_x).flatten()
            ), dim=-1)
        contact_point_offset = contact_point_offset.unsqueeze(0).unsqueeze(1).repeat(self.num_envs, self.num_bodies, 1, 1) # (num_envs, num_bodies, num_contact_points, 3)
        self.contact_point_offset[:, :, :, :] = contact_point_offset
        
    """
    operations.
    """
    
    def update(self):
        raise NotImplementedError("3D RFT not implemented yet.")
    
    def reset(self, env_ids:torch.Tensor):
        raise NotImplementedError("3D RFT not implemented yet.")
    
    
    """
    properties.
    """
    
    @property
    def contact_wrench(self)->torch.Tensor:
        return torch.cat((self.contact_force, self.contact_torque), dim=-1) # (num_envs, num_bodies, 6)
    
    @property
    def contact_wrench_b(self)->torch.Tensor:
        contact_force_b = (self.body_rot_mat.transpose(-1, -2) @ self.contact_force.unsqueeze(-1)).squeeze(-1) # (num_envs, num_bodies, 3)
        contact_torque_b = (self.body_rot_mat.transpose(-1, -2) @ self.contact_torque.unsqueeze(-1)).squeeze(-1) # (num_envs, num_bodies, 3)
        return torch.cat((contact_force_b, contact_torque_b), dim=-1) # (num_envs, num_bodies, 6)
    
    @property
    def contact_point_wrench(self)->torch.Tensor:
        return torch.cat((self.contact_point_force, self.contact_point_torque), dim=-1) # (num_envs, num_bodies, num_contact_points, 6)


if __name__ == "__main__":
    material_cfg = PoppySeedLPCfg()
    num_envs = 10 
    num_bodies = 2
    num_contact_points = 100
    device = torch.device('cuda:0')
    rft = RFT_EMF(material_cfg=material_cfg, num_envs=num_envs, num_bodies=num_bodies, device=device, dt=1/200)
    body_pos = torch.zeros((10, 2, 3), device=torch.device('cuda:0'))
    body_quat = torch.tensor([1, 0, 0, 0], device=torch.device('cuda:0')).unsqueeze(0).unsqueeze(0).repeat(10, 2, 1)
    body_lin_vel = torch.zeros((10, 2, 3), device=torch.device('cuda:0'))
    body_ang_vel = torch.zeros((10, 2, 3), device=torch.device('cuda:0'))
    body_pos[:, :, 2] = 0.01
    body_lin_vel[:, :, 0] = 0.1
    rft.update(body_pos, body_quat, body_lin_vel, body_ang_vel)
    contact_wrench = rft.contact_wrench
    contact_point_wrench = rft.contact_point_wrench
    print(contact_wrench.shape)
    print(contact_point_wrench.shape)