from dataclasses import MISSING
from time import time
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
    """
    Material configuration for soft contact model.

    A00 - D10: quasistatic RFT Fourier coefficients.
    lam, rho: dynamic RFT parameters.
    static_friction_coef, dynamic_friction_coef: friction coefficients.
    kf: tangential force model parameter.
    kh, beta_d, bh: horizontal stroke resistive force model parameters.
    """

    # quasistatic RFT fourier coefficients
    A00: float = MISSING # type: ignore
    A10: float = MISSING # type: ignore
    B11: float = MISSING # type: ignore
    B01: float = MISSING # type: ignore
    B_11: float = MISSING # type: ignore
    C11: float = MISSING # type: ignore
    C01: float = MISSING # type: ignore
    C_11: float = MISSING # type: ignore
    D10: float = MISSING # type: ignore

    # dynamic RFT parameters
    lam: float = MISSING # type: ignore
    rho: float = MISSING # type: ignore

    # material properties
    static_friction_coef: float = MISSING # type: ignore # TODO remove
    dynamic_friction_coef: float = MISSING # type: ignore

    # coulomb friction tangential force model parameters
    kf: float = MISSING # type: ignore

    # horizontal stroke resistive force model parameters
    kh: float = MISSING # type: ignore
    beta_d: float = MISSING # type: ignore
    bh: float = MISSING # type: ignore

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

    # dynamic RFT parameters
    lam: float = 1.0
    rho: float = 638.0 * (1e-6) # kg/mm^3 to kg/cm^3

    # material properties
    static_friction_coef: float = 1.0 # TODO remove
    dynamic_friction_coef: float = 0.5

    # tangential force model parameters
    kf: float = 10.0

    # horizontal stroke resistive force model parameters
    kh: float = 50.0
    beta_d: float = 0.5
    bh: float = 1.0

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

    lam: float = 1.0
    rho: float = 638.0 * (1e-6) # kg/mm^3 to kg/cm^3

    static_friction_coef: float = 1.0 # TODO remove
    dynamic_friction_coef: float = 0.5

    kf: float = 10.0

    # horizontal stroke resistive force model parameters
    kh: float = 50.0
    beta_d: float = 0.5
    bh: float = 1.0


"""
3D RFT material parameters.
"""
        
@configclass
class Material3DRFTCfg:
    """
    Material parameters for 3D RFT soft contact model.
    See https://www.pnas.org/doi/10.1073/pnas.2214017120 supplementary material S3. 

    c1^k, c2^k, c3^k are the coefficients of 3rd order polynomial that computes the resistive force per volume.
    static_friction_coef: static friction coefficient
    dynamic_friction_coef: dynamic friction coefficient
    rho_c: critical media density (effective media density = packing fraction * grain density)
    mu_int: media internal friction coefficient
    """
    # c1^k
    coef_1: list[float] = [
        0.00212, -0.02320, -0.20890, -0.43083, -0.00259, 0.48872, -0.00415, 0.07204, -0.02750, -0.08772,
        0.01992, -0.45961, 0.40799, -0.10107, -0.06576, 0.05664, -0.09269, 0.01892, 0.01033, 0.15120
    ]
    # c2^k
    coef_2: list[float] = [
        -0.06796, -0.10941, 0.04725, -0.06914, -0.05835, -0.65880, -0.11985, -0.25739, -0.26834, 0.02692,
        -0.00736, 0.63758, 0.08997, 0.21069, 0.04748, 0.20406, 0.18519, 0.04934, 0.13527, -0.33207
    ]
    # c3^k
    coef_3: list[float] = [
        -0.02634, -0.03436, 0.45256, 0.00835, 0.02553, -1.31290, -0.05532, 0.06790, -0.16404, 0.02287,
        0.02927, 0.95406, -0.00131, -0.11028, 0.01487, -0.20770, 0.10911, -0.04097, 0.07881, -0.27519
    ]

    # 3D RFT media specific properties
    static_friction_coef: float = 1.0 # TODO remove
    dynamic_friction_coef: float = 0.4
    mu_int: float = 0.4 # media internal friction coefficient
    rho_c: float = 3000.0 # critical media density (effective media density = packing fraction * grain density)


"""
Intruder geometry configuration.
"""

@configclass
class IntruderGeometryCfg:
    """
    Intruder surface is approximated as rectangle.
    contact_edge_*: tuple of lower and upper bounds of corner from intruder link origin.
    """
    contact_edge_x: Tuple[float, float] = MISSING # type: ignore # length in x direction (m)
    contact_edge_y: Tuple[float, float] = MISSING # type: ignore # length in y direction (m)
    contact_edge_z: Tuple[float, float] = MISSING # type: ignore # length in z direction (m)
    num_contact_points:int = 10*10

"""
2D RFT with dynamic inertial modification (DRFT) + exponential moving average filtering.
"""

class RFT_2D:
    def __init__(self, 
                 num_envs: int,
                 num_bodies: int, 
                 device: torch.device,
                 dt:float, 
                 history_length: int = 3,
                 enable_ema_filter: bool = True,
                 material_cfg: MaterialCfg=PoppySeedCPCfg(), 
                 intruder_cfg: IntruderGeometryCfg=IntruderGeometryCfg(),
                 ):
        """
        Soft contact model based on 2D RFT proposed in
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
        self.c_r = 100/(1/self.dt) # 100/f (e.g. f=2000hz -> 0.05)
        self.history_length = history_length
        self.enable_ema_filter = enable_ema_filter
        
        self.contact_edge_x = intruder_cfg.contact_edge_x
        self.contact_edge_y = intruder_cfg.contact_edge_y
        self.contact_edge_z = intruder_cfg.contact_edge_z
        self.foot_depth = self.contact_edge_z[1] - self.contact_edge_z[0]
        self.surface_area = (self.contact_edge_x[1]-self.contact_edge_x[0])*(self.contact_edge_y[1]-self.contact_edge_y[0])

        self._data: SoftContactData = SoftContactData()
        self.create_tensors()
        self.create_contact_points()
        self.initialize_data()
        print("-"*40)
        print("2D RFT soft contact model initialized.")
        print(f"Number of envs: {self.num_envs}")
        print(f"Number of bodies per env: {self.num_bodies}")
        print(f"Number of contact points per body: {self.num_contact_points}")
        print(f"Contact surface area per body: {self.surface_area} m^2")
        print("-"*40)
    
    """
    Initialization.
    """
        
    def create_tensors(self):
        """
        Create buffer tensors.
        """
        self.body_pos = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device)
        self.body_quat = torch.zeros((self.num_envs, self.num_bodies, 4), device=self.device)
        self.body_rot_mat = torch.zeros((self.num_envs, self.num_bodies, 3, 3), device=self.device)
        self.body_lin_vel = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device)
        self.body_ang_vel = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device)
        
        self.contact_point_local = torch.zeros((self.num_envs, self.num_bodies, self.num_contact_points, 3), device=self.device)
        self.n_dir_local = torch.zeros((self.num_envs, self.num_bodies, self.num_contact_points, 3), device=self.device)

        self.contact_point_offset = torch.zeros((self.num_envs, self.num_bodies, self.num_contact_points, 3), device=self.device)
        self.contact_point_pos = torch.zeros((self.num_envs, self.num_bodies, self.num_contact_points, 3), device=self.device)
        self.contact_point_lin_vel = torch.zeros((self.num_envs, self.num_bodies, self.num_contact_points, 3), device=self.device)
        self.contact_point_lin_vel_prev = torch.zeros((self.num_envs, self.num_bodies, self.num_contact_points, 3), device=self.device)
        self.contact_point_tilt_angle = torch.zeros((self.num_envs, self.num_bodies, self.num_contact_points), device=self.device)
        self.contact_point_intrusion_angle = torch.zeros((self.num_envs, self.num_bodies, self.num_contact_points), device=self.device)
        
        # local coordinate
        self.n_dir = torch.zeros((self.num_envs, self.num_bodies, self.num_contact_points, 3), device=self.device)
        
        # contact wrench
        self.contact_point_force = torch.zeros((self.num_envs, self.num_bodies, self.num_contact_points, 3), device=self.device)
        self.contact_point_torque = torch.zeros((self.num_envs, self.num_bodies, self.num_contact_points, 3), device=self.device)
        
        self.vt_filtered = torch.zeros((self.num_envs, self.num_bodies*self.num_contact_points), device=self.device)
        self.vt_unit_filtered = torch.zeros((self.num_envs, self.num_bodies*self.num_contact_points, 2), device=self.device)
        
        # lumped force and torque
        self.contact_force = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device)
        self.contact_torque = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device)
        
        self.force_gm = torch.zeros((self.num_envs, self.num_bodies*self.num_contact_points), device=self.device)
        self.force_ema = torch.zeros((self.num_envs, self.num_bodies*self.num_contact_points), device=self.device)
        self.tau_r = torch.zeros((self.num_envs, self.num_bodies*self.num_contact_points), device=self.device)

        # material parameters
        self.stiffness = torch.ones(self.num_envs, device=self.device)
        self.static_friction_coef = self.cfg.static_friction_coef * torch.ones(self.num_envs, device=self.device)
        self.dynamic_friction_coef = self.cfg.dynamic_friction_coef * torch.ones(self.num_envs, device=self.device)
        self.lam = self.cfg.lam * torch.ones(self.num_envs, device=self.device)
        self.rho = self.cfg.rho * torch.ones(self.num_envs, device=self.device)
        self.kf = self.cfg.kf * torch.ones(self.num_envs, device=self.device)
        
        self._timestamp = torch.zeros(self.num_envs, device=self.device)
        self._timestamp_last_update = torch.zeros(self.num_envs, device=self.device)
        
        
        self.body_rot_mat_roll_yaw = torch.zeros((self.num_envs, self.num_bodies, 3, 3), device=self.device)
        self.contact_point_lin_vel_b = torch.zeros((self.num_envs, self.num_bodies, self.num_contact_points, 3), device=self.device)
        
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
        self.contact_point_local[:, :, :, :] = contact_point_offset
        self.n_dir_local = torch.tensor([0.0, 0.0, -1.0], device=self.device).view(1, 1, 1, 3).repeat(self.num_envs, self.num_bodies, self.num_contact_points, 1)
    
    
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
    operations.
    """
    
    def update(self, body_pos:torch.Tensor, body_quat:torch.Tensor, body_lin_vel:torch.Tensor, body_ang_vel:torch.Tensor)->None:
        """
        Update soft contact model.
        
        Args:
            body_pos: intruder position. (num_envs, num_bodies, 3)
            body_quat: intruder orientation in quaternion form. (num_envs, num_bodies, 4)
            body_lin_vel: intruder linear velocity wrt global frame. (num_envs, num_bodies, 3)
            body_ang_vel: intruder angular velocity wrt global frame. (num_envs, num_bodies, 3)
        """
        
        # copy intruder's kinematic states to tensor buffers
        self.body_pos[:, :, :] = body_pos
        self.body_quat[:, :, :] = body_quat
        self.body_rot_mat[:, :, :, :] = matrix_from_quat(body_quat.view(-1, 4)).view(self.num_envs, self.num_bodies, 3, 3)
        self.body_lin_vel[:, :, :] = body_lin_vel
        self.body_ang_vel[:, :, :] = body_ang_vel

        # evaluate contact forces
        self._eval_contacts()
        
        # update timestamps
        self._timestamp += self.dt
        self._update_data(torch.arange(self.num_envs, device=self.device))
        self._timestamp_last_update[:] = self._timestamp[:]

    def randomize_ground_stiffness(self, env_ids:torch.Tensor, stiffness:torch.Tensor)->None:
        """
        Update ground stiffness (N/m) for each env.
        Implementation is similar to terrain curriculum used in terrain importer class.
        This can be triggered by curriculum manager.
        
        Args:
            env_ids: tensor of env ids to update
            stiffness: tensor of stiffness values (len(env_ids),)
        """
        self.stiffness[env_ids] = stiffness # can be 0.25 - 10.0 ?
        
    def update_friction_params(self, env_ids:torch.Tensor, static_friction_coef:torch.Tensor, dynamic_friction_coef:torch.Tensor)->None:
        """
        Update friction coefficients for each env.
        This can be triggered by event manager.
        
        Args:
            env_ids: tensor of env ids to update
            static_friction_coef: tensor of static friction coefficients (len(env_ids),)
            dynamic_friction_coef: tensor of dynamic friction coefficients (len(env_ids),)
        """
        self.static_friction_coef[env_ids] = static_friction_coef
        self.dynamic_friction_coef[env_ids] = dynamic_friction_coef
        

    
    """
    helper functions.
    """
    
    def _update_data(self, env_ids:torch.Tensor)->None:
        """
        Update soft contact data.
        Majority of implementations are from IsaacLab's contact sensor class.
        
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
        
        
    def _eval_contacts(self)->None:
        """
        Update contact points' kinematic states and compute contact forces.
        """
        
        # step1: calculate contact pos, lin vel, and surface normal
        # compute contact points in global frame
        R = self.body_rot_mat.unsqueeze(2)
        p = self.contact_point_local.unsqueeze(-1)
        self.contact_point_pos = self.body_pos.unsqueeze(2) + (R @ p).squeeze(-1)
        
        # compute contact normal
        self.n_dir = (R @ self.n_dir_local.unsqueeze(-1)).squeeze(-1)
        
        # compute contact point linear velocity in global frame
        self.contact_point_lin_vel = self.body_lin_vel.unsqueeze(2) + \
            torch.cross(self.body_ang_vel.unsqueeze(2), self.contact_point_pos - self.body_pos.unsqueeze(2), dim=-1)
            
        # step2: Find local coordinate frame {r, theta, z}
        # compute local coordinate vectors in global frame.
        # see S7 eq.4 from https://www.pnas.org/doi/10.1073/pnas.2214017120
        thresh = 1e-10
        n = self.n_dir.clone()
        z = torch.tensor([0.0, 0.0, 1.0], device=self.device).view(1, 1, 1, 3).repeat(self.num_envs, self.num_bodies, self.num_contact_points, 1)
        v = self.contact_point_lin_vel.clone()/(torch.norm(self.contact_point_lin_vel, dim=-1, keepdim=True) + 1e-6)
        vr = v - (v * z).sum(dim=-1, keepdim=True) * z
        vr_norm = torch.norm(vr, dim=-1, keepdim=True)
        n_rt = self.n_dir - (self.n_dir * z).sum(dim=-1, keepdim=True) * z
        n_rt_norm = torch.norm(n_rt, dim=-1, keepdim=True)
        n_rt_dir = n_rt / (n_rt_norm+1e-6)
        r = vr / (vr_norm+1e-6)
        r[vr_norm.squeeze(-1) < thresh] = n_rt_dir[vr_norm.squeeze(-1) < thresh]
        t = torch.cross(z, r, dim=-1)
        
        # step3: compute characteristic angles
        
        # compute contact point velocity angle (gamma)
        # see S7 eq.6 from https://www.pnas.org/doi/10.1073/pnas.2214017120
        vdotr = (v * r).sum(dim=-1)
        vdotz = (v * z).sum(dim=-1)
        self.contact_point_intrusion_angle = torch.acos(vdotr) * ((vdotz<0).float() - (vdotz>=0).float())
        self.contact_point_intrusion_angle = torch.nan_to_num(self.contact_point_intrusion_angle, nan=0.0, posinf=0.0, neginf=0.0)
        
        # compute contact point tilt angle (beta)
        # see S7 eq.5 from https://www.pnas.org/doi/10.1073/pnas.2214017120
        ndotr = (n * r).sum(dim=-1)
        ndott = (n * t).sum(dim=-1)
        ndotz = (n * z).sum(dim=-1)
        n_rtz = torch.cat([ndotr.unsqueeze(-1), ndott.unsqueeze(-1), ndotz.unsqueeze(-1)], dim=-1)
        reflection_matrix = (1 - 2 * (ndotr<0).float()).unsqueeze(-1)
        n_rtz = n_rtz * reflection_matrix
        self.contact_point_tilt_angle = -torch.acos(n_rtz[:, :, :, 2]) + torch.pi * (n_rtz[:, :, :, 2] < 0).float()
        self.contact_point_tilt_angle = torch.nan_to_num(self.contact_point_tilt_angle, nan=0.0, posinf=0.0, neginf=0.0)
        
        # step4: compute resistive force per contact points
        f_normal = self._get_normal_force(
            self.contact_point_pos.reshape(self.num_envs, -1, 3),
            self.contact_point_lin_vel.reshape(self.num_envs, -1, 3),
            self.contact_point_lin_vel_prev.reshape(self.num_envs, -1, 3),
            self.contact_point_tilt_angle.reshape(self.num_envs, -1),
            self.contact_point_intrusion_angle.reshape(self.num_envs, -1), # gamma
        )
        
        f_tangential = self._get_tangential_force(
            self.contact_point_lin_vel.reshape(self.num_envs, -1, 3),
            f_normal[:, :, 2], 
        )

        # compute final contact wrench per contact point
        force = (f_normal+f_tangential).reshape(self.num_envs, self.num_bodies, self.num_contact_points, 3)
        torque = torch.cross((self.contact_point_pos - self.body_pos.unsqueeze(2)), force, dim=-1)
        self.contact_point_force[:, :, :, :] = force
        self.contact_point_torque[:, :, :, :] = torque
        
        # sum force and torque over contact points
        self.contact_force[:, :, :] = torch.sum(force, dim=2) # (num_envs, num_bodies, 3)
        self.contact_torque[:, :, :] = torch.sum(torque, dim=2) # (num_envs, num_bodies, 3)
        
        # update velocity history
        self.contact_point_lin_vel_prev[:, :, :, :] = self.contact_point_lin_vel[:, :, :, :]
    
    
    def _get_normal_force(
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
        force_normal = torch.zeros((self.num_envs, self.num_bodies*self.num_contact_points, 3), device=self.device)

        dA = self.surface_area/self.num_contact_points
        depth = -foot_pos[:, :, -1]
        is_contact = depth > 0 # apply resistive force only when foot is penetrating
        
        alpha_x, alpha_z = self._compute_elementary_force(beta, gamma) # get RFT force
        self.force_gm = self.stiffness[:, None] * alpha_z * depth * dA * is_contact * (1e6) #m^3 to cm^3 since alpha is N/cm^3

        if self.enable_ema_filter:
            self._ema_filtering(foot_velocity, foot_velocity_prev, depth)
            force_normal[:, :, -1] = self.force_ema
        else:
            force_normal[:, :, -1] = self.force_gm

        # add dynamic inertial term from DRFT
        vn = foot_velocity[:, :, 2]
        force_normal[:, :, 2] = force_normal[:, :, 2] + is_contact * self.lam.unsqueeze(1) * self.rho.unsqueeze(1) * vn**2

        return force_normal
    
    def _get_tangential_force(self, v:torch.Tensor, fn:torch.Tensor)->torch.Tensor:
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
        vt_norm = torch.sqrt(v[:, :, 0]**2 + v[:, :, 1]**2)
        vt_dir = v[:, :, :2]/(vt_norm.unsqueeze(2) + 1e-6)
        
        # Coulomb friction (see https://github.com/newton-physics/newton/blob/main/newton/_src/solvers/semi_implicit/kernels_contact.py L537-L543)
        # ft = torch.minimum(self.dynamic_friction_coef * fn, kf * vt_norm) # (num_envs, num_bodies*num_contact_points)
        ft = self.kf.unsqueeze(1) * vt_norm
        # cone_violation = self.kf.unsqueeze(1) * vt_norm >= self.static_friction_coef * fn
        cone_violation = self.kf.unsqueeze(1) * vt_norm >= self.dynamic_friction_coef * fn
        ft[cone_violation] = self.dynamic_friction_coef * fn[cone_violation]
        
        tangential_force = torch.zeros((self.num_envs, self.num_bodies*self.num_contact_points, 3), device=self.device)
        tangential_force[:, :, :2] = -ft.unsqueeze(-1) * vt_dir
        
        return tangential_force
    
    def _get_hsr_force(self):
        """
        Horizontal stroke force (HSR) model to model sand drag force.
        """

        pass
    
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
        coef = 0.8 # smoothing coefficient
        increment_mask = velocity[:,:, -1]*velocity_prev[:, :, -1] < 0
        tau_r_boundary = self.tau_r < 1
        depth_mask = depth > 0
        mask = increment_mask & tau_r_boundary
        self.tau_r[mask] += self.c_r
        self.tau_r[~depth_mask] = 0.0
        
        self.force_ema[depth_mask] = (1-coef*self.tau_r[depth_mask])*self.force_gm[depth_mask] + coef*self.tau_r[depth_mask]*self.force_ema[depth_mask]
        self.force_ema[~depth_mask] = 0.0
        
    """
    reset.
    """
    
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
3D RFT soft contact model.
"""
        
class RFT_3D:
    def __init__(self, 
                 num_envs: int,
                 num_bodies: int, 
                 device: torch.device,
                 dt:float, 
                 history_length: int = 3,
                 enable_ema_filter: bool = True,
                 material_cfg: Material3DRFTCfg=Material3DRFTCfg(), 
                 intruder_cfg: IntruderGeometryCfg=IntruderGeometryCfg(),
                 )->None:
        """
        Soft contact model based on 3D RFT proposed in
        https://www.pnas.org/doi/10.1073/pnas.2214017120
        
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
        self.c_r = 100/(1/self.dt) # 100/f (e.g. f=2000hz -> 0.05)
        self.history_length = history_length
        self.enable_ema_filter = enable_ema_filter
        
        self.contact_edge_x = intruder_cfg.contact_edge_x
        self.contact_edge_y = intruder_cfg.contact_edge_y
        self.contact_edge_z = intruder_cfg.contact_edge_z
        self.foot_depth = self.contact_edge_z[1] - self.contact_edge_z[0]
        self.surface_area = (self.contact_edge_x[1]-self.contact_edge_x[0])*(self.contact_edge_y[1]-self.contact_edge_y[0])

        self._data: SoftContactData = SoftContactData()
        self.create_tensors()
        self.create_contact_points()
        self.initialize_data()

        print("-"*40)
        print("3D RFT soft contact model initialized.")
        print(f"Number of envs: {self.num_envs}")
        print(f"Number of bodies per env: {self.num_bodies}")
        print(f"Number of contact points per body: {self.num_contact_points}")
        print(f"Contact surface area per body: {self.surface_area} m^2")
        print(f"mu int: {self.cfg.mu_int}")
        print(f"rho c: {self.cfg.rho_c} kg/m^3")
        print(f"mu_surf: {self.cfg.dynamic_friction_coef}")
        print("-"*40)
    
    """
    Initialization.
    """
        
    def create_tensors(self)->None:
        """
        Create buffer tensors.
        """
        self.body_pos = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device)
        self.body_quat = torch.zeros((self.num_envs, self.num_bodies, 4), device=self.device)
        self.body_rot_mat = torch.zeros((self.num_envs, self.num_bodies, 3, 3), device=self.device)
        self.body_lin_vel = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device)
        self.body_ang_vel = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device)

        self.contact_point_local = torch.zeros((self.num_envs, self.num_bodies, self.num_contact_points, 3), device=self.device)
        self.n_dir_local = torch.zeros((self.num_envs, self.num_bodies, self.num_contact_points, 3), device=self.device)
        
        self.contact_point_pos = torch.zeros((self.num_envs, self.num_bodies, self.num_contact_points, 3), device=self.device)
        self.contact_point_lin_vel = torch.zeros((self.num_envs, self.num_bodies, self.num_contact_points, 3), device=self.device)
        self.contact_point_lin_vel_prev = torch.zeros((self.num_envs, self.num_bodies, self.num_contact_points, 3), device=self.device)
        
        # characteristic angles
        self.contact_point_tilt_angle = torch.zeros((self.num_envs, self.num_bodies, self.num_contact_points), device=self.device)
        self.contact_point_intrusion_angle = torch.zeros((self.num_envs, self.num_bodies, self.num_contact_points), device=self.device)
        self.contact_twist_angle = torch.zeros((self.num_envs, self.num_bodies, self.num_contact_points), device=self.device)
        
        # local coordinate
        self.n_dir = torch.zeros((self.num_envs, self.num_bodies, self.num_contact_points, 3), device=self.device) # b
        self.r_dir = torch.zeros((self.num_envs, self.num_bodies, self.num_contact_points, 3), device=self.device) # r
        self.t_dir = torch.zeros((self.num_envs, self.num_bodies, self.num_contact_points, 3), device=self.device) # theta
        self.z_dir = torch.zeros((self.num_envs, self.num_bodies, self.num_contact_points, 3), device=self.device) # z
        self.v_dir = torch.zeros((self.num_envs, self.num_bodies, self.num_contact_points, 3), device=self.device) # v
        
        # contact point force and torque
        self.contact_point_force = torch.zeros((self.num_envs, self.num_bodies, self.num_contact_points, 3), device=self.device)
        self.contact_point_torque = torch.zeros((self.num_envs, self.num_bodies, self.num_contact_points, 3), device=self.device)
        
        # lumped force and torque
        self.contact_force = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device)
        self.contact_torque = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device)
        
        # cache for EMA filtering
        self.alpha_unfiltered = torch.zeros((self.num_envs, self.num_bodies*self.num_contact_points, 3), device=self.device)
        self.alpha_filtered = torch.zeros((self.num_envs, self.num_bodies*self.num_contact_points, 3), device=self.device)
        self.force_gm = torch.zeros((self.num_envs, self.num_bodies*self.num_contact_points), device=self.device)
        self.force_ema = torch.zeros((self.num_envs, self.num_bodies*self.num_contact_points), device=self.device)
        self.tau_r = torch.zeros((self.num_envs, self.num_bodies*self.num_contact_points), device=self.device)
        
        # material parameters
        self.static_friction_coef = self.cfg.static_friction_coef * torch.ones(self.num_envs, device=self.device)
        self.dynamic_friction_coef = self.cfg.dynamic_friction_coef * torch.ones(self.num_envs, device=self.device)
        self.rho_c = self.cfg.rho_c * torch.ones(self.num_envs, device=self.device)
        self.mu_int = self.cfg.mu_int * torch.ones(self.num_envs, device=self.device)
        
        # timestamps
        self._timestamp = torch.zeros(self.num_envs, device=self.device)
        self._timestamp_last_update = torch.zeros(self.num_envs, device=self.device)
        
    def initialize_data(self)->None:
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
        
    def create_contact_points(self)->None:
        """
        Create contact points in contact surface in root body frame (e.g. foot link frame). 
        Fit cuboid to approximate collider. 
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
        
        self.contact_point_local[:, :, :, :] = contact_point_offset
        self.n_dir_local = torch.tensor([0.0, 0.0, -1.0], device=self.device).view(1, 1, 1, 3).repeat(self.num_envs, self.num_bodies, self.num_contact_points, 1)
        
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
        
        # copy intruder kinematic states to buffers
        self.body_pos[:, :, :] = body_pos
        self.body_quat[:, :, :] = body_quat
        self.body_rot_mat[:, :, :, :] = matrix_from_quat(body_quat.view(-1, 4)).view(self.num_envs, self.num_bodies, 3, 3)
        self.body_lin_vel[:, :, :] = body_lin_vel
        self.body_ang_vel[:, :, :] = body_ang_vel
        
        # evaluate contact forces
        self._eval_contacts()
        
        # update timestamp and data
        self._timestamp += self.dt
        self._update_data(torch.arange(self.num_envs, device=self.device))
        self._timestamp_last_update[:] = self._timestamp[:]
    
    def randomize_ground_stiffness(self, env_ids:torch.Tensor, mu_int:torch.Tensor)->None:
        """
        Update ground stiffness (N/m) for each env.
        Implementation is similar to terrain curriculum used in terrain importer class.
        This can be triggered by curriculum manager.
        
        Args:
            env_ids: tensor of env ids to update
            mu_int: tensor of mu_int values (len(env_ids),)
        """
        self.mu_int[env_ids] = mu_int # can be 0.2 - 1.0

    def update_friction_params(self, env_ids:torch.Tensor, static_friction_coef:torch.Tensor, dynamic_friction_coef:torch.Tensor)->None:
        """
        Update friction coefficients for each env.
        This can be triggered by event manager.
        
        Args:
            env_ids: tensor of env ids to update
            static_friction_coef: tensor of static friction coefficients (len(env_ids),)
            dynamic_friction_coef: tensor of dynamic friction coefficients (len(env_ids),)
        """
        self.static_friction_coef[env_ids] = static_friction_coef
        self.dynamic_friction_coef[env_ids] = dynamic_friction_coef

    
    """
    helper functions.
    """
    
    def _update_data(self, env_ids:torch.Tensor)->None:
        """
        Update soft contact data.
        Majority of implementations are from IsaacLab's contact sensor class.
        
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
        
        
    def _eval_contacts(self)->None:
        """
        Update contact points' kinematic states and compute contact forces.
        """
        
        # step1: calculate contact pos, lin vel, and surface normal
        # compute contact points in global frame
        R = self.body_rot_mat.unsqueeze(2)
        p = self.contact_point_local.unsqueeze(-1)
        self.contact_point_pos = self.body_pos.unsqueeze(2) + (R @ p).squeeze(-1)
        
        # compute contact point linear velocity in global frame
        self.contact_point_lin_vel = self.body_lin_vel.unsqueeze(2) + \
            torch.cross(self.body_ang_vel.unsqueeze(2), self.contact_point_pos - self.body_pos.unsqueeze(2), dim=-1)
        
        # compute contact normal
        # currently, we only consider surface normal at the bottom of foot which faces to -z in local frame.
        self.n_dir = (R @ self.n_dir_local.unsqueeze(-1)).squeeze(-1)
            
        # step2: Find local coordinate frame {r, theta, z}
        # compute unit vectors (z, n, v, r, t)
        # see S7 eq.4 from https://www.pnas.org/doi/10.1073/pnas.2214017120
        thresh = 1e-10
        n = self.n_dir.clone()
        z = torch.tensor([0.0, 0.0, 1.0], device=self.device).view(1, 1, 1, 3).repeat(self.num_envs, self.num_bodies, self.num_contact_points, 1)
        v = self.contact_point_lin_vel.clone()/(torch.norm(self.contact_point_lin_vel, dim=-1, keepdim=True) + 1e-6)
        vr = v - (v * z).sum(dim=-1, keepdim=True) * z
        vr_norm = torch.norm(vr, dim=-1, keepdim=True)
        n_rt = self.n_dir - (self.n_dir * z).sum(dim=-1, keepdim=True) * z
        n_rt_norm = torch.norm(n_rt, dim=-1, keepdim=True)
        n_rt_dir = n_rt / (n_rt_norm+1e-6)
        r = vr / (vr_norm+1e-6)
        r[vr_norm.squeeze(-1) < thresh] = n_rt_dir[vr_norm.squeeze(-1) < thresh]
        t = torch.cross(z, r, dim=-1)
        # copy to buffers
        self.r_dir = r
        self.t_dir = t
        self.z_dir = z
        self.v_dir = v

        # step3: compute characteristic angles
        
        # compute contact point velocity angle (gamma)
        # see S7 eq.6 from https://www.pnas.org/doi/10.1073/pnas.2214017120
        vdotr = (v * r).sum(dim=-1)
        vdotz = (v * z).sum(dim=-1)
        self.contact_point_intrusion_angle = torch.acos(vdotr) * ((vdotz<0).float() - (vdotz>=0).float())
        self.contact_point_intrusion_angle = torch.nan_to_num(self.contact_point_intrusion_angle, nan=0.0, posinf=0.0, neginf=0.0)
        
        # compute contact point tilt angle (beta)
        # see S7 eq.5 from https://www.pnas.org/doi/10.1073/pnas.2214017120
        ndotr = (n * r).sum(dim=-1)
        ndott = (n * t).sum(dim=-1)
        ndotz = (n * z).sum(dim=-1)
        n_rtz = torch.cat([ndotr.unsqueeze(-1), ndott.unsqueeze(-1), ndotz.unsqueeze(-1)], dim=-1)
        reflection_matrix = (1 - 2 * (ndotr<0).float()).unsqueeze(-1)
        n_rtz = n_rtz * reflection_matrix
        self.contact_point_tilt_angle = -torch.acos(n_rtz[:, :, :, 2]) + torch.pi * (n_rtz[:, :, :, 2] < 0).float()
        self.contact_point_tilt_angle = torch.nan_to_num(self.contact_point_tilt_angle, nan=0.0, posinf=0.0, neginf=0.0)
        
        # compute twist angle (psi)
        # see S7 eq.7 from https://www.pnas.org/doi/10.1073/pnas.2214017120
        n_rt_matrix = n_rtz.clone() # (num_envs, num_bodies, num_contact_points, 3)
        n_rt_matrix_norm = torch.norm(n_rt_matrix, dim=-1)
        n_rt_matrix = n_rt_matrix / (n_rt_matrix_norm.unsqueeze(-1)+1e-6)
        n_rt_matrix[n_rt_matrix_norm < thresh, :] = r[n_rt_matrix_norm < thresh, :]
        self.contact_twist_angle = torch.atan2(n_rt_matrix[:, :, :, 1], n_rt_matrix[:, :, :, 0])
        self.contact_twist_angle = torch.nan_to_num(self.contact_twist_angle, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Force needs to be applifed to opposite direction of velocity
        # calculated sign of force in theta direction.
        sign_ft = 1 - 2 * (n_rt_matrix[:, :, :, 1] < 0).float()
        
        # step4: compute resistive force alpha
        # see eq.1 in https://www.pnas.org/doi/10.1073/pnas.2214017120
        force = self._get_resistive_force(
            self.contact_point_pos.reshape(self.num_envs, -1, 3),
            self.contact_point_lin_vel.reshape(self.num_envs, -1, 3),
            self.contact_point_lin_vel_prev.reshape(self.num_envs, -1, 3),
            self.contact_point_tilt_angle.reshape(self.num_envs, -1),
            self.contact_point_intrusion_angle.reshape(self.num_envs, -1),
            self.contact_twist_angle.reshape(self.num_envs, -1),
            sign_ft.reshape(self.num_envs, -1),
        )
        force = force.reshape(self.num_envs, self.num_bodies, self.num_contact_points, 3)
        torque = torch.cross((self.contact_point_pos - self.body_pos.unsqueeze(2)), force, dim=-1)
        self.contact_point_force[:, :, :, :] = force
        self.contact_point_torque[:, :, :, :] = torque
        
        # net force and torque
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
        gamma:torch.Tensor, 
        psi:torch.Tensor,
        sign_fy:torch.Tensor, 
        )->torch.Tensor:
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
        intrusion_mask = ((self.n_dir * self.v_dir).sum(dim=-1) >= 0).reshape(self.num_envs, -1) # intrusion only when normal and velocity direction has obtuse angle
        
        # NOTE: orthogonal base is {r, t, z} here.
        alpha_r, alpha_t, alpha_z = self._compute_elementary_force(beta, gamma, psi) # get RFT force
        alpha_gen = torch.cat([
            alpha_r.unsqueeze(-1), 
            alpha_t.unsqueeze(-1), 
            alpha_z.unsqueeze(-1)], 
            dim=-1) # (num_envs, num_bodies*num_contact_points, 3)
        
        # media specific params (see S4)
        g = 9.81 # m/s^2
        xi = self.rho_c * g * (894 * (self.mu_int **3) - 386 * (self.mu_int **2) + 89 * self.mu_int) # N/m^3

        # alpha_gen -> alpha by multiplying media specific parameter xi
        alpha = xi[:, None, None] * alpha_gen

        # norm vector in {r, t, z} system
        n = torch.cat([
            (torch.sin(beta)*torch.cos(psi)).unsqueeze(-1),
            (torch.sin(beta)*torch.sin(psi)).unsqueeze(-1),
            -(torch.cos(beta)).unsqueeze(-1)
        ], dim=-1)
        
        # retrieve normal and tangential components of alpha.
        alpha_n = (alpha * n).sum(dim=-1, keepdim=True) * n # normal component (num_envs, num_bodies*num_contact_points, 3)
        alpha_n_norm = torch.norm(alpha_n, dim=-1) # (num_envs, num_bodies*num_contact_points)
        alpha_tan = alpha - alpha_n # tangential component (num_envs, num_bodies*num_contact_points, 3)
        alpha_tan_norm = torch.norm(alpha_tan, dim=-1) # (num_envs, num_bodies*num_contact_points)

        # friction cone check
        cone_coef = torch.minimum(torch.ones_like(alpha_tan_norm), 
                                  (self.dynamic_friction_coef[:, None] * alpha_n_norm) / (alpha_tan_norm + 1e-6)).unsqueeze(-1)
        alpha = alpha_n + cone_coef * alpha_tan

        # NOTE: optional EMA filter
        if self.enable_ema_filter:
            self.alpha_unfiltered = alpha.clone()
            self._ema_filtering(foot_velocity, foot_velocity_prev, depth)
            alpha = self.alpha_filtered.clone()

        # compute force by mulitplying depth, area, stiffness
        # force_vec = alpha * depth[:, :, None] * dA * is_contact[:, :, None] * intrusion_mask[:, :, None]
        force_vec = alpha * depth[:, :, None] * dA * is_contact[:, :, None] # more consistent with original paper
        
        # NOTE: orthogonal base is {x, y, z} here.
        kd = 0.0
        z_damp = kd * foot_velocity[:, :, 2:3] * is_contact[:, :, None]
        force = force_vec[:, :, 0:1] * self.r_dir.reshape(self.num_envs, -1, 3) + \
                force_vec[:, :, 1:2] * self.t_dir.reshape(self.num_envs, -1, 3) * sign_fy.unsqueeze(-1) + \
                (force_vec[:, :, 2:3] - z_damp) * self.z_dir.reshape(self.num_envs, -1, 3)

        return force
    
    def _compute_elementary_force(self, beta:torch.Tensor, gamma:torch.Tensor, psi: torch.Tensor)->Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute elementary force per foot using Fourier series expansion. 
        See the original paper. 
        
        Args: 
            beta: intrusion angle (pitch) of contact point. (num_envs, num_bodies*num_contact_points)
            gamma: intrusion direction angle of contact point. (num_envs, num_bodies*num_contact_points)
            psi: twist angle of contact point. (num_envs, num_bodies*num_contact_points)
        Returns:
            alpha_r: elementary force coefficient in r direction. (num_envs, num_bodies*num_contact_points)
            alpha_t: elementary force coefficient in theta direction. (num_envs, num_bodies*num_contact_points)
            alpha_z: elementary force coefficient in z direction. (num_envs, num_bodies*num_contact_points)
        """
        p1 = torch.sin(gamma)
        p2 = torch.cos(beta)
        p3 = torch.cos(psi)*torch.cos(gamma)*torch.sin(beta) + torch.sin(gamma)*torch.cos(beta)
        
        base = [torch.ones_like(p1), p1, p2, p3, p1**2, p2**2, p3**2, p1*p2, p2*p3, p3*p1, p1**3, p2**3, p3**3, p1*(p2)**2, p2*(p1)**2, p2*(p3)**2, p3*(p2)**2, p3*(p1)**2, p1*(p3)**2, p1*p2*p3]

        f1 = torch.stack([self.cfg.coef_1[i] * base[i] for i in range(20)]).sum(dim=0)
        f2 = torch.stack([self.cfg.coef_2[i] * base[i] for i in range(20)]).sum(dim=0)
        f3 = torch.stack([self.cfg.coef_3[i] * base[i] for i in range(20)]).sum(dim=0)
        
        alpha_r = f1 * torch.sin(beta) * torch.cos(psi) + f2 * torch.cos(gamma)
        alpha_t = f1 * torch.sin(beta) * torch.sin(psi)
        alpha_z = -f1 * torch.cos(beta) - f2 * torch.sin(gamma) - f3
        
        return alpha_r, alpha_t, alpha_z
        

    def _ema_filtering(self, velocity:torch.Tensor, velocity_prev:torch.Tensor, depth:torch.Tensor):
        """
        Exponential moving filtering
        See KAIST science robotics supplementary material S12.
        
        Args:
            velocity: contact point velocity in global frame. (num_envs, num_bodies*num_contact_points, 3)
            velocity_prev: contact point velocity at previous time step in global frame. (num_envs, num_bodies*num_contact_points, 3)
            depth: contact point penetration depth. (num_envs, num_bodies*num_contact_points)
        """
        coef = 0.8 # smoothing coefficient
        increment_mask = (velocity * velocity_prev).sum(dim=-1) < 0
        # increment_mask = velocity[:,:, -1]*velocity_prev[:, :, -1] < 0
        tau_r_boundary = self.tau_r < 1
        depth_mask = depth > 0
        mask = increment_mask & tau_r_boundary
        self.tau_r[mask] += self.c_r
        self.tau_r[~depth_mask] = 0.0

        self.alpha_filtered[depth_mask] = (1-coef*self.tau_r[depth_mask]).unsqueeze(-1) * self.alpha_unfiltered[depth_mask] + \
              coef * self.tau_r[depth_mask].unsqueeze(-1) * self.alpha_filtered[depth_mask]
        self.alpha_filtered[~depth_mask] = 0.0

    """
    reset.
    """
    
    def reset(self, env_ids:torch.Tensor):
        """
        Reset internal states for given env ids.
        Args:
            env_ids: tensor of env ids to reset
        """
        self.alpha_filtered[env_ids] = 0.0
        self.alpha_unfiltered[env_ids] = 0.0
        self.tau_r[env_ids] = 0.0
        self.contact_point_lin_vel_prev[env_ids] = 0.0


if __name__ == "__main__":
    material_cfg = PoppySeedLPCfg()
    num_envs = 10 
    num_bodies = 2
    num_contact_points = 100
    device = torch.device('cuda:0')
    rft = RFT_2D(material_cfg=material_cfg, num_envs=num_envs, num_bodies=num_bodies, device=device, dt=1/200)
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