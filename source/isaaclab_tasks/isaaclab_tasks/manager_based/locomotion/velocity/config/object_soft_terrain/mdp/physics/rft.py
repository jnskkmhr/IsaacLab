from dataclasses import MISSING
import math
import torch
from isaaclab.utils import configclass
from isaaclab.utils.math import matrix_from_quat, euler_xyz_from_quat, matrix_from_euler
from typing import Tuple

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
    hardness: float = MISSING # type: ignore # scaling factor applied to resulting force per area (alpha)

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
    hardness: float = 1.0

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
    hardness: float = 3.0


"""
2D RFT class.
"""

class RFT_EMF:
    def __init__(self, 
                 cfg: MaterialCfg, 
                 num_envs: int,
                 num_bodies: int, 
                 num_contact_points: int,
                 device: torch.device,
                 static_friction_coef: float=1.0, 
                 dynamic_friction_coef: float=0.5, 
                 ):
        """
        Resistive Force Theory based soft terrain contact solver.
        https://www.science.org/doi/10.1126/science.1229163
        """
        
        self.cfg = cfg
        self.num_envs = num_envs
        self.num_bodies = num_bodies
        self.num_contact_points = num_contact_points
        self.device = device
        
        self.contact_edge_x = (-0.06, 0.14)
        self.contact_edge_y = (-0.03, 0.03)
        self.surface_area = (self.contact_edge_x[1]-self.contact_edge_x[0])*(self.contact_edge_y[1]-self.contact_edge_y[0])
        
        self.c_r = 0.05 # 100/f (e.g. f=2000hz -> 0.05)
        self.static_friction_coef = static_friction_coef
        self.dynamic_friction_coef = dynamic_friction_coef
        
        self.create_tensors()
        self.create_contact_points()
        
    def create_tensors(self):
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
        
        
    """
    operations.
    """
    
    def update(self, body_pos:torch.Tensor, body_quat:torch.Tensor, body_lin_vel:torch.Tensor, body_ang_vel:torch.Tensor):
        """
        Args:
        body_pos: (num_envs, num_bodies, 3)
        body_quat: (num_envs, num_bodies, 4)
        body_lin_vel: (num_envs, num_bodies, 3)
        body_ang_vel: (num_envs, num_bodies, 3)
        """
        self.body_pos[:, :, :] = body_pos
        self.body_quat[:, :, :] = body_quat
        self.body_rot_mat[:, :, :, :] = matrix_from_quat(body_quat.view(-1, 4)).view(self.num_envs, self.num_bodies, 3, 3)
        
        roll, pitch, yaw = euler_xyz_from_quat(body_quat.view(-1, 4))
        self.body_rot_mat_roll_yaw[:, :, :, :] = \
            matrix_from_euler(
                torch.stack(
                (roll, torch.zeros_like(pitch), yaw), dim=-1),
                convention="XYZ").view(self.num_envs, self.num_bodies, 3, 3)

        self.body_lin_vel[:, :, :] = body_lin_vel
        self.body_ang_vel[:, :, :] = body_ang_vel

        self.process_contact_points()
        self.compute_force()

    
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
        

    
    """
    helper functions.
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
            -0.035/2 * torch.ones_like(contact_point_offset_x).flatten()
            ), dim=-1)
        contact_point_offset = contact_point_offset.unsqueeze(0).unsqueeze(1).repeat(self.num_envs, self.num_bodies, 1, 1) # (num_envs, num_bodies, num_contact_points, 3)
        self.contact_point_offset[:, :, :, :] = contact_point_offset
        
    def process_contact_points(self)->None:
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
        
    def compute_force(self)->None:
        """
        Compute contact forces and torques.
        """
        f_normal = self.get_resistive_force(
            self.contact_point_pos.reshape(self.num_envs, -1, 3),
            self.contact_point_lin_vel.reshape(self.num_envs, -1, 3),
            self.contact_point_lin_vel_prev.reshape(self.num_envs, -1, 3),
            -self.contact_point_euler.reshape(self.num_envs, -1, 3)[..., 1], # beta = - pitch (physics ppl's coordinate is different from robotics)
            self.contact_point_intrusion_angle.reshape(self.num_envs, -1), # gamma
        )
        
        f_tangential = self.get_coulomb_friction_force(
            self.contact_point_lin_vel.reshape(self.num_envs, -1, 3),
            f_normal[:, :, 2]
        )

        force = (f_normal+f_tangential).reshape(self.num_envs, self.num_bodies, self.num_contact_points, 3)
        torque = torch.cross((self.contact_point_pos - self.body_pos.unsqueeze(2)), force, dim=-1)
        
        self.contact_point_force[:, :, :, :] = force
        self.contact_point_torque[:, :, :, :] = torque
        
        # lumped force and torque
        self.contact_force[:, :, :] = torch.sum(force, dim=2) # (num_envs, num_bodies, 3)
        self.contact_torque[:, :, :] = torch.sum(torque, dim=2) # (num_envs, num_bodies, 3)
        
        # update velocity history
        self.contact_point_lin_vel_prev[:, :, :, :] = self.contact_point_lin_vel[:, :, :, :]
    
    def get_resistive_force(
        self, 
        foot_pos:torch.Tensor, 
        foot_velocity:torch.Tensor, 
        foot_velocity_prev:torch.Tensor, 
        beta:torch.Tensor, 
        gamma:torch.Tensor)->torch.Tensor:
        """
        Compute normal force using RFT z component.
        Computed force is in simulation global frame.
        """
        dA = self.surface_area/self.num_contact_points
        depth = -foot_pos[:, :, -1]
        
        alpha_x, alpha_z = self.compute_elementary_force(beta, gamma) # get RFT force
        alpha_z = self.cfg.hardness * alpha_z
        depth_mask = depth > 0 # apply resistive force only when foot is penetrating
        self.force_gm = alpha_z * depth * dA * depth_mask * (1e6) #m^3 to cm^3 since alpha is N/cm^3
        self.emf_filtering(foot_velocity, foot_velocity_prev, depth)
        
        force_normal = torch.zeros((self.num_envs, self.num_bodies*self.num_contact_points, 3), device=self.device)
        force_normal[:, :, -1] = self.force_ema
        
        # rotate normal force back to simulation global frame
        force_normal = force_normal.reshape(self.num_envs, self.num_bodies, self.num_contact_points, 3)
        R = self.body_rot_mat_roll_yaw.unsqueeze(2)
        p = force_normal.unsqueeze(-1)
        force_normal = (R @ p).squeeze(-1).reshape(self.num_envs, -1, 3)
        
        # add dynamic inertial term in DRFT
        lam = 1.0
        rho = 638.0 * (1e-6) # kg/mm^3 to kg/cm^3
        force_normal[:, :, 2] = force_normal[:, :, 2] + lam * rho * foot_velocity[:, :, 2]**2

        return force_normal
    
    def get_coulomb_friction_force(self, foot_velocity:torch.Tensor, fz:torch.Tensor)->torch.Tensor:
        """
        Get tangential force using Coulomb friction model. 
        Computed force is in simulation global frame.
        Idea is from https://iscicra25.github.io/papers/2025-Lee-4_Soft_Contact_Model_for_Robus.pdf
        """
        # combines Coulomb friction and Stribeck friction model
        vt = torch.sqrt(foot_velocity[:, :, 0]**2 + foot_velocity[:, :, 1]**2)
        vt_unit_vec = foot_velocity[:, :, :2]/(vt.unsqueeze(2) + 1e-6)
        # v_cf = 0.05 # Coulomb friction velocity threshold
        # v_st = 0.01 # Stribeck friction velocity threshold
        # friction_force = (self.dynamic_friction_coef * torch.tanh(vt/v_cf) + \
        #     math.sqrt(2*math.e) * (self.static_friction_coef - self.dynamic_friction_coef) * torch.exp(-(vt/v_st)**2) * (vt/v_st)) * torch.abs(fz)
        
        # Coulomb friction under slip
        friction_force = self.dynamic_friction_coef * torch.abs(fz)
        friction_force_vec = -friction_force.unsqueeze(2) * vt_unit_vec
        
        tangential_force = torch.zeros((self.num_envs, self.num_bodies*self.num_contact_points, 3), device=self.device)
        tangential_force[:, :, :2] = friction_force_vec
        return tangential_force

    def emf_filtering(self, velocity:torch.Tensor, velocity_prev:torch.Tensor, depth:torch.Tensor):
        """
        Exponential moving filtering
        See KAIST science robotics supplementary material S12.
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

    def compute_elementary_force(self, beta:torch.Tensor, gamma:torch.Tensor)->Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute elementary force per foot using Fourier series expansion. 
        See the original paper. 
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
    
    def reset(self, env_ids:torch.Tensor):
        self.force_gm[env_ids] = 0.0
        self.force_ema[env_ids] = 0.0
        self.tau_r[env_ids] = 0.0
        self.contact_point_lin_vel_prev[env_ids] = 0.0


if __name__ == "__main__":
    material_cfg = PoppySeedLPCfg()
    num_envs = 10 
    num_bodies = 2
    num_contact_points = 100
    device = torch.device('cuda:0')
    rft = RFT_EMF(cfg=material_cfg, num_envs=num_envs, num_bodies=num_bodies, num_contact_points=num_contact_points, device=device)
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