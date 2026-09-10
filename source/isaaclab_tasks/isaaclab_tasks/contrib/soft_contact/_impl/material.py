# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING

from isaaclab.utils.configclass import configclass


"""
2D RFT material parameters.
"""


@configclass
class MaterialCfg:
    """
    Material configuration for soft contact model.

    A00 - D10: quasistatic RFT Fourier coefficients (dimensionless, from Li et al. 2013).
    rho_c, mu_int: quasistatic stiffness parameters (same as 3D RFT).
      Effective stiffness xi = rho_c * g * (894*mu_int^3 - 386*mu_int^2 + 89*mu_int).
    lam, rho: dynamic RFT (DRFT) inertial parameters.
    static_friction_coef, dynamic_friction_coef: friction coefficients.
    kf: tangential force model parameter.
    kh, beta_d, bh: horizontal stroke resistive force model parameters.
    """

    # quasistatic RFT fourier coefficients (dimensionless)
    A00: float = MISSING  # type: ignore
    A10: float = MISSING  # type: ignore
    B11: float = MISSING  # type: ignore
    B01: float = MISSING  # type: ignore
    B_11: float = MISSING  # type: ignore
    C11: float = MISSING  # type: ignore
    C01: float = MISSING  # type: ignore
    C_11: float = MISSING  # type: ignore
    D10: float = MISSING  # type: ignore

    # quasistatic stiffness parameters (same parameterisation as 3D RFT)
    rho_c: float = MISSING  # type: ignore  # critical media density (kg/m^3)
    mu_int: float = MISSING  # type: ignore  # media internal friction coefficient

    # dynamic RFT parameters
    lam: float = MISSING  # type: ignore
    rho: float = MISSING  # type: ignore

    # material properties
    static_friction_coef: float = MISSING  # type: ignore # TODO remove
    dynamic_friction_coef: float = MISSING  # type: ignore

    # coulomb friction tangential force model parameters
    kf: float = MISSING  # type: ignore

    # horizontal stroke resistive force model parameters
    kh: float = MISSING  # type: ignore
    beta_d: float = MISSING  # type: ignore
    bh: float = MISSING  # type: ignore


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

    # quasistatic stiffness (same parameterisation as 3D RFT)
    rho_c: float = 638.0  # bulk density of poppy seeds (kg/m^3)
    mu_int: float = 0.3  # internal friction coefficient

    # dynamic RFT parameters
    lam: float = 1.0
    rho: float = 638.0 * (1e-6)  # kg/mm^3 to kg/cm^3

    # material properties
    static_friction_coef: float = 1.0  # TODO remove
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

    # quasistatic stiffness (same parameterisation as 3D RFT)
    rho_c: float = 638.0  # bulk density of poppy seeds (kg/m^3)
    mu_int: float = 0.3  # internal friction coefficient

    lam: float = 1.0
    rho: float = 638.0 * (1e-6)  # kg/mm^3 to kg/cm^3

    static_friction_coef: float = 1.0  # TODO remove
    dynamic_friction_coef: float = 0.5

    kf: float = 10.0 # tangential force gain (ft = -k*vt)

    # horizontal stroke resistive force model parameters
    kh: float = 50.0
    beta_d: float = 0.5
    bh: float = 1.0

@configclass
class GenericMaterialCfg(MaterialCfg):
    A00: float = 0.206
    A10: float = 0.169
    B11: float = 0.212
    B01: float = 0.358
    B_11: float = 0.055
    C11: float = -0.124
    C01: float = 0.253
    C_11: float = 0.007
    D10: float = 0.088

    # quasistatic stiffness (same parameterisation as 3D RFT)
    rho_c: float = 638.0  # bulk density of poppy seeds (kg/m^3)
    mu_int: float = 0.3  # internal friction coefficient

    # dynamic inertial correction parameters
    lam: float = 1.0
    rho: float = 638.0 * (1e-6)  # kg/mm^3 to kg/cm^3

    static_friction_coef: float = 1.0  # TODO remove
    dynamic_friction_coef: float = 0.5

    kf: float = 100.0 # tangential force gain (ft = -k*vt)

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
        0.00212,
        -0.02320,
        -0.20890,
        -0.43083,
        -0.00259,
        0.48872,
        -0.00415,
        0.07204,
        -0.02750,
        -0.08772,
        0.01992,
        -0.45961,
        0.40799,
        -0.10107,
        -0.06576,
        0.05664,
        -0.09269,
        0.01892,
        0.01033,
        0.15120,
    ]
    # c2^k
    coef_2: list[float] = [
        -0.06796,
        -0.10941,
        0.04725,
        -0.06914,
        -0.05835,
        -0.65880,
        -0.11985,
        -0.25739,
        -0.26834,
        0.02692,
        -0.00736,
        0.63758,
        0.08997,
        0.21069,
        0.04748,
        0.20406,
        0.18519,
        0.04934,
        0.13527,
        -0.33207,
    ]
    # c3^k
    coef_3: list[float] = [
        -0.02634,
        -0.03436,
        0.45256,
        0.00835,
        0.02553,
        -1.31290,
        -0.05532,
        0.06790,
        -0.16404,
        0.02287,
        0.02927,
        0.95406,
        -0.00131,
        -0.11028,
        0.01487,
        -0.20770,
        0.10911,
        -0.04097,
        0.07881,
        -0.27519,
    ]

    # 3D RFT media specific properties
    static_friction_coef: float = 1.0  # TODO remove
    dynamic_friction_coef: float = 0.3
    mu_int: float = 0.3  # media internal friction coefficient
    rho_c: float = 3000.0  # critical media density (effective media density = packing fraction * grain density)
    kf: float = 10.0 # tangential force gain (ft = -k*vt)

"""
ConeDRFT (granular jammed-cone) contact model parameters.
"""


@configclass
class ConeDRFTCfg:
    """
    Material configuration for the ConeDRFT (granular jammed-cone) soft contact model.

    Implements the model of Choi et al., "Learning quadrupedal locomotion on deformable
    terrain", Sci. Robotics 2023, supplementary sections S11-S16. The foot is treated as a
    single point intruder; the vertical reaction is computed in closed form from the cone
    integrals as a function of penetration depth z, rate z_dot and acceleration z_ddot:

        F_GM = sigma_flat * I_flat(z) + sigma_cone * (pi r_h^2 z - I_flat) / cos(theta)   # quasistatic (S15)
             + c_d * M * A_flat(z) * z_dot^2                                              # inertial drag
             + M * I_flat(z) * z_ddot                                                     # added mass
        with  M = c_g * phi * rho * nu,   k = nu / tan(theta).

    sigma_flat, sigma_cone: flat- and conical-surface resistive stresses (N/m^3). These are the
        primary depth-dependent stiffness terms and the main quantities to domain-randomize.
    nu:    recruitment rate (cone growth speed).
    theta: shear band angle (rad).
    phi:   packing density (volume fraction).
    rho:   grain density (kg/m^3).
    c_g:   surrounding-mass scaling factor (scales added mass).
    c_d:   inertial drag scaling factor.
    eps_f: plastic-deformation offset (m); force turns off once the foot lifts past this from z_max.
    enable_added_mass: include the M * I_flat * z_ddot term (uses finite-difference acceleration).
    enable_ema_filter: apply the anti-drift EMA on the normal force (S16).
    dynamic_friction_coef, kf: Coulomb tangential model, ft = min(mu * fz, kf * vt).

    Note: r_h (hydraulic radius of the foot cross-section) is NOT set here — it is derived from
    the collider footprint in the model's __init__ (r_h = L*W/(L+W) for a box; r = radius for a sphere).
    """

    # depth-dependent stiffness (randomizable)
    sigma_flat: float = MISSING  # type: ignore  # flat-surface resistive stress (N/m^3)
    sigma_cone: float = MISSING  # type: ignore  # conical-surface resistive stress (N/m^3)

    # cone geometry / granular material
    nu: float = MISSING     # type: ignore  # recruitment rate
    theta: float = MISSING  # type: ignore  # shear band angle (rad)
    phi: float = MISSING    # type: ignore  # packing ratio (volume fraction)
    rho: float = MISSING    # type: ignore  # grain density (kg/m^3)
    c_g: float = MISSING    # type: ignore  # surrounding-mass scaling factor
    c_d: float = MISSING    # type: ignore  # inertial drag scaling factor

    # plastic deformation / numerical
    eps_f: float = 1.0e-4
    enable_added_mass: bool = False
    enable_ema_filter: bool = True

    # friction
    static_friction_coef: float = MISSING   # type: ignore # TODO remove
    dynamic_friction_coef: float = MISSING   # type: ignore
    kf: float = MISSING  # type: ignore # tangential force gain (ft = -k*vt)


@configclass
class DefaultConeDRFTCfg(ConeDRFTCfg):
    # depth-dependent stiffness — default terrain parameters (poppy-seed-like, S-material range)
    sigma_flat: float = 5.0e6   # N/m^3  (paper randomizes ~[1, 10] MN/m^3)
    sigma_cone: float = 0.4e6   # N/m^3  (paper randomizes ~[0.15, 0.6] MN/m^3)

    # cone geometry / granular material
    nu: float = 1.0
    theta: float = 0.5236       # 30 deg shear band angle (rad)
    phi: float = 0.6            # packing ratio (volume fraction)
    rho: float = 638.0          # grain density (kg/m^3), poppy seeds
    c_g: float = 1.0            # surrounding mass scaling factor
    c_d: float = 1.0            # inertial drag scaling factor

    eps_f: float = 1.0e-4
    enable_added_mass: bool = False
    enable_ema_filter: bool = True

    static_friction_coef: float = 1.0
    dynamic_friction_coef: float = 0.5
    kf: float = 10.0 # tangential force gain (ft = -k*vt)


"""
Spring-damper contact model parameters.
"""


@configclass
class SpringDamperCfg:
    """
    Material configuration for spring-damper contact model.

    Normal force per contact point: fz = max((k * depth - b * vn) * dA, 0)
    Tangential force: ft = min(mu * fz, kf * vt)

    k: spring stiffness density (N/m^3) — scales with contact area element dA.
    b: damping density (N*s/m^3) — scales with contact area element dA.
    dynamic_friction_coef: Coulomb friction coefficient.
    kf: tangential viscous cap coefficient (N*s/m).
    """

    k: float = MISSING  # type: ignore  # spring stiffness density (N/m^3)
    b: float = MISSING  # type: ignore  # damping density (N*s/m^3)
    static_friction_coef: float = MISSING  # type: ignore
    dynamic_friction_coef: float = MISSING  # type: ignore
    kf: float = MISSING  # type: ignore  # tangential viscous cap (N*s/m)


@configclass
class DefaultSpringDamperCfg(SpringDamperCfg):
    k: float = 1.0e5   # N/m^3  (soft ground, similar order to granular media)
    b: float = 1.0e3   # N*s/m^3
    static_friction_coef: float = 1.0
    dynamic_friction_coef: float = 0.5
    kf: float = 10.0   # N*s/m
