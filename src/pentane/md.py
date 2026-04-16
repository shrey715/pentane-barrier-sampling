"""
NVT Molecular Dynamics - Nose-Hoover Thermostatted Dihedral Dynamics
=====================================================================

Implements a 1D NVT molecular dynamics simulation for the backbone
dihedral angle of n-pentane, using the Velocity Verlet integrator
coupled with a Nose-Hoover thermostat.

Physical model
--------------
The dihedral phi is treated as a single rotational degree of freedom with
an effective moment of inertia I_eff = m_CH3 * r_perp^2, where r_perp is the
perpendicular distance from the C4 atom to the C2-C3 rotation axis.

Equations of motion (Nose-Hoover NVT):

    I_eff * phi_ddot = -dU/dphi - I_eff * xi * phi_dot

    xi_dot = (1/tau_T^2) * (T_inst/T - 1)

where xi is the thermostat friction variable, tau_T is the coupling time,
and T_inst = I_eff * omega^2 / k_B is the instantaneous temperature.

Integration scheme
------------------
Modified Velocity Verlet with half-step thermostat updates:
  1. Half-step velocity update (force + friction)
  2. Full-step position update
  3. Recompute force at new position
  4. Half-step thermostat update
  5. Half-step velocity update (force + friction)
  6. Half-step thermostat update
"""

import numpy as np

from pentane.forcefield import (
    I_EFF,
    KB_KJ_MOL,
    UNIT_CONV,
    torsion_force,
)


def nvt_md_simulation(
    T: float,
    n_steps: int = 200_000,
    phi_init: float = np.radians(180.0),
    dt: float = 0.002,
    tau_T: float = 0.5,
    seed: int = 42,
) -> np.ndarray:
    """
    Run an NVT molecular dynamics simulation for the dihedral angle.

    Parameters
    ----------
    T : float
        Target temperature [K].
    n_steps : int
        Number of integration time steps.
    phi_init : float
        Initial dihedral angle [radians].
    dt : float
        Integration timestep [ps].
    tau_T : float
        Nose-Hoover coupling time [ps].
    seed : int
        Random number generator seed for initial velocity sampling.

    Returns
    -------
    dihedrals : ndarray, shape (n_steps,)
        Dihedral angle trajectory [radians].

    Notes
    -----
    The torque is converted from Kelvin/radian to internal MD units
    (amu*Ang^2/ps^2) via:
        F_internal = tau_K * k_B[kJ/(mol*K)] / unit_conv[kJ*mol^-1 per amu*Ang^2*ps^-2]

    Note the sign: torsion_force() returns -dU/dphi (see forcefield module),
    so we multiply by -1 here to get the physical torque in MD units.
    """
    rng = np.random.default_rng(seed)

    phi = float(phi_init)

    # Initialize angular velocity from Maxwell-Boltzmann distribution
    omega_std = np.sqrt(KB_KJ_MOL * T / (I_EFF * UNIT_CONV))
    omega = rng.normal(0.0, omega_std)

    # Nose-Hoover friction variable
    xi = 0.0

    dihedrals = np.zeros(n_steps)

    for i in range(n_steps):
        # --- Half-step velocity update ---
        # torsion_force returns -dU/dphi in K/rad; convert to amu*Ang^2/ps^2
        F_ang = -torsion_force(phi) * KB_KJ_MOL / UNIT_CONV
        omega += 0.5 * dt * (F_ang / I_EFF - xi * omega)

        # --- Full-step position update ---
        phi = (phi + dt * omega + np.pi) % (2.0 * np.pi) - np.pi

        # --- Recompute force at new position ---
        F_ang = -torsion_force(phi) * KB_KJ_MOL / UNIT_CONV

        # --- First half-step thermostat update ---
        KE = 0.5 * I_EFF * omega**2
        T_inst = KE * UNIT_CONV / (0.5 * KB_KJ_MOL)
        xi += 0.5 * dt * (T_inst / T - 1.0) / tau_T**2

        # --- Second half-step velocity update ---
        omega += 0.5 * dt * (F_ang / I_EFF - xi * omega)

        # --- Second half-step thermostat update ---
        KE = 0.5 * I_EFF * omega**2
        T_inst = KE * UNIT_CONV / (0.5 * KB_KJ_MOL)
        xi += 0.5 * dt * (T_inst / T - 1.0) / tau_T**2

        dihedrals[i] = phi

    print(f"  MD @ {T:6.1f} K: {n_steps} steps, dt = {dt} ps, "
          f"tau_T = {tau_T} ps")

    return dihedrals
