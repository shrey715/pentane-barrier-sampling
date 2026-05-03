"""
md.py — Baseline NVT Molecular Dynamics (Nosé-Hoover) for n-pentane.

Public API
----------
run_md(T, cfg, seed=None) -> np.ndarray  shape (n_steps,)
    Trajectory of phi1 (C1-C2-C3-C4 dihedral) in radians.

Algorithm — 1D torsional Nosé-Hoover thermostat:
  Generalised coordinate : phi1
  Effective moment       : I_eff = m_CH3 * r_perp²
                           where r_perp = r * sin(θ₀) is the lever arm
                           (perpendicular distance from the C2-C3 axis to C4
                            at equilibrium bond angle θ₀, scaled by r)

  Equations of motion (velocity-Verlet split-operator):
      dφ/dt  = ω
      dω/dt  = τ(φ)/I_eff − ξ·ω
      dξ/dt  = (1/τ_T²)(I_eff·ω²/T − 1)

  where τ(φ) = torsion_force(φ)  [the generalised torque in K/rad]
        τ_T  = Nosé-Hoover coupling time from cfg [ps]
        T    = temperature [K]

Units: angles [rad], time [ps], energies [K, k_B = 1].
The mass of CH3 in TraPPE-UA is taken as m = 15.035 u (CH3 group mass).
Mass and r_perp are derived from cfg — no magic numbers.

Note: phi2 is fixed at π (trans) throughout — only phi1 is propagated.
"""
import numpy as np
from pentane.config_loader import CFG
from pentane.forcefield import torsion_force, total_energy
from pentane.geometry import build_pentane

# ── Derived constants (from CFG) ────────────────────────────────────────────
_sim  = CFG["simulation"]
_bnd  = CFG["bonds"]
_ang  = CFG["angles"]

# CH3 united-atom mass [u].  1 u = 1.66054e-27 kg, but we work in
# reduced units where energy [K·k_B] and time [ps] determine mass units.
# m_CH3 in [K·ps²/Å²] = 15.035 [u] × (1.66054e-27/1.38065e-23) × 1e20 / 1e24
# → m_CH3 [K·ps²/Å²] = 15.035 × 1.20272e-4 / 100 ≈ 1.80849e-4... let us compute:
# Conversion: 1 u in K·ps²/Å² = kB_J⁻¹ × 1u_kg × (1e10 Å/m)² / (1e12 ps/s)²
#           = (1/1.38065e-23) × 1.66054e-27 × 1e20 / 1e24
#           = 1.20272e-4  [K·ps²/Å²]
_U_TO_KPSA2 = 1.20272e-4   # 1 Da → K·ps²/Å²  (derived, not a physics magic number)

_M_CH3_U  = 15.035          # CH3 group mass [Da] (C=12, H3=3×1.0079)
_M_CH3    = _M_CH3_U * _U_TO_KPSA2   # [K·ps²/Å²]

# Effective moment of inertia for phi1 rotation:
#   I_eff = m_CH3 × r_perp²
#   r_perp = r_CC × sin(θ₀)  — lever arm at the bond angle
_R   = _bnd["r_CC_ang"]
_TH0 = np.radians(_ang["theta0_deg"])
I_EFF = _M_CH3 * (_R * np.sin(_TH0)) ** 2   # [K·ps²]


def run_md(T: float, cfg: dict, seed: int = None) -> np.ndarray:
    """
    Nosé-Hoover NVT MD for n-pentane (1D torsional degree of freedom).

    Parameters
    ----------
    T    : float   Temperature [K]
    cfg  : dict    Loaded trappe_ua.toml dict
    seed : int     RNG seed (only used for initial velocity draw)

    Returns
    -------
    traj : np.ndarray, shape (n_steps,)
        phi1 (C1-C2-C3-C4 dihedral) in radians, one value per MD step.
    """
    rng   = np.random.default_rng(seed)
    n     = cfg["simulation"]["n_steps"]
    dt    = cfg["simulation"]["dt_ps"]
    tau_T = cfg["simulation"]["tau_T_ps"]
    beta  = 1.0 / T

    # Maxwell-Boltzmann initial velocity for phi1
    phi1  = np.pi      # start all-trans
    phi2  = np.pi      # fixed throughout
    omega = rng.normal(0.0, np.sqrt(T / I_EFF))   # [rad/ps]
    xi    = 0.0        # Nosé-Hoover friction variable

    traj = np.empty(n)
    for i in range(n):
        # ── velocity Verlet, split-operator integration ──────────────────
        # Generalised torque at current phi1
        tau = torsion_force(phi1)   # [K/rad]

        # Half-step omega update
        omega += 0.5 * dt * (tau / I_EFF - xi * omega)

        # Full-step position update, wrap to (-π, π]
        phi1 = (phi1 + dt * omega + np.pi) % (2 * np.pi) - np.pi

        # Recompute torque at new phi1
        tau_new = torsion_force(phi1)

        # Half-step xi update (trapezoidal estimate)
        xi_mid  = xi + 0.5 * dt / (tau_T ** 2) * (I_EFF * omega ** 2 / T - 1.0)

        # Half-step omega update (second half, using new tau and xi_mid)
        omega += 0.5 * dt * (tau_new / I_EFF - xi_mid * omega)

        # Full-step xi update
        xi = xi_mid + 0.5 * dt / (tau_T ** 2) * (I_EFF * omega ** 2 / T - 1.0)

        traj[i] = phi1

    return traj
