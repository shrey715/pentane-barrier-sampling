"""
mc.py — Baseline NVT Monte Carlo (Metropolis) for n-pentane.

Public API
----------
run_mc(T, cfg, seed=None) -> np.ndarray  shape (n_steps,)
    Trajectory of phi1 (C1-C2-C3-C4 dihedral) in radians.

Algorithm (Frenkel & Smit, Ch. 3 — Metropolis):
  1. Start from all-trans configuration (phi1 = phi2 = π)
  2. Propose  phi1_new = phi1 + U(-δ, +δ),  wrapping to (-π, π]
  3. Rebuild full 3D coords with build_pentane(phi1_new, phi2, cfg)
  4. Compute  ΔU = total_energy(new) − total_energy(old)
  5. Accept if ΔU < 0 or random() < exp(−ΔU/T)
  6. Record phi1 every step

Note: phi2 is kept fixed at π (trans) for the baseline.  Only phi1
(C1-C2-C3-C4) is the order parameter per the project specification.
"""
import numpy as np
from pentane.geometry import build_pentane, calc_dihedral
from pentane.forcefield import total_energy


def _wrap(phi: float) -> float:
    """Wrap dihedral to (-π, π]."""
    return (phi + np.pi) % (2 * np.pi) - np.pi


def run_mc(T: float, cfg: dict, seed: int = None) -> np.ndarray:
    """
    Metropolis NVT Monte Carlo for n-pentane.

    Parameters
    ----------
    T    : float   Temperature [K]
    cfg  : dict    Loaded trappe_ua.toml dict
    seed : int     RNG seed for reproducibility (optional)

    Returns
    -------
    traj : np.ndarray, shape (n_steps,)
        phi1 (C1-C2-C3-C4 dihedral) in radians, one value per MC step.
    """
    rng   = np.random.default_rng(seed)
    n     = cfg["simulation"]["n_steps"]
    delta = np.radians(cfg["simulation"]["mc_delta_phi_deg"])
    beta  = 1.0 / T

    # Start at all-trans: phi1 = phi2 = π
    phi1  = np.pi
    phi2  = np.pi
    coords = build_pentane(phi1, phi2, cfg)
    E_old  = total_energy(coords)

    traj = np.empty(n)
    for i in range(n):
        phi1_new = _wrap(phi1 + rng.uniform(-delta, delta))
        coords_new = build_pentane(phi1_new, phi2, cfg)
        E_new = total_energy(coords_new)

        dE = E_new - E_old
        if dE < 0.0 or rng.random() < np.exp(-beta * dE):
            phi1, coords, E_old = phi1_new, coords_new, E_new

        traj[i] = phi1

    return traj
