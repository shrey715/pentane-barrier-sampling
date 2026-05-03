"""
umbrella.py — Single-window biased Monte Carlo for umbrella sampling.

Public API
----------
run_window(phi0, T, cfg, seed=None) -> np.ndarray  shape (n_steps,)
    Biased phi1 samples in radians.  Bias is NOT removed here — WHAM does that.

Algorithm:
  Identical Metropolis logic as mc.run_mc, except the acceptance criterion
  uses the biased energy:

      U_biased(phi) = U_phys(phi) + U_bias(phi)
      U_bias(phi)   = (k/2)(wrap(phi − phi0))²

  where phi0 is the window centre and k = cfg["umbrella"]["window_k_K_per_rad2"].

  The returned trajectory contains the raw (biased) sampled phi1 values.
  WHAM (wham.run_wham) unbiases them.
"""
import numpy as np
from pentane.geometry import build_pentane
from pentane.forcefield import total_energy


def _wrap(phi: float) -> float:
    """Wrap dihedral to (-π, π]."""
    return (phi + np.pi) % (2 * np.pi) - np.pi


def _bias(phi: float, phi0: float, k: float) -> float:
    """Harmonic umbrella bias [K].  U = (k/2)(wrap(phi − phi0))²"""
    d = _wrap(phi - phi0)
    return 0.5 * k * d * d


def run_window(phi0: float, T: float, cfg: dict,
               seed: int = None) -> np.ndarray:
    """
    Single umbrella-sampling MC window.

    Parameters
    ----------
    phi0 : float   Window centre [rad]
    T    : float   Temperature [K]
    cfg  : dict    Loaded trappe_ua.toml dict
    seed : int     RNG seed for reproducibility (optional)

    Returns
    -------
    traj : np.ndarray, shape (n_steps,)
        Biased phi1 samples [rad], one per MC step.
        The bias is still present — pass to run_wham to remove it.
    """
    rng   = np.random.default_rng(seed)
    k     = cfg["umbrella"]["window_k_K_per_rad2"]
    n     = cfg["umbrella"]["n_steps_per_window"]
    delta = np.radians(cfg["simulation"]["mc_delta_phi_deg"])
    beta  = 1.0 / T

    # Start at the window centre; phi2 fixed at trans
    phi1  = phi0
    phi2  = np.pi
    coords  = build_pentane(phi1, phi2, cfg)
    E_phys  = total_energy(coords)
    E_bias  = _bias(phi1, phi0, k)

    traj = np.empty(n)
    for i in range(n):
        phi1_new   = _wrap(phi1 + rng.uniform(-delta, delta))
        coords_new = build_pentane(phi1_new, phi2, cfg)
        E_phys_new = total_energy(coords_new)
        E_bias_new = _bias(phi1_new, phi0, k)

        dE = (E_phys_new + E_bias_new) - (E_phys + E_bias)
        if dE < 0.0 or rng.random() < np.exp(-beta * dE):
            phi1, coords, E_phys, E_bias = (
                phi1_new, coords_new, E_phys_new, E_bias_new
            )

        traj[i] = phi1

    return traj
