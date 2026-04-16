"""
Wang-Landau Enhanced Sampling — Flat-Histogram Monte Carlo
==========================================================

Implements the Wang-Landau (WL) algorithm for the backbone dihedral of
n-pentane. This is a flat-histogram method that iteratively estimates
the density of states g(φ) to achieve uniform sampling across all
dihedral bins, even when thermal barriers prevent ergodic exploration.

Algorithm
---------
1. Initialize: log g(φ) = 0 for all bins; modification factor f = 1.0
2. At each step:
   a. Propose φ' = φ + δφ
   b. Accept with probability:
      P_acc = min(1, exp(−β·ΔU_phys + Δg))
      where Δg = log g(φ_old) − log g(φ_new) favors under-visited bins
   c. Update: log g(current bin) += f;  H(current bin) += 1
3. Flatness check (every `check_interval` steps):
   If H_min ≥ flatness × <H> and all bins have ≥ min_visits:
     f → f/2;  H → 0  (reset histogram)
4. Repeat until n_steps exhausted.

The bias potential V_bias(φ) = −log g(φ) accumulates, progressively
penalizing over-visited states and rewarding under-explored regions.
As f → 0, the estimate converges and sampling becomes increasingly flat.

References
----------
- F. Wang & D. P. Landau, Phys. Rev. Lett. 86, 2050 (2001).
- F. Wang & D. P. Landau, Phys. Rev. E 64, 056101 (2001).
"""

import numpy as np

from pentane.forcefield import torsion_energy


def wang_landau_sampling(
    T: float,
    n_steps: int = 200_000,
    n_bins: int = 36,
    phi_init: float = np.radians(180.0),
    step_size: float = 0.5,
    flatness: float = 0.8,
    check_interval: int = 1000,
    min_visits: int = 100,
    seed: int = 99,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run the Wang-Landau flat-histogram Monte Carlo simulation.

    Parameters
    ----------
    T : float
        Temperature [K] (sets the physical Boltzmann weight).
    n_steps : int
        Total number of MC steps.
    n_bins : int
        Number of histogram bins spanning [−π, π].
    phi_init : float
        Initial dihedral angle [radians].
    step_size : float
        Maximum trial displacement [radians].
    flatness : float
        Flatness criterion: H_min ≥ flatness × <H> triggers f reduction.
    check_interval : int
        Number of steps between flatness checks.
    min_visits : int
        Minimum visits per bin before flatness check is applied.
    seed : int
        Random number generator seed.

    Returns
    -------
    dihedrals : ndarray, shape (n_steps,)
        Dihedral angle trajectory [radians].
    log_g : ndarray, shape (n_bins,)
        Final estimate of log density of states.
    """
    rng = np.random.default_rng(seed)

    # Log density of states and visit histogram
    log_g = np.zeros(n_bins)
    hist = np.zeros(n_bins, dtype=int)

    # Modification factor (initial)
    f = 1.0

    def get_bin(phi: float) -> int:
        """Map dihedral angle φ ∈ [−π, π] to bin index."""
        idx = int((phi + np.pi) / (2.0 * np.pi) * n_bins)
        return min(max(idx, 0), n_bins - 1)

    # Initial state
    phi = float(phi_init)
    E = float(torsion_energy(phi))
    b = get_bin(phi)
    beta = 1.0 / T

    dihedrals = np.zeros(n_steps)
    n_f_reductions = 0

    for i in range(n_steps):
        # --- Propose trial move ---
        phi_new = phi + rng.uniform(-step_size, step_size)
        phi_new = (phi_new + np.pi) % (2.0 * np.pi) - np.pi

        b_new = get_bin(phi_new)
        E_new = float(torsion_energy(phi_new))

        # --- Wang-Landau acceptance criterion ---
        # Combines physical Boltzmann factor with bias from log g
        log_acc = -beta * (E_new - E) + (log_g[b] - log_g[b_new])

        if log_acc >= 0 or rng.random() < np.exp(log_acc):
            phi = phi_new
            E = E_new
            b = b_new

        # --- Update density of states and histogram ---
        log_g[b] += f
        hist[b] += 1
        dihedrals[i] = phi

        # --- Flatness check ---
        if (i + 1) % check_interval == 0:
            if hist.min() >= min_visits:
                mean_hist = hist[hist > 0].mean()
                if hist.min() >= flatness * mean_hist:
                    f *= 0.5
                    hist[:] = 0
                    n_f_reductions += 1

    print(f"  WL @ {T:6.1f} K: {n_steps} steps, "
          f"f reductions = {n_f_reductions}, "
          f"final f = {f:.2e}")

    return dihedrals, log_g
