"""
Metropolis Monte Carlo Simulation — Dihedral Sampling
=====================================================

Implements a standard Metropolis Monte Carlo sampler for the backbone
dihedral angle of n-pentane under the TraPPE-UA torsion potential.

The algorithm:
1. Propose a trial move: φ' = φ + δφ,  δφ ~ U(−step_size, +step_size)
2. Compute energy change: ΔU = U(φ') − U(φ)
3. Accept with probability: min(1, exp(−βΔU)),  where β = 1/T

Since energies are in Kelvin and we use k_B = 1, β = 1/T directly.

The dihedral is wrapped to [−π, π] after each move to maintain
consistent bin assignment.
"""

import numpy as np

from pentane.forcefield import torsion_energy


def mc_simulation(
    T: float,
    n_steps: int = 200_000,
    phi_init: float = np.radians(180.0),
    step_size: float = 0.3,
    seed: int = 42,
) -> np.ndarray:
    """
    Run a Metropolis Monte Carlo simulation on the torsion potential.

    Parameters
    ----------
    T : float
        Temperature [K].
    n_steps : int
        Number of MC steps.
    phi_init : float
        Initial dihedral angle [radians].
    step_size : float
        Maximum displacement magnitude [radians] for trial moves.
    seed : int
        Random number generator seed for reproducibility.

    Returns
    -------
    dihedrals : ndarray, shape (n_steps,)
        Dihedral angle trajectory [radians].
    """
    rng = np.random.default_rng(seed)
    beta = 1.0 / T

    phi = phi_init
    E = float(torsion_energy(phi))

    dihedrals = np.zeros(n_steps)
    n_accept = 0

    for i in range(n_steps):
        # Propose trial move
        phi_new = phi + rng.uniform(-step_size, step_size)
        # Wrap to [−π, π]
        phi_new = (phi_new + np.pi) % (2.0 * np.pi) - np.pi

        E_new = float(torsion_energy(phi_new))
        dE = E_new - E

        # Metropolis acceptance criterion
        if dE < 0 or rng.random() < np.exp(-beta * dE):
            phi = phi_new
            E = E_new
            n_accept += 1

        dihedrals[i] = phi

    acceptance_rate = n_accept / n_steps
    print(f"  MC @ {T:6.1f} K: acceptance = {acceptance_rate:.3f}, "
          f"{n_steps} steps")

    return dihedrals
