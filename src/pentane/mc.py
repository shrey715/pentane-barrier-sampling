"""
mc.py — Cartesian single-atom-displacement Metropolis MC for n-pentane.

Move: pick one atom at random, displace it by a uniform random vector in
[-dr_max, dr_max]^3. Compute total energy before/after. Accept with the
Metropolis criterion. Observable is the C1-C2-C3-C4 dihedral extracted
from the Cartesian configuration at each step.

The MD (md.py) already propagates full Cartesian coordinates via velocity
Verlet + Nosé-Hoover — this MC implementation uses the same Cartesian
representation so both methods are directly comparable.
"""
import numpy as np

from pentane.forcefield import total_energy
from pentane.geometry import build_all_trans, calc_dihedral


def run_mc(T: float, cfg: dict, seed: int = None) -> np.ndarray:
    """Run Cartesian single-atom-displacement NVT MC; return φ₁ trajectory.

    Parameters
    ----------
    T   : Temperature [K]
    cfg : Configuration dict (from config_loader.CFG)
    seed: RNG seed for reproducibility

    Returns
    -------
    traj : np.ndarray, shape (n_steps,) [rad]
        C1-C2-C3-C4 dihedral angle at every step.
    """
    rng   = np.random.default_rng(seed)
    n     = int(cfg["simulation"]["n_steps"])
    dr    = float(cfg["simulation"]["dr_max_ang"])
    beta  = 1.0 / T

    # Pre-generate all random numbers outside the loop (~2× speedup)
    atoms    = rng.integers(0, 5, size=n)               # which atom to move
    deltas   = rng.uniform(-dr, dr, size=(n, 3))        # displacement vector
    uniforms = rng.random(n)                            # Metropolis draw

    pos   = build_all_trans(cfg)
    E_old = total_energy(pos)

    traj = np.empty(n, dtype=float)

    for i in range(n):
        trial          = pos.copy()
        trial[atoms[i]] += deltas[i]

        E_new = total_energy(trial)
        dE    = E_new - E_old

        if dE < 0.0 or uniforms[i] < np.exp(-beta * dE):
            pos   = trial
            E_old = E_new

        traj[i] = calc_dihedral(pos[0], pos[1], pos[2], pos[3])

    return traj
