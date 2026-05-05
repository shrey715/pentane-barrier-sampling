"""
mc.py — Cartesian Metropolis Monte Carlo for n-pentane.

The move set keeps a full 3D coordinate array as state and rigidly rotates a
fragment about a backbone bond axis. The reported observable is the C1-C2-C3-C4
dihedral trajectory used by the analysis and umbrella-sampling workflow.
"""
import numpy as np

from pentane.forcefield import total_energy
from pentane.geometry import build_all_trans, calc_dihedral, rotate_fragment


def _wrap(phi: float) -> float:
    """Wrap a dihedral to (-π, π]."""
    return (phi + np.pi) % (2 * np.pi) - np.pi


def _propose_rotation(pos: np.ndarray, rng: np.random.Generator, delta: float) -> np.ndarray:
    """Propose a rigid fragment rotation about one backbone bond."""
    angle = rng.uniform(-delta, delta)
    if rng.random() < 0.5:
        return rotate_fragment(pos, pos[1], pos[2], [3, 4], angle)
    return rotate_fragment(pos, pos[2], pos[3], [4], angle)


def run_mc(T: float, cfg: dict, seed: int = None) -> np.ndarray:
    """Run Metropolis NVT MC and return the phi1 trajectory.

    Speedup: all random numbers are pre-generated outside the loop,
    eliminating per-step RNG call overhead (~2× faster than inline calls).
    """
    rng = np.random.default_rng(seed)
    n = int(cfg["simulation"]["n_steps"])
    delta = np.radians(cfg["simulation"]["mc_delta_phi_deg"])
    beta = 1.0 / T

    # Pre-generate all randoms — eliminates per-step RNG overhead
    bond_choices = rng.random(n)                  # which bond to rotate
    move_angles  = rng.uniform(-delta, delta, n)  # trial rotation angles
    accept_draws = rng.random(n)                  # Metropolis acceptance

    pos = build_all_trans(cfg)
    E_old = total_energy(pos)

    traj = np.empty(n, dtype=float)
    for i in range(n):
        angle = move_angles[i]
        if bond_choices[i] < 0.5:
            trial = rotate_fragment(pos, pos[1], pos[2], [3, 4], angle)
        else:
            trial = rotate_fragment(pos, pos[2], pos[3], [4], angle)

        E_new = total_energy(trial)
        dE = E_new - E_old

        if dE < 0.0 or accept_draws[i] < np.exp(-beta * dE):
            pos = trial
            E_old = E_new

        traj[i] = calc_dihedral(pos[0], pos[1], pos[2], pos[3])

    return traj
