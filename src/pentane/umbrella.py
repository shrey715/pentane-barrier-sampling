"""
umbrella.py — Cartesian umbrella-sampling Monte Carlo for n-pentane.

The windowed move set is identical to the baseline Cartesian MC move; the only
difference is the additional harmonic bias on the phi1 dihedral. The raw phi1
samples are returned for WHAM post-processing.
"""
import numpy as np

from pentane.forcefield import total_energy
from pentane.geometry import build_pentane, calc_dihedral, rotate_fragment
from pentane.wham import run_wham


def _wrap(phi: float) -> float:
    """Wrap a dihedral to (-π, π]."""
    return (phi + np.pi) % (2 * np.pi) - np.pi


def _bias(phi: float, phi0: float, k: float) -> float:
    """Harmonic umbrella bias [K]."""
    dphi = _wrap(phi - phi0)
    return 0.5 * k * dphi * dphi


def _propose_rotation(pos: np.ndarray, rng: np.random.Generator, delta: float) -> np.ndarray:
    """Propose the same rigid Cartesian move used in baseline MC."""
    angle = rng.uniform(-delta, delta)
    if rng.random() < 0.5:
        return rotate_fragment(pos, pos[1], pos[2], [3, 4], angle)
    return rotate_fragment(pos, pos[2], pos[3], [4], angle)


def _window_centres(cfg: dict) -> np.ndarray:
    n_windows = int(cfg["umbrella"]["n_windows"])
    edges = np.linspace(-np.pi, np.pi, n_windows + 1)
    return 0.5 * (edges[:-1] + edges[1:])


def run_window(phi0: float, T: float, cfg: dict, seed: int = None) -> np.ndarray:
    """Run one umbrella window and return the biased phi1 trajectory."""
    rng = np.random.default_rng(seed)
    k = float(cfg["umbrella"]["window_k_K_per_rad2"])
    n = int(cfg["umbrella"]["n_steps_per_window"])
    delta = np.radians(cfg["simulation"]["mc_delta_phi_deg"])
    beta = 1.0 / T

    pos = build_pentane(phi0, np.pi, cfg)
    e_phys = total_energy(pos)
    e_bias = _bias(calc_dihedral(pos[0], pos[1], pos[2], pos[3]), phi0, k)

    traj = np.empty(n, dtype=float)
    for i in range(n):
        trial = _propose_rotation(pos, rng, delta)
        phi_trial = calc_dihedral(trial[0], trial[1], trial[2], trial[3])
        e_phys_new = total_energy(trial)
        e_bias_new = _bias(phi_trial, phi0, k)

        dE = (e_phys_new + e_bias_new) - (e_phys + e_bias)
        if dE < 0.0 or rng.random() < np.exp(-beta * dE):
            pos = trial
            e_phys = e_phys_new
            e_bias = e_bias_new
            phi_trial = phi_trial

        traj[i] = calc_dihedral(pos[0], pos[1], pos[2], pos[3])

    return traj


def run_umbrella_sampling(T: float, cfg: dict, seed_base: int = 0):
    """Run all umbrella windows and return the WHAM PMF."""
    phi0s = _window_centres(cfg)
    phi_samples_per_window = [run_window(phi0, T, cfg, seed=seed_base + i) for i, phi0 in enumerate(phi0s)]
    bin_centres, pmf = run_wham(phi_samples_per_window, phi0s, cfg, T)
    return bin_centres, pmf
