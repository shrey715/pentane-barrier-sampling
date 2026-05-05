"""
umbrella.py — Cartesian umbrella-sampling Monte Carlo for n-pentane.

The move is single-atom Cartesian displacement, identical to the baseline
MC in mc.py. The only difference is the additional harmonic bias on the
phi1 dihedral. Raw phi1 samples are returned for WHAM post-processing.
"""
import numpy as np
from concurrent.futures import ProcessPoolExecutor

from pentane.forcefield import total_energy
from pentane.geometry import build_pentane, calc_dihedral
from pentane.wham import run_wham


def _wrap(phi: float) -> float:
    """Wrap a dihedral to (-π, π]."""
    return (phi + np.pi) % (2 * np.pi) - np.pi


def _bias(phi: float, phi0: float, k: float) -> float:
    """Harmonic umbrella bias [K]."""
    dphi = _wrap(phi - phi0)
    return 0.5 * k * dphi * dphi


def _propose_cartesian(pos: np.ndarray, rng: np.random.Generator,
                        dr_max: float) -> np.ndarray:
    """Displace one randomly chosen atom by a uniform vector in [-dr_max, dr_max]^3."""
    trial = pos.copy()
    atom = rng.integers(0, 5)
    trial[atom] += rng.uniform(-dr_max, dr_max, size=3)
    return trial


def _window_centres(cfg: dict) -> np.ndarray:
    n_windows = int(cfg["umbrella"]["n_windows"])
    edges = np.linspace(-np.pi, np.pi, n_windows + 1)
    return 0.5 * (edges[:-1] + edges[1:])


def run_window(phi0: float, T: float, cfg: dict, seed: int = None) -> np.ndarray:
    """Run one umbrella window with single-atom Cartesian displacement MC.

    Returns the biased phi1 trajectory for WHAM post-processing.
    """
    rng  = np.random.default_rng(seed)
    k    = float(cfg["umbrella"]["window_k_K_per_rad2"])
    n    = int(cfg["umbrella"]["n_steps_per_window"])
    # Umbrella windows use their own (larger) dr_max; fall back to simulation key.
    dr   = float(cfg["umbrella"].get("dr_max_ang", cfg["simulation"]["dr_max_ang"]))
    beta = 1.0 / T

    pos       = build_pentane(phi0, np.pi, cfg)
    phi       = calc_dihedral(pos[0], pos[1], pos[2], pos[3])
    e_phys    = total_energy(pos)
    e_bias    = _bias(phi, phi0, k)

    traj = np.empty(n, dtype=float)
    for i in range(n):
        trial      = _propose_cartesian(pos, rng, dr)
        phi_trial  = calc_dihedral(trial[0], trial[1], trial[2], trial[3])
        e_phys_new = total_energy(trial)
        e_bias_new = _bias(phi_trial, phi0, k)

        dE = (e_phys_new + e_bias_new) - (e_phys + e_bias)
        if dE < 0.0 or rng.random() < np.exp(-beta * dE):
            pos    = trial
            phi    = phi_trial
            e_phys = e_phys_new
            e_bias = e_bias_new

        traj[i] = phi

    return traj


def _run_window_worker(args: tuple) -> np.ndarray:
    """Top-level worker required so multiprocessing can pickle the task."""
    phi0, T, cfg, seed = args
    return run_window(phi0, T, cfg, seed)


def run_umbrella_sampling(T: float, cfg: dict, seed_base: int = 0,
                          n_workers: int | None = None) -> tuple:
    """Run all umbrella windows in parallel, then combine with WHAM.

    Parameters
    ----------
    T         : float   Temperature [K]
    cfg       : dict    Configuration dict (from config_loader)
    seed_base : int     Windows receive seeds seed_base+0, seed_base+1, …
    n_workers : int | None
        Number of worker processes.  ``None`` (default) uses all logical
        CPU cores.  Falls back to the ``umbrella.n_workers`` config key
        when the argument is not supplied explicitly.
    """
    phi0s = _window_centres(cfg)

    # Resolve worker count: explicit arg > config key > None (all cores)
    if n_workers is None:
        n_workers = cfg.get("umbrella", {}).get("n_workers") or None

    task_args = [
        (phi0, T, cfg, seed_base + i)
        for i, phi0 in enumerate(phi0s)
    ]

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        phi_samples_per_window = list(pool.map(_run_window_worker, task_args))

    bin_centres, pmf = run_wham(phi_samples_per_window, phi0s, cfg, T)
    return bin_centres, pmf
