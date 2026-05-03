"""
wham.py — Weighted Histogram Analysis Method (WHAM).

The implementation keeps the same binning and bias model as the umbrella
windows and can optionally return the per-iteration free-energy offsets for
convergence diagnostics.
"""
import warnings

import numpy as np


def _wrap(phi: float | np.ndarray) -> np.ndarray | float:
    """Wrap angles to (-π, π]."""
    return (np.asarray(phi) + np.pi) % (2 * np.pi) - np.pi


def wham(
    trajs: list[np.ndarray],
    phi0s: np.ndarray,
    cfg: dict,
    T: float,
    return_history: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Combine umbrella trajectories into an unbiased PMF."""
    wham_cfg = cfg["wham"]
    umbrella_cfg = cfg["umbrella"]
    sim_cfg = cfg["simulation"]

    n_bins = int(sim_cfg["n_bins"])
    k = float(umbrella_cfg["window_k_K_per_rad2"])
    tol = float(wham_cfg["tolerance"])
    max_iter = int(wham_cfg["max_iter"])
    M = len(trajs)

    edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    bin_centres = 0.5 * (edges[:-1] + edges[1:])
    d_phi = edges[1] - edges[0]

    n_i = np.array([len(traj) for traj in trajs], dtype=float)
    h = np.zeros((M, n_bins), dtype=float)
    for i, traj in enumerate(trajs):
        h[i], _ = np.histogram(traj, bins=edges)

    n_total = h.sum(axis=0)

    bias = np.zeros((M, n_bins), dtype=float)
    for i, phi0 in enumerate(phi0s):
        dphi = _wrap(bin_centres - phi0)
        bias[i] = 0.5 * k * dphi ** 2

    f = np.zeros(M, dtype=float)
    history = []

    for _ in range(max_iter):
        denom = np.einsum("i,ij->j", n_i, np.exp((f[:, None] - bias) / T))
        with np.errstate(divide="ignore", invalid="ignore"):
            rho = np.where(denom > 0, n_total / denom, 0.0)

        f_new = -T * np.log(np.einsum("j,ij->i", rho, np.exp(-bias / T)) * d_phi + 1e-300)
        f_new -= f_new[0]
        history.append(f_new.copy())

        if np.max(np.abs(f_new - f)) < tol:
            f = f_new
            break
        f = f_new
    else:
        warnings.warn(f"WHAM did not converge after {max_iter} iterations.", RuntimeWarning)

    with np.errstate(divide="ignore", invalid="ignore"):
        pmf = np.where(rho > 0, -T * np.log(rho + 1e-300), np.nan)

    pmf -= np.nanmin(pmf)

    if return_history:
        return bin_centres, pmf, np.asarray(history)
    return bin_centres, pmf


def run_wham(
    trajs: list[np.ndarray],
    phi0s: np.ndarray,
    cfg: dict,
    T: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compatibility wrapper returning only the WHAM PMF."""
    bin_centres, pmf = wham(trajs, phi0s, cfg, T, return_history=False)
    return bin_centres, pmf
