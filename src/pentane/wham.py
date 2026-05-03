"""
wham.py — Weighted Histogram Analysis Method (WHAM).

Public API
----------
run_wham(trajs, phi0s, cfg, T) -> (bin_centres, pmf)
    Combines biased umbrella-sampling trajectories into an unbiased PMF.

Algorithm (Kumar et al. 1992; lecture notes):
  Given M windows, each with biasing potential U_i(φ) = (k/2)(φ − φ₀ᵢ)²:

  1.  Bin all trajectories into a shared histogram over [-π, π].
  2.  Iteratively solve the WHAM equations:

          ρ_unb(φ_m) = Σᵢ nᵢ(φ_m)
                       ─────────────────────────────────
                       Σᵢ Nᵢ · exp(fᵢ − βU_i(φ_m))

          exp(−fᵢ)   = Σ_m ρ_unb(φ_m) · exp(−βU_i(φ_m)) · Δφ

      until |Δf| < tolerance.

  3.  PMF(φ) = −T · ln ρ_unb(φ) + const  (min set to zero).

The dihedral distance used in the bias always wraps through the periodic
boundary:  Δφ = wrap(φ − φ₀),  wrap ∈ (-π, π].
"""
import numpy as np


def _wrap(phi: float | np.ndarray) -> np.ndarray | float:
    """Wrap to (-π, π]."""
    return (np.asarray(phi) + np.pi) % (2 * np.pi) - np.pi


def run_wham(
    trajs: list[np.ndarray],
    phi0s: np.ndarray,
    cfg: dict,
    T: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    WHAM unbiasing of umbrella-sampling trajectories.

    Parameters
    ----------
    trajs  : list of M arrays, each shape (n_steps,)
             Biased phi1 trajectories from run_window [rad].
    phi0s  : array, shape (M,)
             Window centres [rad].
    cfg    : dict    Loaded trappe_ua.toml dict.
    T      : float   Temperature at which US was run [K].

    Returns
    -------
    bin_centres : np.ndarray, shape (n_bins,)   [rad]
    pmf         : np.ndarray, shape (n_bins,)   [K]
        Unbiased potential of mean force, minimum set to zero.
    """
    _wham   = cfg["wham"]
    _umb    = cfg["umbrella"]
    _sim    = cfg["simulation"]
    n_bins  = _sim["n_bins"]
    k       = _umb["window_k_K_per_rad2"]
    tol     = _wham["tolerance"]
    max_it  = _wham["max_iter"]
    beta    = 1.0 / T
    M       = len(trajs)

    # ── Bin edges and centres ──────────────────────────────────────────────
    edges       = np.linspace(-np.pi, np.pi, n_bins + 1)
    bin_centres = 0.5 * (edges[:-1] + edges[1:])
    d_phi       = edges[1] - edges[0]

    # ── Build per-window histograms and sample counts ──────────────────────
    n_i = np.array([len(t) for t in trajs], dtype=float)   # shape (M,)
    h   = np.zeros((M, n_bins))                             # raw counts
    for i, traj in enumerate(trajs):
        counts, _ = np.histogram(traj, bins=edges)
        h[i] = counts

    n_total = h.sum(axis=0)   # total counts per bin, shape (n_bins,)

    # ── Bias matrix: bias_ij = U_i(φ_j) [K] ──────────────────────────────
    # shape (M, n_bins)
    bias = np.zeros((M, n_bins))
    for i, phi0 in enumerate(phi0s):
        dphi = _wrap(bin_centres - phi0)
        bias[i] = 0.5 * k * dphi ** 2

    # ── WHAM iteration ─────────────────────────────────────────────────────
    # Initialise free energy offsets to zero
    f = np.zeros(M)           # f_i = −ln Z_i  (dimensionless, β already in)

    for iteration in range(max_it):
        # Denominator for each bin:  Σᵢ Nᵢ exp(fᵢ − β·U_i(φ_m))
        # shape (n_bins,)
        denom = np.einsum("i,ij->j", n_i, np.exp(f[:, None] - beta * bias))

        # Unbiased density ρ_unb(φ_m)
        # Avoid division by zero for empty bins
        with np.errstate(divide="ignore", invalid="ignore"):
            rho = np.where(denom > 0, n_total / denom, 0.0)

        # Update free energies:  exp(−fᵢ) = Σ_m ρ(φ_m)·exp(−β·U_i(φ_m))·Δφ
        f_new = -np.log(
            np.einsum("j,ij->i", rho, np.exp(-beta * bias)) * d_phi
            + 1e-300    # guard against log(0)
        )

        # Shift so f[0] = 0 (removes arbitrary additive constant)
        f_new -= f_new[0]

        if np.max(np.abs(f_new - f)) < tol:
            f = f_new
            break
        f = f_new
    else:
        import warnings
        warnings.warn(f"WHAM did not converge after {max_it} iterations.", RuntimeWarning)

    # ── PMF ────────────────────────────────────────────────────────────────
    with np.errstate(divide="ignore", invalid="ignore"):
        pmf = np.where(rho > 0, -T * np.log(rho + 1e-300), np.nan)

    # Set minimum to zero (shift by min of finite values)
    finite_min = np.nanmin(pmf)
    pmf -= finite_min

    return bin_centres, pmf
