"""
Analysis Module — Sampling Metrics and Free-Energy Profiles
============================================================

Provides quantitative tools to evaluate the sampling efficiency of
different simulation methods:

1. **Exploration entropy** S — measures how uniformly the dihedral
   space is sampled. S = −Σ Pᵢ ln Pᵢ, with S_max = ln(N_bins).

2. **Early exploration score** E — time-averaged entropy, rewarding
   methods that discover conformations sooner.

3. **Potential of Mean Force (PMF)** — free-energy profile F(φ)
   obtained by Boltzmann inversion of the sampled probability density:
   F(φ) = −T · ln P(φ) + const.

4. **Bin occupancy** — fraction of histogram bins visited at least once.
"""

import numpy as np

# Standard bin edges: 36 bins of 10° each spanning [−180°, 180°]
N_BINS: int = 36
BIN_EDGES_DEG: np.ndarray = np.linspace(-180.0, 180.0, N_BINS + 1)
BIN_CENTERS_DEG: np.ndarray = 0.5 * (BIN_EDGES_DEG[:-1] + BIN_EDGES_DEG[1:])
S_MAX: float = np.log(N_BINS)  # ≈ 3.584 (uniform distribution entropy)


def compute_entropy(dihedrals_deg: np.ndarray) -> float:
    """
    Compute the exploration entropy of a dihedral trajectory.

    Parameters
    ----------
    dihedrals_deg : ndarray
        Dihedral angle trajectory in degrees.

    Returns
    -------
    S : float
        Shannon entropy S = −Σ Pᵢ ln Pᵢ.  Range: [0, ln(36) ≈ 3.584].
    """
    hist, _ = np.histogram(dihedrals_deg, bins=BIN_EDGES_DEG)
    P = hist / hist.sum()
    P_nonzero = P[P > 0]
    return float(-np.sum(P_nonzero * np.log(P_nonzero)))


def compute_entropy_timeseries(
    dihedrals_deg: np.ndarray,
    chunk: int = 500,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the exploration entropy as a function of simulation progress.

    Parameters
    ----------
    dihedrals_deg : ndarray
        Full dihedral trajectory in degrees.
    chunk : int
        Interval (in steps) at which to evaluate S.

    Returns
    -------
    steps : ndarray
        Step numbers at which S was evaluated.
    S_t : ndarray
        Exploration entropy at each evaluation point.
    """
    N = len(dihedrals_deg)
    steps = np.arange(chunk, N + 1, chunk)
    S_t = np.array([compute_entropy(dihedrals_deg[:t]) for t in steps])
    return steps, S_t


def compute_early_exploration_score(
    dihedrals_deg: np.ndarray,
    chunk: int = 500,
) -> float:
    """
    Compute the early exploration score E = <S(t)>.

    A higher score indicates faster discovery of conformational states.

    Parameters
    ----------
    dihedrals_deg : ndarray
        Dihedral trajectory in degrees.
    chunk : int
        Evaluation interval for the entropy time series.

    Returns
    -------
    E : float
        Mean entropy over the trajectory (early exploration score).
    """
    _, S_t = compute_entropy_timeseries(dihedrals_deg, chunk)
    return float(S_t.mean())


def count_bins_visited(dihedrals_deg: np.ndarray) -> int:
    """
    Count the number of histogram bins visited at least once.

    Parameters
    ----------
    dihedrals_deg : ndarray
        Dihedral trajectory in degrees.

    Returns
    -------
    n_visited : int
        Number of bins with at least one visit (out of 36).
    """
    hist, _ = np.histogram(dihedrals_deg, bins=BIN_EDGES_DEG)
    return int(np.sum(hist > 0))


def compute_pmf(dihedrals_deg: np.ndarray, T: float) -> np.ndarray:
    """
    Compute the Potential of Mean Force (PMF) via Boltzmann inversion.

    Parameters
    ----------
    dihedrals_deg : ndarray
        Dihedral trajectory in degrees.
    T : float
        Temperature [K].

    Returns
    -------
    pmf : ndarray, shape (36,)
        Free-energy profile F(φ) in Kelvin, shifted so min(F) = 0.

    Notes
    -----
    Empty bins are filled with a small pseudocount (1e-10) to avoid
    log(0). These points will show as very high free energy, correctly
    indicating unsampled regions.
    """
    hist, _ = np.histogram(dihedrals_deg, bins=BIN_EDGES_DEG, density=True)
    # Avoid log(0) for unvisited bins
    hist = np.where(hist > 0, hist, 1e-10)
    pmf = -T * np.log(hist)  # k_B = 1 in Kelvin units
    pmf -= pmf.min()
    return pmf
