"""
analysis.py — Statistical analysis of dihedral trajectories.

Public API
----------
exploration_entropy(phi_traj, n_bins=36)  -> float
    Shannon entropy of the sampled dihedral distribution.

early_exploration_score(phi_traj, n_bins=36, n_checkpoints=500)  -> float
    Time-averaged cumulative entropy sampled at logarithmically spaced
    checkpoints: E = (1/C) Σ_{c} S(t_c).  Avoids an O(N) Python loop.

boltzmann_pmf(phi_traj, T, n_bins=36)  -> (bin_centres_rad, F_K)
    PMF from Boltzmann inversion of the sampled histogram.
    F(φ) = −T ln P(φ) + C,  minimum set to zero.
"""
import numpy as np


def exploration_entropy(phi_traj: np.ndarray, n_bins: int = 36) -> float:
    """
    Shannon exploration entropy of the dihedral distribution.

    S = −Σᵢ Pᵢ ln Pᵢ    (natural log, nats)

    Parameters
    ----------
    phi_traj : np.ndarray, shape (n_steps,)   [rad]
    n_bins   : int   Number of histogram bins over [-π, π]

    Returns
    -------
    S : float   Entropy [nats]
    """
    counts, _ = np.histogram(phi_traj, bins=n_bins, range=(-np.pi, np.pi))
    probs = counts / counts.sum()
    # Only include bins with non-zero probability
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log(probs)))


def early_exploration_score(
    phi_traj: np.ndarray, n_bins: int = 36, n_checkpoints: int = 500
) -> float:
    """
    Early exploration score: time-average of cumulative entropy.

    E = (1/T) Σ_{t=1}^{T} S(t)

    Approximated by evaluating S at ``n_checkpoints`` linearly spaced
    strides so that each checkpoint represents the same number of steps
    and the average faithfully approximates the spec definition.

    A higher score indicates faster discovery of phase space.

    Parameters
    ----------
    phi_traj     : np.ndarray, shape (n_steps,)   [rad]
    n_bins       : int   Histogram bins over [-π, π]
    n_checkpoints: int   Number of linearly spaced evaluation points

    Returns
    -------
    score : float   [nats]
    """
    n = len(phi_traj)
    edges = np.linspace(-np.pi, np.pi, n_bins + 1)

    # Bin entire trajectory in one vectorised call
    bins = np.clip(np.searchsorted(edges[1:], phi_traj), 0, n_bins - 1)

    # Linear stride: each checkpoint spans the same number of steps,
    # so averaging over them approximates (1/T)Σ_{t=1}^{T} S(t)
    stride = max(1, n // n_checkpoints)
    checkpoints = np.arange(stride, n + 1, stride)

    running = np.zeros(n_bins, dtype=np.int64)
    prev_t = 0
    total_entropy = 0.0

    for t in checkpoints:
        for b in bins[prev_t:t]:
            running[b] += 1
        prev_t = t

        nz = running[running > 0]
        p = nz / t  # t == running.sum() at this point
        total_entropy += -np.sum(p * np.log(p))

    return total_entropy / len(checkpoints)


def boltzmann_pmf(
    phi_traj: np.ndarray, T: float, n_bins: int = 36
) -> tuple[np.ndarray, np.ndarray]:
    """
    Potential of mean force (PMF) via Boltzmann inversion of a histogram.

    F(φ) = −T · ln P(φ) + C,   minimum set to zero.

    Parameters
    ----------
    phi_traj : np.ndarray, shape (n_steps,)   [rad]
    T        : float   Temperature [K]
    n_bins   : int

    Returns
    -------
    bin_centres : np.ndarray, shape (n_bins,)   [rad]
    F_K         : np.ndarray, shape (n_bins,)   [K]
        PMF in Kelvin (k_B = 1), NaN where no samples were collected.
    """
    edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    bin_centres = 0.5 * (edges[:-1] + edges[1:])
    counts, _ = np.histogram(phi_traj, bins=edges)

    # Normalise to probability before taking the log — semantically correct
    # and safe when comparing trajectories of different lengths.
    # (The additive constant T·ln(N) cancels via the nanmin shift below,
    # so the PMF shape is numerically identical to using raw counts.)
    probs = counts / counts.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        pmf = np.where(probs > 0, -T * np.log(probs), np.nan)

    # Shift so minimum = 0
    finite_min = np.nanmin(pmf)
    pmf = pmf - finite_min

    return bin_centres, pmf
