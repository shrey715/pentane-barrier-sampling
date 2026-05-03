"""
analysis.py — Statistical analysis of dihedral trajectories.

Public API
----------
exploration_entropy(phi_traj, n_bins=36)  -> float
    Shannon entropy of the sampled dihedral distribution.

early_exploration_score(phi_traj, n_bins=36)  -> float
    Time-averaged cumulative entropy: E = (1/T) Σ_{t=1}^{T} S(t).

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


def early_exploration_score(phi_traj: np.ndarray, n_bins: int = 36) -> float:
    """
    Early exploration score: time-average of cumulative entropy.

    E = (1/T) Σ_{t=1}^{T} S(t)

    where S(t) is the exploration entropy of the first t steps.
    A higher score indicates faster discovery of phase space.

    Parameters
    ----------
    phi_traj : np.ndarray, shape (n_steps,)   [rad]
    n_bins   : int

    Returns
    -------
    score : float   [nats]
    """
    n = len(phi_traj)
    edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    running_counts = np.zeros(n_bins, dtype=float)
    total_entropy = 0.0

    for t in range(n):
        # Find bin index for this sample
        idx = int(np.searchsorted(edges[1:], phi_traj[t]))
        idx = min(idx, n_bins - 1)
        running_counts[idx] += 1

        # Compute entropy of current distribution
        total = running_counts.sum()
        probs = running_counts[running_counts > 0] / total
        total_entropy += -np.sum(probs * np.log(probs))

    return total_entropy / n


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

    with np.errstate(divide="ignore", invalid="ignore"):
        pmf = np.where(counts > 0, -T * np.log(counts.astype(float)), np.nan)

    # Shift so minimum = 0
    finite_min = np.nanmin(pmf)
    pmf = pmf - finite_min

    return bin_centres, pmf
