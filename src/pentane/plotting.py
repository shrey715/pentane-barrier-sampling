"""
Plotting Module — Publication-Quality Figures
==============================================

Generates all figures required for the pentane barrier mini-project:

- Torsion potential energy surface U(φ)
- Initial molecular geometry (3D scatter)
- Dihedral time series (φ vs step) for all methods
- Dihedral angle histograms (probability distributions)
- Exploration entropy time series S(t)
- Potential of Mean Force (PMF) comparisons
- Summary bar chart of exploration metrics

All figures are saved as high-resolution PNG files to the specified
output directory.
"""

import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from pentane.forcefield import torsion_energy
from pentane.analysis import (
    BIN_CENTERS_DEG,
    BIN_EDGES_DEG,
    S_MAX,
    compute_entropy,
    compute_entropy_timeseries,
    compute_early_exploration_score,
    compute_pmf,
    count_bins_visited,
)

# ---------------------------------------------------------------------------
# Matplotlib style configuration
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#f8f9fa",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
})

# Consistent color palette for the three methods
COLORS = {
    "MC": "#2196F3",   # blue
    "MD": "#FF9800",   # orange
    "WL": "#4CAF50",   # green
}


def _ensure_dir(path: str | Path) -> Path:
    """Create directory if it doesn't exist and return as Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def plot_torsion_potential(output_dir: str | Path) -> str:
    """
    Plot the TraPPE-UA torsion potential U(φ) for n-pentane.

    Labels the trans minimum, gauche minima, and eclipsed barriers.

    Returns
    -------
    filepath : str
        Path to the saved figure.
    """
    out = _ensure_dir(output_dir)
    phi_deg = np.linspace(-180, 180, 1000)
    phi_rad = np.radians(phi_deg)
    U = torsion_energy(phi_rad)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(phi_deg, U, color="#d32f2f", linewidth=2.0)
    ax.fill_between(phi_deg, U, alpha=0.08, color="#d32f2f")

    # Annotate key states
    ax.annotate("Trans\n(global min)", xy=(180, 0), xytext=(150, 400),
                fontsize=9, ha="center",
                arrowprops=dict(arrowstyle="->", color="gray"))
    ax.annotate("Trans\n(global min)", xy=(-180, 0), xytext=(-150, 400),
                fontsize=9, ha="center",
                arrowprops=dict(arrowstyle="->", color="gray"))

    # Gauche minima
    gauche_phi = np.radians(60)
    gauche_E = float(torsion_energy(gauche_phi))
    ax.annotate(f"Gauche+\n({gauche_E:.0f} K)", xy=(60, gauche_E),
                xytext=(85, gauche_E + 500), fontsize=9, ha="center",
                arrowprops=dict(arrowstyle="->", color="gray"))
    ax.annotate(f"Gauche−\n({gauche_E:.0f} K)", xy=(-60, gauche_E),
                xytext=(-85, gauche_E + 500), fontsize=9, ha="center",
                arrowprops=dict(arrowstyle="->", color="gray"))

    # Eclipsed barriers
    eclipsed_phi = 0.0
    eclipsed_E = float(torsion_energy(eclipsed_phi))
    ax.annotate(f"Eclipsed\n({eclipsed_E:.0f} K)", xy=(0, eclipsed_E),
                xytext=(25, eclipsed_E + 300), fontsize=9, ha="center",
                arrowprops=dict(arrowstyle="->", color="gray"))

    ax.set_xlabel("Dihedral Angle φ [degrees]")
    ax.set_ylabel("Torsion Energy U(φ) [K]")
    ax.set_title("TraPPE-UA Torsion Potential for n-Pentane")
    ax.set_xlim(-180, 180)
    ax.set_ylim(-50, max(U) * 1.15)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(60))

    filepath = str(out / "torsion_potential.png")
    fig.savefig(filepath)
    plt.close(fig)
    print(f"  Saved: {filepath}")
    return filepath


def plot_initial_geometry(coords: np.ndarray, output_dir: str | Path) -> str:
    """
    Plot the 3D structure of the initial all-trans pentane configuration.

    Parameters
    ----------
    coords : ndarray, shape (5, 3)
        Cartesian coordinates of the 5 UA sites.

    Returns
    -------
    filepath : str
    """
    out = _ensure_dir(output_dir)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    # Draw bonds
    ax.plot(coords[:, 0], coords[:, 1], coords[:, 2],
            "o-", color="#1565C0", markersize=12, linewidth=3,
            markerfacecolor="#42A5F5", markeredgecolor="#0D47A1",
            markeredgewidth=1.5)

    # Label atoms
    labels = ["CH₃", "CH₂", "CH₂", "CH₂", "CH₃"]
    for i, (label, pos) in enumerate(zip(labels, coords)):
        ax.text(pos[0], pos[1], pos[2] + 0.15, f"C{i+1}\n({label})",
                fontsize=9, ha="center", va="bottom")

    ax.set_xlabel("x [Å]")
    ax.set_ylabel("y [Å]")
    ax.set_zlabel("z [Å]")
    ax.set_title("Initial All-Trans n-Pentane Configuration")

    filepath = str(out / "initial_geometry.png")
    fig.savefig(filepath)
    plt.close(fig)
    print(f"  Saved: {filepath}")
    return filepath


def plot_dihedral_timeseries(
    trajectories: dict[str, np.ndarray],
    T: float,
    output_dir: str | Path,
) -> str:
    """
    Plot dihedral angle vs simulation step for multiple methods.

    Parameters
    ----------
    trajectories : dict
        Mapping of method name → dihedral trajectory [degrees].
    T : float
        Temperature [K] (for title).

    Returns
    -------
    filepath : str
    """
    out = _ensure_dir(output_dir)
    n_methods = len(trajectories)

    fig, axes = plt.subplots(n_methods, 1, figsize=(12, 3.5 * n_methods),
                             sharex=True)
    if n_methods == 1:
        axes = [axes]

    for ax, (name, traj) in zip(axes, trajectories.items()):
        color_key = name.split()[0]  # "MC", "MD", or "WL"
        color = COLORS.get(color_key, "#666666")

        # Subsample for plotting efficiency
        step = max(1, len(traj) // 5000)
        x = np.arange(0, len(traj), step)
        y = traj[::step]

        ax.plot(x, y, color=color, alpha=0.6, linewidth=0.5, rasterized=True)
        ax.set_ylabel("φ [°]")
        ax.set_ylim(-200, 200)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(60))

        # Highlight trans/gauche regions
        ax.axhline(180, color="gray", linestyle=":", alpha=0.4)
        ax.axhline(-180, color="gray", linestyle=":", alpha=0.4)
        ax.axhline(60, color="gray", linestyle=":", alpha=0.3)
        ax.axhline(-60, color="gray", linestyle=":", alpha=0.3)
        ax.axhline(0, color="gray", linestyle=":", alpha=0.3)

        bins_v = count_bins_visited(traj)
        ax.set_title(f"{name}  (bins: {bins_v}/36)", fontsize=12)

    axes[-1].set_xlabel("MC/MD Step")
    fig.suptitle(f"Dihedral Angle Trajectories — T = {T:.0f} K",
                 fontsize=15, y=1.01)
    fig.tight_layout()

    filepath = str(out / f"dihedral_timeseries_{T:.0f}K.png")
    fig.savefig(filepath)
    plt.close(fig)
    print(f"  Saved: {filepath}")
    return filepath


def plot_dihedral_histograms(
    all_trajectories: dict[str, np.ndarray],
    output_dir: str | Path,
) -> str:
    """
    Plot side-by-side dihedral angle histograms for all methods and temperatures.

    Parameters
    ----------
    all_trajectories : dict
        Mapping of label → dihedral trajectory [degrees].
        Labels should be like "MC 120K", "MD 250K", "WL 120K", etc.

    Returns
    -------
    filepath : str
    """
    out = _ensure_dir(output_dir)

    n = len(all_trajectories)
    ncols = 3
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.atleast_2d(axes).ravel()

    for ax, (name, traj) in zip(axes, all_trajectories.items()):
        color_key = name.split()[0]
        color = COLORS.get(color_key, "#666666")

        ax.hist(traj, bins=BIN_EDGES_DEG, density=True, color=color,
                alpha=0.7, edgecolor="white", linewidth=0.5)
        ax.set_title(name, fontsize=11)
        ax.set_xlabel("φ [°]")
        ax.set_ylabel("P(φ)")
        ax.set_xlim(-180, 180)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(60))

    # Hide unused axes
    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle("Dihedral Angle Distributions", fontsize=15, y=1.02)
    fig.tight_layout()

    filepath = str(out / "dihedral_histograms.png")
    fig.savefig(filepath)
    plt.close(fig)
    print(f"  Saved: {filepath}")
    return filepath


def plot_entropy_timeseries(
    all_trajectories: dict[str, np.ndarray],
    output_dir: str | Path,
    chunk: int = 500,
) -> str:
    """
    Plot exploration entropy S(t) for all methods, showing convergence.

    Returns
    -------
    filepath : str
    """
    out = _ensure_dir(output_dir)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for temp_label, ax, temp_K in [("120 K", axes[0], 120), ("250 K", axes[1], 250)]:
        ax.axhline(S_MAX, color="black", linestyle="--", alpha=0.5,
                   label=f"S_max = {S_MAX:.3f}")

        for name, traj in all_trajectories.items():
            if f"{temp_K}K" not in name:
                continue
            color_key = name.split()[0]
            color = COLORS.get(color_key, "#666666")

            steps, S_t = compute_entropy_timeseries(traj, chunk)
            ax.plot(steps, S_t, color=color, linewidth=1.5, label=name)

        ax.set_xlabel("Step")
        ax.set_ylabel("Exploration Entropy S")
        ax.set_title(f"T = {temp_label}")
        ax.legend(loc="lower right")
        ax.set_ylim(0, S_MAX * 1.15)

    fig.suptitle("Exploration Entropy Convergence", fontsize=15, y=1.02)
    fig.tight_layout()

    filepath = str(out / "entropy_timeseries.png")
    fig.savefig(filepath)
    plt.close(fig)
    print(f"  Saved: {filepath}")
    return filepath


def plot_pmf_comparison(
    trajectories: dict[str, np.ndarray],
    T: float,
    output_dir: str | Path,
) -> str:
    """
    Overlay the Potential of Mean Force (PMF) for multiple methods.

    Also plots the exact torsion potential for reference.

    Returns
    -------
    filepath : str
    """
    out = _ensure_dir(output_dir)

    fig, ax = plt.subplots(figsize=(10, 5))

    # Reference: exact torsion potential
    phi_exact = np.linspace(-180, 180, 500)
    U_exact = torsion_energy(np.radians(phi_exact))
    ax.plot(phi_exact, U_exact, "k--", linewidth=1.5, alpha=0.5,
            label="Exact U(φ)")

    for name, traj in trajectories.items():
        color_key = name.split()[0]
        color = COLORS.get(color_key, "#666666")
        pmf = compute_pmf(traj, T)
        ax.plot(BIN_CENTERS_DEG, pmf, "o-", color=color, markersize=3,
                linewidth=1.5, label=f"{name} PMF")

    ax.set_xlabel("Dihedral Angle φ [degrees]")
    ax.set_ylabel("Free Energy F(φ) [K]")
    ax.set_title(f"Potential of Mean Force — T = {T:.0f} K")
    ax.set_xlim(-180, 180)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(60))
    ax.legend()

    filepath = str(out / f"pmf_comparison_{T:.0f}K.png")
    fig.savefig(filepath)
    plt.close(fig)
    print(f"  Saved: {filepath}")
    return filepath


def plot_exploration_summary(
    all_trajectories: dict[str, np.ndarray],
    output_dir: str | Path,
) -> str:
    """
    Bar chart summarizing bins visited and exploration entropy for all methods.

    Returns
    -------
    filepath : str
    """
    out = _ensure_dir(output_dir)

    names = list(all_trajectories.keys())
    bins_visited = [count_bins_visited(t) for t in all_trajectories.values()]
    entropies = [compute_entropy(t) for t in all_trajectories.values()]
    e_scores = [compute_early_exploration_score(t) for t in all_trajectories.values()]
    bar_colors = [COLORS.get(n.split()[0], "#666666") for n in names]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Bins visited
    axes[0].bar(names, bins_visited, color=bar_colors, edgecolor="white")
    axes[0].axhline(36, color="black", linestyle="--", alpha=0.4)
    axes[0].set_ylabel("Bins Visited (out of 36)")
    axes[0].set_title("Configurational Coverage")
    axes[0].tick_params(axis="x", rotation=45)

    # Exploration entropy
    axes[1].bar(names, entropies, color=bar_colors, edgecolor="white")
    axes[1].axhline(S_MAX, color="black", linestyle="--", alpha=0.4,
                    label=f"S_max = {S_MAX:.2f}")
    axes[1].set_ylabel("Exploration Entropy S")
    axes[1].set_title("Sampling Uniformity")
    axes[1].legend()
    axes[1].tick_params(axis="x", rotation=45)

    # Early exploration score
    axes[2].bar(names, e_scores, color=bar_colors, edgecolor="white")
    axes[2].set_ylabel("Early Exploration Score E")
    axes[2].set_title("Speed of Discovery")
    axes[2].tick_params(axis="x", rotation=45)

    fig.suptitle("Sampling Efficiency Comparison", fontsize=15, y=1.02)
    fig.tight_layout()

    filepath = str(out / "exploration_summary.png")
    fig.savefig(filepath)
    plt.close(fig)
    print(f"  Saved: {filepath}")
    return filepath
