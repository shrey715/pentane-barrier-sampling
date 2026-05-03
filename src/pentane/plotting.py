"""plotting.py — Matplotlib figures for the pentane barrier sampling project."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from pathlib import Path
from pentane.analysis import boltzmann_pmf

sns.set_theme(style="whitegrid", context="notebook", font_scale=1.1)
plt.rcParams.update({"figure.dpi": 150, "lines.linewidth": 1.4, "font.family": "sans-serif"})

# Colours matching seaborn's default blue/orange/green/purple
COLORS = ["#4C72B0", "#DD8452", "#55A868", "#8172B2"]
LABELS = {"mc_120": "MC 120 K", "mc_250": "MC 250 K",
          "md_120": "MD 120 K", "md_250": "MD 250 K"}
R2D = 180.0 / np.pi   # radians → degrees


def _deg_fmt():
    return ticker.FuncFormatter(lambda x, _: f"{x:.0f}°")


def _save(fig, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {path}")


# ── 1. Dihedral time series ──────────────────────────────────────────────────

def plot_dihedral_timeseries(trajs: dict, out_path: str):
    """4-panel φ₁(t) for MC/MD × 120 K/250 K."""
    keys = ["mc_120", "mc_250", "md_120", "md_250"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharey=True)
    fig.suptitle("Dihedral φ₁ Time Series", fontsize=14, y=1.01)

    for ax, key, colour in zip(axes.flat, keys, COLORS):
        traj = trajs[key]
        ax.plot(np.arange(len(traj)), traj * R2D, color=colour, lw=0.4, alpha=0.85)
        ax.set_title(LABELS[key], fontsize=11)
        ax.set_xlabel("Step")
        ax.set_ylabel("φ₁ [°]")
        ax.set_ylim(-185, 185)
        ax.yaxis.set_major_formatter(_deg_fmt())

    fig.tight_layout()
    _save(fig, out_path)


# ── 2. Baseline distributions ────────────────────────────────────────────────

def plot_baseline_distributions(trajs: dict, n_bins: int, out_path: str):
    """4-panel dihedral probability histograms."""
    edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:]) * R2D
    bar_w = R2D * (edges[1] - edges[0]) * 0.9

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharey=False)
    fig.suptitle("φ₁ Dihedral Distributions", fontsize=14, y=1.01)

    for ax, key, colour in zip(axes.flat, ["mc_120", "mc_250", "md_120", "md_250"], COLORS):
        counts, _ = np.histogram(trajs[key], bins=edges)
        probs = counts / counts.sum()
        ax.bar(centres, probs, width=bar_w, color=colour, alpha=0.75, edgecolor="none")
        ax.set_title(LABELS[key], fontsize=11)
        ax.set_xlabel("φ₁ [°]")
        ax.set_ylabel("Probability")
        ax.xaxis.set_major_formatter(_deg_fmt())

    fig.tight_layout()
    _save(fig, out_path)


# ── 3. Baseline PMF ──────────────────────────────────────────────────────────

def plot_baseline_pmf(trajs: dict, T_list: list, n_bins: int, out_path: str):
    """Boltzmann-inversion PMF at both temperatures (MC only)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Baseline PMF (Boltzmann Inversion)", fontsize=14)

    for ax, (key, T, colour, label) in zip(axes, [
        ("mc_120", T_list[0], COLORS[0], "MC 120 K"),
        ("mc_250", T_list[1], COLORS[1], "MC 250 K"),
    ]):
        centres, pmf = boltzmann_pmf(trajs[key], T, n_bins)
        ax.plot(centres * R2D, pmf, color=colour, lw=2.0)
        ax.fill_between(centres * R2D, pmf, alpha=0.15, color=colour)
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("φ₁ [°]")
        ax.set_ylabel("F(φ₁) [K]")
        ax.xaxis.set_major_formatter(_deg_fmt())
        ax.set_xlim(-180, 180)

    fig.tight_layout()
    _save(fig, out_path)


# ── 4. Entropy curves ────────────────────────────────────────────────────────

def plot_entropy_curves(trajs: dict, n_bins: int, out_path: str):
    """Cumulative Shannon entropy S(t) for all 4 baseline runs."""
    edges = np.linspace(-np.pi, np.pi, n_bins + 1)

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.set_title("Exploration Entropy S(t) vs Step", fontsize=13)
    ax.set_xlabel("Step")
    ax.set_ylabel("Shannon entropy S [nats]")

    for key, colour in zip(["mc_120", "mc_250", "md_120", "md_250"], COLORS):
        traj = trajs[key]
        n = len(traj)
        stride = max(1, n // 500)
        checkpoints = np.arange(stride, n + 1, stride)
        entropies = []
        for t in checkpoints:
            counts, _ = np.histogram(traj[:t], bins=edges)
            p = counts / counts.sum()
            p = p[p > 0]
            entropies.append(-np.sum(p * np.log(p)))
        ax.plot(checkpoints, entropies, color=colour, label=LABELS[key], lw=1.8)

    ax.axhline(np.log(n_bins), ls="--", color="grey", lw=1.0,
               label=f"S_max = ln({n_bins}) = {np.log(n_bins):.2f}")
    ax.legend(fontsize=9)
    fig.tight_layout()
    _save(fig, out_path)


# ── 5. US window histograms ──────────────────────────────────────────────────

def plot_us_window_histograms(trajs: list, phi0s: np.ndarray, out_path: str):
    """Overlapping biased-window histograms (5° bins for legibility)."""
    edges = np.linspace(-np.pi, np.pi, 73)   # 72 bins = 5° each
    centres = 0.5 * (edges[:-1] + edges[1:]) * R2D
    cmap = plt.cm.hsv(np.linspace(0, 0.9, len(trajs)))

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_title("Umbrella Sampling — Biased Window Histograms", fontsize=13)
    ax.set_xlabel("φ₁ [°]")
    ax.set_ylabel("Probability")
    ax.xaxis.set_major_formatter(_deg_fmt())

    for i, (traj, phi0, colour) in enumerate(zip(trajs, phi0s, cmap)):
        counts, _ = np.histogram(traj, bins=edges)
        probs = counts / counts.sum()
        ax.plot(centres, probs, color=colour, lw=1.2, alpha=0.8,
                label=f"{phi0 * R2D:.0f}°" if i % 3 == 0 else None)

    ax.legend(fontsize=7, ncol=3, title="φ₀", title_fontsize=8)
    ax.set_xlim(-180, 180)
    fig.tight_layout()
    _save(fig, out_path)


# ── 6. WHAM convergence ──────────────────────────────────────────────────────

def plot_wham_convergence(f_history: np.ndarray, out_path: str, T: float = None):
    """Free-energy offsets fᵢ vs WHAM iteration."""
    if f_history.ndim != 2 or f_history.size == 0:
        raise ValueError("f_history must be shape (n_iter, n_windows)")

    T_label = f" ({T:.0f} K)" if T is not None else ""
    n_iter, n_windows = f_history.shape
    cmap = plt.cm.viridis(np.linspace(0.1, 0.9, n_windows))

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.set_title(f"WHAM Convergence: Free-Energy Offsets{T_label}", fontsize=13)
    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"$f_i$ [K]")
    for i in range(n_windows):
        ax.plot(np.arange(1, n_iter + 1), f_history[:, i], color=cmap[i], lw=1.0, alpha=0.9)
    ax.set_xlim(1, n_iter)
    fig.tight_layout()
    _save(fig, out_path)


# ── 7. WHAM PMF vs baseline ──────────────────────────────────────────────────

def plot_wham_pmf(bin_centres: np.ndarray, pmf_wham: np.ndarray,
                   trajs: dict, T: float, out_path: str, n_bins: int = 36):
    """WHAM PMF overlaid on the matching MC baseline PMF."""
    mc_key = f"mc_{int(round(T))}"
    centres_bl, pmf_bl = boltzmann_pmf(trajs[mc_key], T, n_bins)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title(f"WHAM PMF vs Baseline PMF ({T:.0f} K)", fontsize=13)
    ax.set_xlabel("φ₁ [°]")
    ax.set_ylabel("F(φ₁) [K]")
    ax.xaxis.set_major_formatter(_deg_fmt())
    ax.plot(bin_centres * R2D, pmf_wham, color=COLORS[3], lw=2.5,
            label="WHAM (umbrella sampling)")
    ax.plot(centres_bl * R2D, pmf_bl, color=COLORS[0], lw=2.0,
            ls="--", label=f"Baseline MC {int(round(T)):.0f} K")
    ax.set_xlim(-180, 180)
    ax.legend(fontsize=10)
    fig.tight_layout()
    _save(fig, out_path)


# ── 8. PMF side-by-side comparison ──────────────────────────────────────────

def plot_pmf_comparison(bin_centres: np.ndarray, pmf_wham: np.ndarray,
                         trajs: dict, T: float, out_path: str, n_bins: int = 36):
    """WHAM PMF vs Boltzmann PMF side-by-side."""
    mc_key = f"mc_{int(round(T))}"
    centres_bl, pmf_bl = boltzmann_pmf(trajs[mc_key], T, n_bins)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"PMF Comparison: WHAM vs Boltzmann Inversion ({T:.0f} K)", fontsize=13)

    for ax, pmf, centres, colour, title in [
        (ax1, pmf_wham, bin_centres, COLORS[3], "WHAM PMF"),
        (ax2, pmf_bl,   centres_bl,  COLORS[0], f"Boltzmann PMF (MC {int(round(T)):.0f} K)"),
    ]:
        ax.plot(centres * R2D, pmf, color=colour, lw=2.2)
        ax.fill_between(centres * R2D, pmf, alpha=0.12, color=colour)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("φ₁ [°]")
        ax.set_ylabel("F(φ₁) [K]")
        ax.xaxis.set_major_formatter(_deg_fmt())
        ax.set_xlim(-180, 180)

    fig.tight_layout()
    _save(fig, out_path)
