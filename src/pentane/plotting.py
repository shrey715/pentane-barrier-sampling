"""
plotting.py — All matplotlib figures for the pentane barrier sampling project.

Public API
----------
plot_dihedral_timeseries(trajs_dict, out_path)
plot_baseline_distributions(trajs_dict, cfg, out_path)
plot_baseline_pmf(trajs_dict, T_list, cfg, out_path)
plot_entropy_curves(trajs_dict, cfg, out_path)
plot_us_window_histograms(trajs, phi0s, cfg, out_path)
plot_wham_pmf(bin_centres, pmf_wham, trajs_dict, T_us, cfg, out_path)
plot_pmf_comparison(bin_centres, pmf_wham, trajs_dict, T_us, cfg, out_path)

trajs_dict keys: "mc_120", "mc_250", "md_120", "md_250"
All temperatures in K.  All dihedrals in rad.  PMF in K.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
from pentane.analysis import boltzmann_pmf, exploration_entropy

# ── Style ───────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 150,
    "figure.facecolor": "#0d1117",
    "axes.facecolor": "#161b22",
    "axes.edgecolor": "#30363d",
    "axes.labelcolor": "#c9d1d9",
    "axes.titlecolor": "#e6edf3",
    "axes.grid": True,
    "grid.color": "#21262d",
    "grid.linewidth": 0.6,
    "xtick.color": "#8b949e",
    "ytick.color": "#8b949e",
    "text.color": "#c9d1d9",
    "legend.framealpha": 0.15,
    "legend.edgecolor": "#30363d",
    "legend.labelcolor": "#c9d1d9",
    "lines.linewidth": 1.4,
    "font.family": "sans-serif",
})

_CMAP  = ["#58a6ff", "#f78166", "#56d364", "#d2a8ff"]   # blue, red, green, purple
_LABEL = {"mc_120": "MC 120 K", "mc_250": "MC 250 K",
          "md_120": "MD 120 K", "md_250": "MD 250 K"}
_RAD2DEG = 180.0 / np.pi


def _deg_formatter():
    return ticker.FuncFormatter(lambda x, _: f"{x:.0f}°")


def _save(fig, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  saved → {path}")


# ── 1. Dihedral time series ─────────────────────────────────────────────────

def plot_dihedral_timeseries(trajs_dict: dict, out_path: str):
    """4-panel dihedral φ₁(t) for MC/MD × 120K/250K."""
    keys = ["mc_120", "mc_250", "md_120", "md_250"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharey=True)
    fig.suptitle("Dihedral φ₁ Time Series", fontsize=14, color="#e6edf3", y=1.01)

    for ax, key, colour in zip(axes.flat, keys, _CMAP):
        traj = trajs_dict[key]
        n    = len(traj)
        ax.plot(np.arange(n), traj * _RAD2DEG, color=colour, lw=0.4, alpha=0.85)
        ax.set_title(_LABEL[key], fontsize=11)
        ax.set_xlabel("MC/MD step")
        ax.set_ylabel("φ₁ [°]")
        ax.set_ylim(-185, 185)
        ax.yaxis.set_major_formatter(_deg_formatter())

    fig.tight_layout()
    _save(fig, out_path)


# ── 2. Baseline distributions ───────────────────────────────────────────────

def plot_baseline_distributions(trajs_dict: dict, cfg: dict, out_path: str):
    """4-panel dihedral histograms."""
    keys  = ["mc_120", "mc_250", "md_120", "md_250"]
    n_bins = cfg["simulation"]["n_bins"]
    edges  = np.linspace(-np.pi, np.pi, n_bins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:]) * _RAD2DEG

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharey=False)
    fig.suptitle("φ₁ Dihedral Distributions", fontsize=14, color="#e6edf3", y=1.01)

    for ax, key, colour in zip(axes.flat, keys, _CMAP):
        traj   = trajs_dict[key]
        counts, _ = np.histogram(traj, bins=edges)
        probs  = counts / counts.sum()
        ax.bar(centres, probs, width=(_RAD2DEG * (edges[1] - edges[0])) * 0.9,
               color=colour, alpha=0.75, edgecolor="none")
        ax.set_title(_LABEL[key], fontsize=11)
        ax.set_xlabel("φ₁ [°]")
        ax.set_ylabel("Probability")
        ax.xaxis.set_major_formatter(_deg_formatter())

    fig.tight_layout()
    _save(fig, out_path)


# ── 3. Baseline PMF ─────────────────────────────────────────────────────────

def plot_baseline_pmf(trajs_dict: dict, T_list: list, cfg: dict, out_path: str):
    """Boltzmann-inversion PMF at both temperatures (MC only)."""
    n_bins = cfg["simulation"]["n_bins"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
    fig.suptitle("Baseline PMF (Boltzmann Inversion)", fontsize=14, color="#e6edf3")

    pairs = [("mc_120", T_list[0], _CMAP[0], "MC 120 K"),
             ("mc_250", T_list[1], _CMAP[1], "MC 250 K")]

    for ax, (key, T, colour, label) in zip(axes, pairs):
        traj = trajs_dict[key]
        centres, pmf = boltzmann_pmf(traj, T, n_bins)
        ax.plot(centres * _RAD2DEG, pmf, color=colour, lw=2.0)
        ax.fill_between(centres * _RAD2DEG, pmf,
                        alpha=0.15, color=colour)
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("φ₁ [°]")
        ax.set_ylabel("F(φ₁) [K]")
        ax.xaxis.set_major_formatter(_deg_formatter())
        ax.set_xlim(-180, 180)

    fig.tight_layout()
    _save(fig, out_path)


# ── 4. Entropy curves ───────────────────────────────────────────────────────

def plot_entropy_curves(trajs_dict: dict, cfg: dict, out_path: str):
    """S(t) vs step for all 4 baseline runs."""
    keys   = ["mc_120", "mc_250", "md_120", "md_250"]
    n_bins = cfg["simulation"]["n_bins"]
    edges  = np.linspace(-np.pi, np.pi, n_bins + 1)

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.set_title("Exploration Entropy S(t) vs Step", fontsize=13, color="#e6edf3")
    ax.set_xlabel("MC/MD step")
    ax.set_ylabel("Shannon entropy S [nats]")

    for key, colour in zip(keys, _CMAP):
        traj  = trajs_dict[key]
        n     = len(traj)
        # Compute cumulative entropy every 1000 steps for speed
        stride = max(1, n // 500)
        checkpoints = np.arange(stride, n + 1, stride)
        entropies = []
        for t in checkpoints:
            counts, _ = np.histogram(traj[:t], bins=edges)
            probs = counts / counts.sum()
            probs = probs[probs > 0]
            entropies.append(-np.sum(probs * np.log(probs)))
        ax.plot(checkpoints, entropies, color=colour, label=_LABEL[key], lw=1.8)

    # Reference: maximum entropy = ln(n_bins)
    ax.axhline(np.log(n_bins), ls="--", color="#8b949e", lw=1.0,
               label=f"S_max = ln({n_bins}) = {np.log(n_bins):.2f}")
    ax.legend(fontsize=9)
    fig.tight_layout()
    _save(fig, out_path)


# ── 5. US window histograms ─────────────────────────────────────────────────

def plot_us_window_histograms(trajs: list, phi0s: np.ndarray,
                               cfg: dict, out_path: str):
    """18 overlapping biased histograms, one colour per window."""
    n_bins = cfg["simulation"]["n_bins"]
    edges  = np.linspace(-np.pi, np.pi, n_bins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:]) * _RAD2DEG

    cmap = plt.cm.hsv(np.linspace(0, 0.9, len(trajs)))
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_title("Umbrella Sampling — Biased Window Histograms", fontsize=13, color="#e6edf3")
    ax.set_xlabel("φ₁ [°]")
    ax.set_ylabel("Probability")
    ax.xaxis.set_major_formatter(_deg_formatter())

    for i, (traj, phi0, colour) in enumerate(zip(trajs, phi0s, cmap)):
        counts, _ = np.histogram(traj, bins=edges)
        probs = counts / counts.sum()
        ax.plot(centres, probs, color=colour, lw=1.2, alpha=0.8,
                label=f"{phi0 * _RAD2DEG:.0f}°" if i % 3 == 0 else None)

    ax.legend(fontsize=7, ncol=3, title="φ₀", title_fontsize=8)
    ax.set_xlim(-180, 180)
    fig.tight_layout()
    _save(fig, out_path)


# ── 6. WHAM PMF vs baseline PMF ────────────────────────────────────────────

def plot_wham_pmf(bin_centres: np.ndarray, pmf_wham: np.ndarray,
                   trajs_dict: dict, T_us: float, cfg: dict, out_path: str):
    """Unbiased WHAM PMF overlaid on the 120K MC baseline PMF."""
    n_bins = cfg["simulation"]["n_bins"]
    mc_traj = trajs_dict["mc_120"]
    centres_bl, pmf_bl = boltzmann_pmf(mc_traj, T_us, n_bins)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title("WHAM PMF vs Baseline PMF (120 K)", fontsize=13, color="#e6edf3")
    ax.set_xlabel("φ₁ [°]")
    ax.set_ylabel("F(φ₁) [K]")
    ax.xaxis.set_major_formatter(_deg_formatter())

    ax.plot(bin_centres * _RAD2DEG, pmf_wham, color=_CMAP[3], lw=2.5,
            label="WHAM (umbrella sampling)")
    ax.plot(centres_bl * _RAD2DEG, pmf_bl, color=_CMAP[0], lw=2.0,
            ls="--", label="Baseline MC 120 K")

    ax.set_xlim(-180, 180)
    ax.legend(fontsize=10)
    fig.tight_layout()
    _save(fig, out_path)


# ── 7. PMF comparison side-by-side ─────────────────────────────────────────

def plot_pmf_comparison(bin_centres: np.ndarray, pmf_wham: np.ndarray,
                         trajs_dict: dict, T_us: float, cfg: dict, out_path: str):
    """WHAM PMF vs Boltzmann PMF side-by-side panel."""
    n_bins = cfg["simulation"]["n_bins"]
    mc_traj = trajs_dict["mc_120"]
    centres_bl, pmf_bl = boltzmann_pmf(mc_traj, T_us, n_bins)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
    fig.suptitle("PMF Comparison: WHAM vs Boltzmann Inversion (120 K)",
                 fontsize=13, color="#e6edf3")

    for ax, pmf, centres, colour, title in [
        (ax1, pmf_wham,  bin_centres, _CMAP[3], "WHAM PMF"),
        (ax2, pmf_bl,    centres_bl,  _CMAP[0], "Boltzmann PMF (MC 120 K)"),
    ]:
        ax.plot(centres * _RAD2DEG, pmf, color=colour, lw=2.2)
        ax.fill_between(centres * _RAD2DEG, pmf, alpha=0.12, color=colour)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("φ₁ [°]")
        ax.set_ylabel("F(φ₁) [K]")
        ax.xaxis.set_major_formatter(_deg_formatter())
        ax.set_xlim(-180, 180)

    fig.tight_layout()
    _save(fig, out_path)
