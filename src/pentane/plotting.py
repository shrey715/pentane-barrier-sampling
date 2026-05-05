"""plotting.py — Matplotlib figures for the pentane barrier sampling project."""
from pentane.analysis import boltzmann_pmf
from pathlib import Path
import seaborn as sns
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use("Agg")

sns.set_theme(style="whitegrid", context="notebook", font_scale=1.1)
plt.rcParams.update(
    {"figure.dpi": 150, "lines.linewidth": 1.4, "font.family": "sans-serif"})

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


def _count_crossings(traj: np.ndarray, threshold_deg: float = 90.0) -> int:
    """Count transitions that cross ±threshold through the barrier region.

    A crossing is defined as a step where φ passes through +threshold or
    -threshold (i.e., a trans↔gauche transition).
    """
    threshold = np.radians(threshold_deg)
    crossings = 0
    for i in range(1, len(traj)):
        a, b = traj[i - 1], traj[i]
        if (a < threshold and b >= threshold) or (a >= threshold and b < threshold):
            crossings += 1
        if (a > -threshold and b <= -threshold) or (a <= -threshold and b > -threshold):
            crossings += 1
    return crossings


# ── 1. Dihedral time series ──────────────────────────────────────────────────

def plot_dihedral_timeseries(trajs: dict, out_path: str):
    """4-panel φ₁(t) — vertically stacked, full width, shared x-axis.

    Each panel is annotated with the total number of barrier crossings
    (transitions through ±90°) observed in the trajectory.
    """
    keys = ["mc_120", "mc_250", "md_120", "md_250"]

    fig, axes = plt.subplots(
        4, 1,
        figsize=(16, 10),
        sharex=True,
        sharey=True,
        constrained_layout=True,
        gridspec_kw={"hspace": 0.08},
    )
    fig.suptitle("Dihedral φ₁ Time Series", fontsize=14)

    for ax, key, colour in zip(axes, keys, COLORS):
        traj = trajs[key]
        steps = np.arange(len(traj))
        ax.plot(steps, traj * R2D, color=colour,
                lw=0.3, alpha=0.75, rasterized=True)
        ax.set_ylim(-185, 185)
        ax.yaxis.set_major_formatter(_deg_fmt())
        ax.yaxis.set_major_locator(ticker.MultipleLocator(60))
        ax.set_ylabel("φ₁ [°]", fontsize=9)
        # Label on right margin
        ax.annotate(
            LABELS[key],
            xy=(1.0, 0.5), xycoords="axes fraction",
            xytext=(6, 0), textcoords="offset points",
            va="center", ha="left", fontsize=10, color=colour,
            fontweight="bold",
        )
        # Barrier-crossing count annotation
        n_cross = _count_crossings(traj)
        ax.annotate(
            f"{n_cross} crossings",
            xy=(0.01, 0.88), xycoords="axes fraction",
            fontsize=8.5, color=colour, fontstyle="italic",
            bbox=dict(boxstyle="round,pad=0.2",
                      fc="white", alpha=0.7, ec="none"),
        )
        ax.tick_params(axis="x", labelbottom=False)
        # Subtle horizontal guide lines
        for phi_ref in (-120, 0, 120):
            ax.axhline(phi_ref, ls=":", lw=0.6, color="grey", alpha=0.5)
        # ±90° barrier reference
        ax.axhline(90, ls="--", lw=0.8, color="red", alpha=0.35)
        ax.axhline(-90, ls="--", lw=0.8, color="red", alpha=0.35)

    axes[-1].tick_params(axis="x", labelbottom=True)
    axes[-1].set_xlabel("Step", fontsize=11)

    _save(fig, out_path)


# ── 2. Baseline distributions ────────────────────────────────────────────────

def plot_baseline_distributions(trajs: dict, n_bins: int, out_path: str):
    """4-panel dihedral probability histograms + MC-vs-MD overlay figure."""
    edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:]) * R2D
    bar_w = R2D * (edges[1] - edges[0]) * 0.9

    # — Figure A: individual histograms —
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharey=False)
    fig.suptitle("φ₁ Dihedral Distributions", fontsize=14, y=1.01)

    for ax, key, colour in zip(axes.flat, ["mc_120", "mc_250", "md_120", "md_250"], COLORS):
        counts, _ = np.histogram(trajs[key], bins=edges)
        probs = counts / counts.sum()
        ax.bar(centres, probs, width=bar_w, color=colour,
               alpha=0.75, edgecolor="none")
        ax.set_title(LABELS[key], fontsize=11)
        ax.set_xlabel("φ₁ [°]")
        ax.set_ylabel("Probability")
        ax.xaxis.set_major_formatter(_deg_fmt())

    fig.tight_layout()
    _save(fig, out_path)

    # — Figure B: MC vs MD overlay at each temperature —
    overlay_path = str(Path(out_path).with_name(
        "baseline_mc_vs_md_overlay.png"))
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
    fig2.suptitle("φ₁ Distributions: MC vs MD", fontsize=14)

    for ax, T_tag, T_label, c_mc, c_md in [
        (ax1, "120", "120 K", COLORS[0], COLORS[2]),
        (ax2, "250", "250 K", COLORS[1], COLORS[3]),
    ]:
        for key, colour, ls, lw, alpha in [
            (f"mc_{T_tag}", c_mc, "-",  2.0, 0.80),
            (f"md_{T_tag}", c_md, "--", 2.0, 0.80),
        ]:
            counts, _ = np.histogram(trajs[key], bins=edges)
            probs = counts / counts.sum()
            ax.step(np.append(centres, centres[-1] + (centres[1]-centres[0])),
                    np.append(probs, probs[-1]),
                    where="post", color=colour, ls=ls, lw=lw, alpha=alpha,
                    label=LABELS[key])
        ax.set_title(f"MC vs MD — {T_label}", fontsize=12)
        ax.set_xlabel("φ₁ [°]")
        ax.set_ylabel("Probability")
        ax.xaxis.set_major_formatter(_deg_fmt())
        ax.legend(fontsize=10)

    fig2.tight_layout()
    _save(fig2, overlay_path)


# ── 3. Baseline PMF ──────────────────────────────────────────────────────────

def plot_baseline_pmf(trajs: dict, T_list: list, n_bins: int, out_path: str):
    """Boltzmann-inversion PMF at both temperatures: MC and MD overlaid."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Baseline PMF (Boltzmann Inversion) — MC and MD", fontsize=14)

    for ax, (T_tag, T, mc_colour, md_colour) in zip(axes, [
        ("120", T_list[0], COLORS[0], COLORS[2]),
        ("250", T_list[1], COLORS[1], COLORS[3]),
    ]):
        mc_key, md_key = f"mc_{T_tag}", f"md_{T_tag}"
        centres_mc, pmf_mc = boltzmann_pmf(trajs[mc_key], T, n_bins)
        centres_md, pmf_md = boltzmann_pmf(trajs[md_key], T, n_bins)

        ax.plot(centres_mc * R2D, pmf_mc, color=mc_colour, lw=2.0,
                label=f"MC {T_tag} K")
        ax.fill_between(centres_mc * R2D, pmf_mc, alpha=0.12, color=mc_colour)
        ax.plot(centres_md * R2D, pmf_md, color=md_colour, lw=2.0,
                ls="--", label=f"MD {T_tag} K")
        ax.fill_between(centres_md * R2D, pmf_md, alpha=0.08, color=md_colour)

        ax.set_title(f"{T_tag} K", fontsize=11)
        ax.set_xlabel("φ₁ [°]")
        ax.set_ylabel("F(φ₁) [K]")
        ax.xaxis.set_major_formatter(_deg_fmt())
        ax.set_xlim(-180, 180)
        ax.legend(fontsize=10)

    fig.tight_layout()
    _save(fig, out_path)


# ── 4. Entropy curves ────────────────────────────────────────────────────────

def plot_entropy_curves(trajs: dict, n_bins: int, out_path: str,
                        enhanced_trajs: dict = None):
    """Cumulative Shannon entropy S(t) for baseline and (optionally) enhanced runs.

    Parameters
    ----------
    trajs          : dict  Baseline trajectories keyed by e.g. 'mc_120'.
    n_bins         : int
    out_path       : str
    enhanced_trajs : dict, optional
        Extra trajectories to overlay (e.g. ``{"remd_120": arr, "umbrella_120": arr}``).
        Plotted as dashed lines with distinct colours.
        These must already be sub-sampled to the same length as the baseline
        trajectories so the x-axis is directly comparable.
    """
    edges = np.linspace(-np.pi, np.pi, n_bins + 1)

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.set_title("Exploration Entropy S(t) vs Step", fontsize=13)
    ax.set_xlabel("Step")
    ax.set_ylabel("Shannon entropy S [nats]")

    def _entropy_curve(traj):
        n = len(traj)
        stride = max(1, n // 500)
        checkpoints = np.arange(stride, n + 1, stride)
        entropies = []
        for t in checkpoints:
            counts, _ = np.histogram(traj[:t], bins=edges)
            p = counts / counts.sum()
            p = p[p > 0]
            entropies.append(-np.sum(p * np.log(p)))
        return checkpoints, np.asarray(entropies)

    for key, colour in zip(["mc_120", "mc_250", "md_120", "md_250"], COLORS):
        chk, ent = _entropy_curve(trajs[key])
        ax.plot(chk, ent, color=colour, label=LABELS[key], lw=1.8)

    # Enhanced trajectories — dashed lines with distinct colours
    if enhanced_trajs:
        _ENH_COLORS = {"remd_120": "#e377c2", "umbrella_120": "#17becf",
                       "remd_250": "#bcbd22",  "umbrella_250": "#ff7f0e"}
        _ENH_LABELS = {"remd_120": "REMD 120 K",      "umbrella_120": "Umbrella 120 K",
                       "remd_250": "REMD 250 K",       "umbrella_250": "Umbrella 250 K"}
        for key, traj in enhanced_trajs.items():
            colour = _ENH_COLORS.get(key, "black")
            label = _ENH_LABELS.get(key, key)
            chk, ent = _entropy_curve(traj)
            ax.plot(chk, ent, color=colour, label=label, lw=2.2, ls="--")

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
    ax.set_title(
        f"WHAM Convergence: Free-Energy Offsets{T_label}", fontsize=13)
    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"$f_i$ [K]")
    for i in range(n_windows):
        ax.plot(np.arange(1, n_iter + 1),
                f_history[:, i], color=cmap[i], lw=1.0, alpha=0.9)
    ax.set_xlim(1, n_iter)
    fig.tight_layout()
    _save(fig, out_path)


# ── 7. WHAM PMF vs baseline ──────────────────────────────────────────────────

def plot_wham_pmf(bin_centres: np.ndarray, pmf_wham: np.ndarray,
                  trajs: dict, T: float, out_path: str, n_bins: int = 36):
    """WHAM PMF overlaid on both the MC and MD baseline PMFs."""
    T_tag = int(round(T))
    mc_key = f"mc_{T_tag}"
    md_key = f"md_{T_tag}"
    centres_mc, pmf_mc = boltzmann_pmf(trajs[mc_key], T, n_bins)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title(f"WHAM PMF vs Baseline PMF ({T:.0f} K)", fontsize=13)
    ax.set_xlabel("φ₁ [°]")
    ax.set_ylabel("F(φ₁) [K]")
    ax.xaxis.set_major_formatter(_deg_fmt())

    ax.plot(bin_centres * R2D, pmf_wham, color=COLORS[3], lw=2.5,
            label="WHAM (umbrella sampling)")
    ax.plot(centres_mc * R2D, pmf_mc, color=COLORS[0], lw=2.0,
            ls="--", label=f"Baseline MC {T_tag} K")

    # MD line if available
    if md_key in trajs:
        centres_md, pmf_md = boltzmann_pmf(trajs[md_key], T, n_bins)
        ax.plot(centres_md * R2D, pmf_md, color=COLORS[2], lw=2.0,
                ls=":", label=f"Baseline MD {T_tag} K")

    ax.set_xlim(-180, 180)
    ax.legend(fontsize=10)
    fig.tight_layout()
    _save(fig, out_path)


# ── 8. PMF side-by-side comparison ──────────────────────────────────────────

def plot_pmf_comparison(bin_centres: np.ndarray, pmf_wham: np.ndarray,
                        trajs: dict, T: float, out_path: str, n_bins: int = 36):
    """WHAM PMF vs MC and MD Boltzmann PMFs, side-by-side."""
    T_tag = int(round(T))
    mc_key = f"mc_{T_tag}"
    md_key = f"md_{T_tag}"
    centres_mc, pmf_mc = boltzmann_pmf(trajs[mc_key], T, n_bins)

    # Determine number of panels
    has_md = md_key in trajs
    n_panels = 3 if has_md else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))
    fig.suptitle(
        f"PMF Comparison: WHAM vs Boltzmann Inversion ({T:.0f} K)", fontsize=13)

    panels = [
        (axes[0], pmf_wham,  bin_centres, COLORS[3], "WHAM PMF"),
        (axes[1], pmf_mc,    centres_mc,  COLORS[0],
         f"Boltzmann PMF (MC {T_tag} K)"),
    ]
    if has_md:
        centres_md, pmf_md = boltzmann_pmf(trajs[md_key], T, n_bins)
        panels.append((axes[2], pmf_md, centres_md,
                      COLORS[2], f"Boltzmann PMF (MD {T_tag} K)"))

    for ax, pmf, centres, colour, title in panels:
        ax.plot(centres * R2D, pmf, color=colour, lw=2.2)
        ax.fill_between(centres * R2D, pmf, alpha=0.12, color=colour)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("φ₁ [°]")
        ax.set_ylabel("F(φ₁) [K]")
        ax.xaxis.set_major_formatter(_deg_fmt())
        ax.set_xlim(-180, 180)

    fig.tight_layout()
    _save(fig, out_path)


# ── 9. Early exploration score bar chart ─────────────────────────────────────

def plot_early_exploration_bar(scores: dict, out_path: str):
    """Bar chart of early exploration scores across methods.

    Parameters
    ----------
    scores : dict
        Mapping of method label → score (nats).
        E.g. {"MC 120 K": 2.1, "MD 120 K": 1.8, "Umbrella 120 K": 3.5}
    out_path : str
    """
    labels = list(scores.keys())
    values = [scores[k] for k in labels]

    # Colour palette: reuse COLORS then cycle through enhanced colours
    palette = ["#4C72B0", "#DD8452", "#55A868", "#8172B2",
               "#17becf", "#ff7f0e", "#e377c2", "#bcbd22"]
    colours = [palette[i % len(palette)] for i in range(len(labels))]

    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 1.4), 5))
    ax.set_title("Early Exploration Score by Method", fontsize=13)
    ax.set_ylabel("Early exploration score E [nats]", fontsize=11)

    bars = ax.bar(labels, values, color=colours, alpha=0.82,
                  edgecolor="white", linewidth=0.8)

    # Annotate bars with numeric values
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02 * max(values),
            f"{val:.3f}",
            ha="center", va="bottom", fontsize=9.5, fontweight="bold",
        )

    ax.set_ylim(0, max(values) * 1.18)
    ax.axhline(np.log(36), ls="--", color="grey", lw=1.0, alpha=0.7,
               label=f"S_max = {np.log(36):.2f} nats")
    ax.legend(fontsize=9)
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    _save(fig, out_path)
