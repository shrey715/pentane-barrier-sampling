"""
run_baseline.py — NVT MC and MD baseline runs at 120 K and 250 K.

Outputs (all in results/):
  trajectories/mc_120.npy, mc_250.npy, md_120.npy, md_250.npy
  dihedral_timeseries.png, baseline_distributions.png,
  baseline_pmf.png, entropy_curves.png

Usage:
  python scripts/run_baseline.py
"""
from pentane.plotting import (
    plot_dihedral_timeseries,
    plot_baseline_distributions,
    plot_baseline_pmf,
    plot_entropy_curves,
    plot_early_exploration_bar,
)
from pentane.analysis import exploration_entropy, early_exploration_score
from pentane.md import run_md
from pentane.mc import run_mc
from pentane.config_loader import CFG
import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))


RESULTS = Path(__file__).parents[1] / "results"
TRAJ_DIR = RESULTS / "trajectories"
TRAJ_DIR.mkdir(parents=True, exist_ok=True)

T_LIST = CFG["simulation"]["temperatures_K"]   # [120.0, 250.0]
N_BINS = CFG["simulation"]["n_bins"]
N_STEPS = CFG["simulation"]["n_steps"]

LABELS_SCORE = {
    "mc_120": "MC 120 K",
    "mc_250": "MC 250 K",
    "md_120": "MD 120 K",
    "md_250": "MD 250 K",
}


def _acceptance(traj: np.ndarray) -> float:
    return float(np.mean(np.diff(traj) != 0))


def run_sim(key: str, fn, T: float, seed: int) -> np.ndarray:
    cache = TRAJ_DIR / f"{key}.npy"
    if cache.exists():
        print(f"  {key}: loaded from cache")
        return np.load(cache)
    print(f"  {key} (T={T} K) … ", end="", flush=True)
    t0 = time.perf_counter()
    traj = fn(T, CFG, seed=seed)
    np.save(cache, traj)
    print(
        f"done in {time.perf_counter() - t0:.1f}s  |  acceptance ~ {_acceptance(traj):.1%}")
    return traj


def main():
    print("=" * 60)
    print("Baseline NVT MC + MD")
    print(f"  n_steps = {N_STEPS}   T = {T_LIST}")
    print("=" * 60)

    trajs = {
        "mc_120": run_sim("mc_120", run_mc, T_LIST[0], seed=42),
        "mc_250": run_sim("mc_250", run_mc, T_LIST[1], seed=43),
        "md_120": run_sim("md_120", run_md, T_LIST[0], seed=44),
        "md_250": run_sim("md_250", run_md, T_LIST[1], seed=45),
    }

    print("\nStatistics:")
    early_scores = {}
    for key, traj in trajs.items():
        T = T_LIST[0] if "120" in key else T_LIST[1]
        S = exploration_entropy(traj, N_BINS)
        E = early_exploration_score(traj, N_BINS)
        early_scores[LABELS_SCORE.get(key, key)] = E
        print(f"  {key:<8}  S(final)={S:.4f} nats   early_score={E:.4f}")

    print("\nGenerating plots …")
    plot_dihedral_timeseries(trajs, str(RESULTS / "dihedral_timeseries.png"))
    plot_baseline_distributions(trajs, N_BINS, str(
        RESULTS / "baseline_distributions.png"))
    plot_baseline_pmf(trajs, T_LIST, N_BINS, str(RESULTS / "baseline_pmf.png"))

    # Opportunistically load cached umbrella trajectories to show enhanced
    # sampling coverage in the entropy plot (no recomputation if missing).
    enhanced = {}
    for T, tag in zip(T_LIST, ["120K", "250K"]):
        us_files = sorted(TRAJ_DIR.glob(f"us_window_{tag}_*.npy"))
        if us_files:
            key = f"umbrella_{int(round(T))}"
            pooled = np.concatenate([np.load(f) for f in us_files])
            rng = np.random.default_rng(42)
            idx = rng.choice(len(pooled), size=N_STEPS, replace=False)
            enhanced[key] = pooled[idx]
            print(
                f"  entropy plot: loaded {len(us_files)} umbrella windows for {key} (sub-sampled to {N_STEPS})")

    plot_entropy_curves(trajs, N_BINS, str(RESULTS / "entropy_curves.png"),
                        enhanced_trajs=enhanced if enhanced else None)

    # Add umbrella early-exploration scores if cached windows exist
    for T, tag in zip(T_LIST, ["120K", "250K"]):
        us_files = sorted(TRAJ_DIR.glob(f"us_window_{tag}_*.npy"))
        if us_files:
            pooled = np.concatenate([np.load(f) for f in us_files])
            rng2 = np.random.default_rng(0)
            idx2 = rng2.choice(len(pooled), size=N_STEPS, replace=False)
            us_traj = pooled[idx2]
            E_us = early_exploration_score(us_traj, N_BINS)
            label = f"Umbrella {int(round(T))} K"
            early_scores[label] = E_us
            print(f"  umbrella_{int(round(T))}: early_score={E_us:.4f}")

    plot_early_exploration_bar(
        early_scores,
        str(RESULTS / "early_exploration_scores.png"),
    )

    print("\nAll baseline outputs written to results/")


if __name__ == "__main__":
    main()
