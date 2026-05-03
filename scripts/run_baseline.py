"""
run_baseline.py — Task 2: Baseline NVT MC and MD at 120 K and 250 K.

Outputs (all in results/):
  trajectories/mc_120.npy, mc_250.npy, md_120.npy, md_250.npy
  dihedral_timeseries.png
  baseline_distributions.png
  baseline_pmf.png
  entropy_curves.png

Usage:
  python scripts/run_baseline.py
"""
import sys
import time
import numpy as np
from pathlib import Path

# Ensure src/ is importable when running directly
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from pentane.config_loader import CFG
from pentane.mc import run_mc
from pentane.md import run_md
from pentane.analysis import exploration_entropy, early_exploration_score
from pentane.plotting import (
    plot_dihedral_timeseries,
    plot_baseline_distributions,
    plot_baseline_pmf,
    plot_entropy_curves,
)

RESULTS = Path(__file__).parents[1] / "results"
TRAJ_DIR = RESULTS / "trajectories"
TRAJ_DIR.mkdir(parents=True, exist_ok=True)

T_LIST = CFG["simulation"]["temperatures_K"]   # [120.0, 250.0]


def _run_and_save(key: str, fn, T: float, seed: int) -> np.ndarray:
    print(f"  Running {key} (T={T} K) … ", end="", flush=True)
    t0 = time.perf_counter()
    traj = fn(T, CFG, seed=seed)
    dt = time.perf_counter() - t0
    np.save(TRAJ_DIR / f"{key}.npy", traj)
    print(f"done in {dt:.1f}s  |  acceptance ~ {_acceptance(traj):.1%}")
    return traj


def _acceptance(traj: np.ndarray) -> float:
    """Fraction of steps where phi1 actually changed."""
    return float(np.mean(np.diff(traj) != 0))


def main():
    print("=" * 60)
    print("Baseline NVT MC + MD")
    print(f"  n_steps = {CFG['simulation']['n_steps']}")
    print(f"  T       = {T_LIST}")
    print("=" * 60)

    trajs = {}
    # MC at both temperatures
    trajs["mc_120"] = _run_and_save("mc_120", run_mc, T_LIST[0], seed=42)
    trajs["mc_250"] = _run_and_save("mc_250", run_mc, T_LIST[1], seed=43)
    # MD at both temperatures
    trajs["md_120"] = _run_and_save("md_120", run_md, T_LIST[0], seed=44)
    trajs["md_250"] = _run_and_save("md_250", run_md, T_LIST[1], seed=45)

    # ── Print statistics ──────────────────────────────────────────────────
    print("\nStatistics:")
    for key, traj in trajs.items():
        T = T_LIST[0] if "120" in key else T_LIST[1]
        S  = exploration_entropy(traj, CFG["simulation"]["n_bins"])
        E  = early_exploration_score(traj, CFG["simulation"]["n_bins"])
        print(f"  {key:<8}  S(final)={S:.4f} nats   early_score={E:.4f}")

    # ── Plots ─────────────────────────────────────────────────────────────
    print("\nGenerating plots …")
    plot_dihedral_timeseries(trajs,
        str(RESULTS / "dihedral_timeseries.png"))
    plot_baseline_distributions(trajs, CFG,
        str(RESULTS / "baseline_distributions.png"))
    plot_baseline_pmf(trajs, T_LIST, CFG,
        str(RESULTS / "baseline_pmf.png"))
    plot_entropy_curves(trajs, CFG,
        str(RESULTS / "entropy_curves.png"))

    print("\nAll baseline outputs written to results/")


if __name__ == "__main__":
    main()
