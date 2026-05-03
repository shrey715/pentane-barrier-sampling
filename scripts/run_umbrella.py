"""
run_umbrella.py — Task 3 + 4: Umbrella sampling + WHAM + plots.

Steps:
    1. Loop over 36 window centres at the 10° histogram midpoints
    2. Run run_window(phi0, T=120 K, cfg, seed=i) for each window
    3. Save per-window trajectories to results/trajectories/us_window_<i>.npy
    4. Run WHAM to get the unbiased PMF and convergence history
    5. Generate: us_window_histograms.png, wham_convergence.png, wham_pmf.png,
         pmf_comparison.png

Usage:
    python scripts/run_umbrella.py
"""
import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from pentane.config_loader import CFG
from pentane.umbrella import run_window
from pentane.wham import wham
from pentane.plotting import (
    plot_us_window_histograms,
    plot_wham_convergence,
    plot_wham_pmf,
    plot_pmf_comparison,
)

RESULTS  = Path(__file__).parents[1] / "results"
TRAJ_DIR = RESULTS / "trajectories"
TRAJ_DIR.mkdir(parents=True, exist_ok=True)

_us  = CFG["umbrella"]
T_US = _us["temperature_K"]          # 120.0 K
N_W  = _us["n_windows"]              # 18


def _load_baseline_trajs() -> dict:
    """Load baseline MC trajectories if available, else return empty dict."""
    trajs = {}
    for key in ("mc_120", "mc_250", "md_120", "md_250"):
        p = TRAJ_DIR / f"{key}.npy"
        if p.exists():
            trajs[key] = np.load(p)
    return trajs


def main():
    step = 2.0 * np.pi / N_W
    phi0s = -np.pi + 0.5 * step + step * np.arange(N_W)

    print("=" * 60)
    print(f"Umbrella Sampling — {N_W} windows at T = {T_US} K")
    print(f"  n_steps_per_window = {_us['n_steps_per_window']}")
    print(f"  k = {_us['window_k_K_per_rad2']} K/rad²")
    print("=" * 60)

    # ── Run windows ───────────────────────────────────────────────────────
    trajs_us = []
    for i, phi0 in enumerate(phi0s):
        path = TRAJ_DIR / f"us_window_{i:02d}.npy"
        if path.exists():
            print(f"  Window {i:02d} ({np.degrees(phi0):+.1f}°): loaded from cache")
            trajs_us.append(np.load(path))
        else:
            print(f"  Window {i:02d} ({np.degrees(phi0):+.1f}°): running … ", end="", flush=True)
            t0 = time.perf_counter()
            traj = run_window(phi0, T_US, CFG, seed=i)
            dt = time.perf_counter() - t0
            np.save(path, traj)
            trajs_us.append(traj)
            print(f"done in {dt:.1f}s")

    # ── WHAM ─────────────────────────────────────────────────────────────
    print("\nRunning WHAM … ", end="", flush=True)
    t0 = time.perf_counter()
    bin_centres, pmf_wham, f_history = wham(trajs_us, phi0s, CFG, T_US, return_history=True)
    print(f"done in {time.perf_counter() - t0:.2f}s")

    # Save WHAM output
    np.save(RESULTS / "wham_bin_centres.npy", bin_centres)
    np.save(RESULTS / "wham_pmf.npy", pmf_wham)
    np.save(RESULTS / "wham_history.npy", f_history)

    # Print barrier heights from WHAM PMF
    finite_pmf = np.where(np.isfinite(pmf_wham), pmf_wham, np.nan)
    _trans_region = np.abs(bin_centres) > np.radians(150)
    _gauche_region = (np.abs(bin_centres) > np.radians(40)) & (np.abs(bin_centres) < np.radians(80))

    if np.any(_trans_region) and np.any(_gauche_region):
        E_trans  = np.nanmin(finite_pmf[_trans_region])
        E_gauche = np.nanmin(finite_pmf[_gauche_region])
        E_barrier = np.nanmax(finite_pmf)
        print(f"\n  PMF summary (WHAM):")
        print(f"    Trans minimum  : {E_trans:.1f} K")
        print(f"    Gauche minimum : {E_gauche:.1f} K")
        print(f"    Barrier height : {E_barrier:.1f} K")

    # ── Load baseline trajectories for overlay ────────────────────────────
    baseline_trajs = _load_baseline_trajs()
    if "mc_120" not in baseline_trajs:
        print("\n  Warning: baseline trajectories not found. "
              "Run scripts/run_baseline.py first for overlay plots.")
        # Create dummy all-trans trajectory for plotting
        baseline_trajs["mc_120"] = np.full(_us["n_steps_per_window"], np.pi)

    # ── Plots ─────────────────────────────────────────────────────────────
    print("\nGenerating plots …")
    plot_us_window_histograms(trajs_us, phi0s, CFG,
        str(RESULTS / "us_window_histograms.png"))
    plot_wham_convergence(f_history,
        str(RESULTS / "wham_convergence.png"))
    plot_wham_pmf(bin_centres, pmf_wham, baseline_trajs, T_US, CFG,
        str(RESULTS / "wham_pmf.png"))
    plot_pmf_comparison(bin_centres, pmf_wham, baseline_trajs, T_US, CFG,
        str(RESULTS / "pmf_comparison.png"))

    print("\nAll umbrella sampling outputs written to results/")


if __name__ == "__main__":
    main()
