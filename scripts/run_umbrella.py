"""
run_umbrella.py — Umbrella sampling + WHAM + plots.

Steps:
    1. Loop over 36 window centres at the 10° histogram midpoints
    2. Run run_window(phi0, T, cfg, seed) for each window at both baseline
       temperatures
    3. Save per-window trajectories to results/trajectories/us_window_<T>K_<i>.npy
    4. Run WHAM to get the unbiased PMF and convergence history
    5. Generate temperature-specific histogram, convergence, and PMF plots

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
N_W  = _us["n_windows"]              # 36
T_LIST = [float(t) for t in CFG["simulation"]["temperatures_K"]]


def _temperature_tag(T: float) -> str:
    return f"{T:.0f}K"


def _window_centres(cfg: dict) -> np.ndarray:
    n_windows = int(cfg["umbrella"]["n_windows"])
    step = 2.0 * np.pi / n_windows
    return -np.pi + 0.5 * step + step * np.arange(n_windows)


def _load_baseline_trajs() -> dict:
    """Load baseline MC trajectories if available, else return empty dict."""
    trajs = {}
    for key in ("mc_120", "mc_250", "md_120", "md_250"):
        p = TRAJ_DIR / f"{key}.npy"
        if p.exists():
            trajs[key] = np.load(p)
    return trajs


def _save_outputs_for_temperature(T_us: float, stem: str, value, primary: bool = False):
    tag = _temperature_tag(T_us)
    np.save(RESULTS / f"{stem}_{tag}.npy", value)
    if primary:
        np.save(RESULTS / f"{stem}.npy", value)


def _run_temperature(T_us: float, baseline_trajs: dict, seed_base: int = 0) -> None:
    tag = _temperature_tag(T_us)
    phi0s = _window_centres(CFG)

    print("\n" + "=" * 60)
    print(f"Umbrella Sampling — {N_W} windows at T = {T_us:.0f} K")
    print(f"  n_steps_per_window = {_us['n_steps_per_window']}")
    print(f"  k = {_us['window_k_K_per_rad2']} K/rad²")
    print("=" * 60)

    trajs_us = []
    for i, phi0 in enumerate(phi0s):
        path = TRAJ_DIR / f"us_window_{tag}_{i:02d}.npy"
        if path.exists():
            print(f"  Window {i:02d} ({np.degrees(phi0):+.1f}°): loaded from cache")
            trajs_us.append(np.load(path))
        else:
            print(f"  Window {i:02d} ({np.degrees(phi0):+.1f}°): running … ", end="", flush=True)
            t0 = time.perf_counter()
            traj = run_window(phi0, T_us, CFG, seed=seed_base + i)
            dt = time.perf_counter() - t0
            np.save(path, traj)
            trajs_us.append(traj)
            print(f"done in {dt:.1f}s")

    print("\nRunning WHAM … ", end="", flush=True)
    t0 = time.perf_counter()
    bin_centres, pmf_wham, f_history = wham(trajs_us, phi0s, CFG, T_us, return_history=True)
    print(f"done in {time.perf_counter() - t0:.2f}s")

    primary = np.isclose(T_us, T_LIST[0])
    _save_outputs_for_temperature(T_us, "wham_bin_centres", bin_centres, primary=primary)
    _save_outputs_for_temperature(T_us, "wham_pmf", pmf_wham, primary=primary)
    _save_outputs_for_temperature(T_us, "wham_history", f_history, primary=primary)

    finite_pmf = np.where(np.isfinite(pmf_wham), pmf_wham, np.nan)
    _trans_region = np.abs(bin_centres) > np.radians(150)
    _gauche_region = (np.abs(bin_centres) > np.radians(40)) & (np.abs(bin_centres) < np.radians(80))
    trans_vals = finite_pmf[_trans_region]
    trans_vals = trans_vals[np.isfinite(trans_vals)]
    gauche_vals = finite_pmf[_gauche_region]
    gauche_vals = gauche_vals[np.isfinite(gauche_vals)]

    if trans_vals.size > 0 and gauche_vals.size > 0:
        E_trans = np.nanmin(trans_vals)
        E_gauche = np.nanmin(gauche_vals)
        E_barrier = np.nanmax(finite_pmf)
        print(f"\n  PMF summary (WHAM):")
        print(f"    Trans minimum  : {E_trans:.1f} K")
        print(f"    Gauche minimum : {E_gauche:.1f} K")
        print(f"    Barrier height : {E_barrier:.1f} K")

    baseline_key = f"mc_{int(round(T_us))}"
    overlay_trajs = dict(baseline_trajs)
    if baseline_key not in baseline_trajs:
        fallback_key = f"mc_{int(round(T_LIST[0]))}"
        print(f"\n  Warning: {baseline_key} not found, falling back to {fallback_key}.")
        if fallback_key in baseline_trajs:
            overlay_trajs[baseline_key] = baseline_trajs[fallback_key]

    print("\nGenerating plots …")
    plot_us_window_histograms(trajs_us, phi0s, CFG,
        str(RESULTS / f"us_window_histograms_{tag}.png"))
    plot_wham_convergence(f_history,
        str(RESULTS / f"wham_convergence_{tag}.png"))
    plot_wham_pmf(bin_centres, pmf_wham, overlay_trajs, T_us, CFG,
        str(RESULTS / f"wham_pmf_{tag}.png"))
    plot_pmf_comparison(bin_centres, pmf_wham, overlay_trajs, T_us, CFG,
        str(RESULTS / f"pmf_comparison_{tag}.png"))

    if primary:
        np.save(RESULTS / "wham_bin_centres.npy", bin_centres)
        np.save(RESULTS / "wham_pmf.npy", pmf_wham)
        np.save(RESULTS / "wham_history.npy", f_history)
        plot_us_window_histograms(trajs_us, phi0s, CFG,
            str(RESULTS / "us_window_histograms.png"))
        plot_wham_convergence(f_history,
            str(RESULTS / "wham_convergence.png"))
        plot_wham_pmf(bin_centres, pmf_wham, overlay_trajs, T_us, CFG,
            str(RESULTS / "wham_pmf.png"))
        plot_pmf_comparison(bin_centres, pmf_wham, overlay_trajs, T_us, CFG,
            str(RESULTS / "pmf_comparison.png"))


def main():
    baseline_trajs = _load_baseline_trajs()

    print("=" * 60)
    print("Umbrella Sampling + WHAM")
    print(f"  temperatures = {T_LIST}")
    print(f"  n_windows    = {N_W}")
    print(f"  n_steps      = {_us['n_steps_per_window']}")
    print("=" * 60)

    for idx, T_us in enumerate(T_LIST):
        _run_temperature(T_us, baseline_trajs, seed_base=1000 * (idx + 1))

    print("\nAll umbrella sampling outputs written to results/")


if __name__ == "__main__":
    main()
