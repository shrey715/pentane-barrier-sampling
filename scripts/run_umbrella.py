"""
run_umbrella.py — Umbrella sampling + WHAM for both baseline temperatures.

For each temperature in config:
  1. Run (or load from cache) one MC window per φ₀ centre.
  2. Run WHAM to get the unbiased PMF.
  3. Save results and generate plots.

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
from pentane.analysis import exploration_entropy, early_exploration_score
from pentane.plotting import (
    plot_us_window_histograms,
    plot_wham_convergence,
    plot_wham_pmf,
    plot_pmf_comparison,
)

RESULTS  = Path(__file__).parents[1] / "results"
TRAJ_DIR = RESULTS / "trajectories"
TRAJ_DIR.mkdir(parents=True, exist_ok=True)

T_LIST  = [float(t) for t in CFG["simulation"]["temperatures_K"]]
N_BINS  = CFG["simulation"]["n_bins"]
N_WIN   = CFG["umbrella"]["n_windows"]
K_BIAS  = CFG["umbrella"]["window_k_K_per_rad2"]
N_STEPS = CFG["umbrella"]["n_steps_per_window"]


def window_centres() -> np.ndarray:
    step = 2.0 * np.pi / N_WIN
    return -np.pi + 0.5 * step + step * np.arange(N_WIN)


def load_baseline_trajs() -> dict:
    trajs = {}
    for key in ("mc_120", "mc_250", "md_120", "md_250"):
        p = TRAJ_DIR / f"{key}.npy"
        if p.exists():
            trajs[key] = np.load(p)
    return trajs


def run_temperature(T: float, baseline_trajs: dict, seed_base: int = 0) -> None:
    tag   = f"{T:.0f}K"
    phi0s = window_centres()

    print(f"\n{'='*60}")
    print(f"Umbrella Sampling — {N_WIN} windows at T = {T:.0f} K")
    print(f"  n_steps = {N_STEPS}   k = {K_BIAS} K/rad²")
    print(f"{'='*60}")

    # ── Run / load windows ────────────────────────────────────────────────────
    # Two-phase: load cached windows immediately; dispatch the rest in parallel.
    from concurrent.futures import ProcessPoolExecutor
    from pentane.umbrella import _run_window_worker

    n_workers = CFG.get("umbrella", {}).get("n_workers") or None

    # Phase 1 — sort windows into cached vs. pending
    trajs_us: list = [None] * len(phi0s)
    pending: list  = []   # list of (list-index, task_args_tuple)

    for i, phi0 in enumerate(phi0s):
        cache = TRAJ_DIR / f"us_window_{tag}_{i:02d}.npy"
        if cache.exists():
            print(f"  Window {i:02d} ({np.degrees(phi0):+.1f}°): from cache")
            trajs_us[i] = np.load(cache)
        else:
            pending.append((i, phi0, cache))

    # Phase 2 — run uncached windows in parallel
    if pending:
        print(f"\n  Running {len(pending)} uncached window(s) "
              f"with {n_workers or 'all'} worker(s) …")
        import concurrent.futures
        
        t0 = time.perf_counter()
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            # Submit all tasks and map futures back to their (i, phi0, cache) metadata
            future_to_info = {
                pool.submit(_run_window_worker, (phi0, T, CFG, seed_base + i)): (i, phi0, cache)
                for i, phi0, cache in pending
            }
            
            # As each worker finishes its window, grab the result and print
            for future in concurrent.futures.as_completed(future_to_info):
                i, phi0, cache = future_to_info[future]
                traj = future.result()
                np.save(cache, traj)
                trajs_us[i] = traj
                print(f"  Window {i:02d} ({np.degrees(phi0):+.1f}°): finished and saved")
                
        elapsed = time.perf_counter() - t0
        print(f"  … all done in {elapsed:.1f}s  ({len(pending)} windows × {N_STEPS} steps)")

    # ── WHAM ─────────────────────────────────────────────────────────────────
    print("\nRunning WHAM … ", end="", flush=True)
    t0 = time.perf_counter()
    bin_centres, pmf_wham, f_history = wham(trajs_us, phi0s, CFG, T, return_history=True)
    print(f"done in {time.perf_counter() - t0:.2f}s")

    # Save arrays (tagged + primary for first temperature)
    primary = np.isclose(T, T_LIST[0])
    for stem, val in [("wham_bin_centres", bin_centres),
                      ("wham_pmf", pmf_wham),
                      ("wham_history", f_history)]:
        np.save(RESULTS / f"{stem}_{tag}.npy", val)
        if primary:
            np.save(RESULTS / f"{stem}.npy", val)

    # PMF summary
    finite = np.where(np.isfinite(pmf_wham), pmf_wham, np.nan)
    trans  = finite[np.abs(bin_centres) > np.radians(150)]
    gauche = finite[(np.abs(bin_centres) > np.radians(40)) & (np.abs(bin_centres) < np.radians(80))]
    if trans[np.isfinite(trans)].size and gauche[np.isfinite(gauche)].size:
        print(f"\n  PMF summary (WHAM):")
        print(f"    Trans min   : {np.nanmin(trans):.1f} K")
        print(f"    Gauche min  : {np.nanmin(gauche):.1f} K")
        print(f"    Barrier     : {np.nanmax(finite):.1f} K")

    # ── Exploration entropy / early-exploration score ─────────────────────────
    # S_final and bins_visited come from the WHAM-reconstructed unbiased
    # probabilities — exact, no sub-sampling noise.
    # early_score is computed on the full pooled trajectory with a stride
    # (avoids the 7.2M-point Python loop while remaining unbiased).
    wham_p = np.exp(-pmf_wham / T)
    wham_p = wham_p / wham_p.sum()
    S_final = float(-np.sum(wham_p[wham_p > 0] * np.log(wham_p[wham_p > 0])))
    bins_visited = int(np.sum(wham_p > 0))

    pooled = np.concatenate(trajs_us)
    stride = max(1, len(pooled) // (N_STEPS * 5))   # keep ~1M pts max
    E_score = early_exploration_score(pooled[::stride], N_BINS)

    print(f"\n  Umbrella statistics ({T:.0f} K):")
    print(f"    S(final)     : {S_final:.4f} nats")
    print(f"    early_score  : {E_score:.4f} nats")
    print(f"    bins visited : {bins_visited} / {N_BINS}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    # Make sure the baseline trajectory for this T is available for overlay
    mc_key = f"mc_{int(round(T))}"
    overlay = dict(baseline_trajs)
    if mc_key not in overlay:
        fallback = f"mc_{int(round(T_LIST[0]))}"
        print(f"\n  Warning: {mc_key} not found, falling back to {fallback}.")
        if fallback in baseline_trajs:
            overlay[mc_key] = baseline_trajs[fallback]

    print("\nGenerating plots …")
    plot_us_window_histograms(trajs_us, phi0s,
        str(RESULTS / f"us_window_histograms_{tag}.png"))
    plot_wham_convergence(f_history,
        str(RESULTS / f"wham_convergence_{tag}.png"), T=T)
    plot_wham_pmf(bin_centres, pmf_wham, overlay, T,
        str(RESULTS / f"wham_pmf_{tag}.png"), n_bins=N_BINS)
    plot_pmf_comparison(bin_centres, pmf_wham, overlay, T,
        str(RESULTS / f"pmf_comparison_{tag}.png"), n_bins=N_BINS)

    if primary:
        plot_us_window_histograms(trajs_us, phi0s,
            str(RESULTS / "us_window_histograms.png"))
        plot_wham_convergence(f_history,
            str(RESULTS / "wham_convergence.png"), T=T)
        plot_wham_pmf(bin_centres, pmf_wham, overlay, T,
            str(RESULTS / "wham_pmf.png"), n_bins=N_BINS)
        plot_pmf_comparison(bin_centres, pmf_wham, overlay, T,
            str(RESULTS / "pmf_comparison.png"), n_bins=N_BINS)


def main():
    print("=" * 60)
    print("Umbrella Sampling + WHAM")
    print(f"  temperatures = {T_LIST}  |  n_windows = {N_WIN}  |  n_steps = {N_STEPS}")
    print("=" * 60)

    baseline_trajs = load_baseline_trajs()
    for idx, T in enumerate(T_LIST):
        run_temperature(T, baseline_trajs, seed_base=1000 * (idx + 1))

    print("\nAll umbrella sampling outputs written to results/")


if __name__ == "__main__":
    main()
