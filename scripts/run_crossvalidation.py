"""
run_crossvalidation.py - Cross-validate REMD (OpenMM) against umbrella+WHAM.

The script compares three PMFs at a shared temperature:
  - REMD PMF reweighted from remd/remd_output/*.csv
  - Umbrella+WHAM PMF from results/wham_pmf*.npy
  - Baseline PMF built from the saved MC and MD trajectories

Outputs:
  - results/plots/crossvalidation_<T>K.png
  - results/report/crossvalidation_<T>K.txt
"""
import argparse
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

ROOT = Path(__file__).parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(ROOT))

from pentane.analysis import boltzmann_pmf
from pentane.config_loader import CFG
from pentane.units import K_to_kJmol, rad_to_degrees
from pentane.wham import wham as umbrella_wham

from remd.remd import block_pmf as remd_block_pmf
from remd.remd import wham_pmf as remd_wham_pmf


PRIMARY_T = float(CFG["simulation"]["temperatures_K"][-1])
N_BINS = int(CFG["simulation"]["n_bins"])
PRIMARY_T_TAG = f"{PRIMARY_T:.0f}K"
PRIMARY_BARRIER_TOL = 2.0
PRIMARY_DELTAF_TOL = 1.0
PRIMARY_RMSD_TOL = 0.5


_REMD_CSV_RE = re.compile(r"replica_(\d+)_T([0-9.]+)K\.csv$")


def _temperature_tag(T: float) -> str:
    return f"{T:.0f}K"


def _fill_nan_linear(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float).copy()
    finite = np.isfinite(values)
    if finite.all():
        return values
    if not finite.any():
        raise ValueError("cannot interpolate an array with no finite values")
    x = np.arange(values.size)
    values[~finite] = np.interp(x[~finite], x[finite], values[finite])
    return values


def _shift_to_zero(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).any():
        return values
    return values - np.nanmin(values)


def _interp_to_grid(source_deg: np.ndarray, source_values: np.ndarray, target_deg: np.ndarray) -> np.ndarray:
    source_deg = np.asarray(source_deg, dtype=float)
    source_values = _fill_nan_linear(source_values)
    kind = "cubic" if source_deg.size >= 4 else "linear"
    interpolator = interp1d(source_deg, source_values, kind=kind, fill_value="extrapolate")
    return np.asarray(interpolator(target_deg), dtype=float)


def _read_csv_replica(path: Path) -> tuple[float, np.ndarray, np.ndarray]:
    match = _REMD_CSV_RE.match(path.name)
    if match is None:
        raise ValueError(f"unexpected REMD filename: {path.name}")
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    data = np.atleast_2d(data)
    temperature = float(match.group(2))
    dihedral_deg = np.asarray(data[:, 0], dtype=float)
    energy_kjmol = np.asarray(data[:, 1], dtype=float)
    return temperature, dihedral_deg, energy_kjmol


def load_remd_data(remd_dir: Path) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
    records = []
    for path in sorted(remd_dir.glob("replica_*_T*.csv")):
        temperature, dihedral_deg, energy_kjmol = _read_csv_replica(path)
        replica_match = _REMD_CSV_RE.match(path.name)
        replica_index = int(replica_match.group(1)) if replica_match else len(records)
        records.append((temperature, replica_index, dihedral_deg, energy_kjmol, path))

    if not records:
        raise FileNotFoundError(f"no REMD CSVs found in {remd_dir}")

    records.sort(key=lambda item: (item[0], item[1]))
    temps = np.array([item[0] for item in records], dtype=float)
    dihedrals = [item[2] for item in records]
    energies = [item[3] for item in records]
    return temps, dihedrals, energies


def _require_matching_remd_temperature(temps: np.ndarray, T_ref: float) -> None:
    if not np.isclose(np.min(temps), T_ref):
        ladder = ", ".join(f"{t:.1f}" for t in temps)
        raise ValueError(
            f"REMD lowest replica temperature is {np.min(temps):.1f} K, but the requested "
            f"cross-validation temperature is {T_ref:.1f} K. Rerun remd/remd.py with "
            f"--T-min {T_ref:.0f} --T-max 600 --replicas ... so the ladder includes the target temperature. "
            f"Current ladder: [{ladder}]"
        )


def _umbrella_window_centres(cfg: dict) -> np.ndarray:
    n_windows = int(cfg["umbrella"]["n_windows"])
    step = 2.0 * np.pi / n_windows
    return -np.pi + 0.5 * step + step * np.arange(n_windows)


def _load_umbrella_trajectories(results_dir: Path, cfg: dict, T_ref: float) -> list[np.ndarray]:
    tag = _temperature_tag(T_ref)
    n_windows = int(cfg["umbrella"]["n_windows"])
    primary_tag = _temperature_tag(float(CFG["simulation"]["temperatures_K"][0]))
    allow_legacy = np.isclose(T_ref, float(CFG["simulation"]["temperatures_K"][0]))

    trajs = []
    for i in range(n_windows):
        candidates = [results_dir / "trajectories" / f"us_window_{tag}_{i:02d}.npy"]
        if allow_legacy:
            candidates.append(results_dir / "trajectories" / f"us_window_{i:02d}.npy")
        chosen = next((path for path in candidates if path.exists()), None)
        if chosen is None:
            raise FileNotFoundError(
                f"missing umbrella trajectory for window {i:02d} at {T_ref:.0f} K. "
                f"Looked for: {', '.join(str(path) for path in candidates)}"
            )
        trajs.append(np.load(chosen))

    return trajs


def _load_umbrella_pmf(results_dir: Path, T_ref: float) -> tuple[np.ndarray, np.ndarray]:
    tag = _temperature_tag(T_ref)
    candidates = [results_dir / f"wham_bin_centres_{tag}.npy"]
    if np.isclose(T_ref, PRIMARY_T):
        candidates.append(results_dir / "wham_bin_centres.npy")
    bin_centres_path = next((path for path in candidates if path.exists()), None)
    if bin_centres_path is None:
        raise FileNotFoundError(
            f"missing umbrella WHAM bin-centres file for {T_ref:.0f} K. Looked for: "
            f"{', '.join(str(path) for path in candidates)}"
        )

    pmf_candidates = [results_dir / f"wham_pmf_{tag}.npy"]
    if np.isclose(T_ref, PRIMARY_T):
        pmf_candidates.append(results_dir / "wham_pmf.npy")
    pmf_path = next((path for path in pmf_candidates if path.exists()), None)
    if pmf_path is None:
        raise FileNotFoundError(
            f"missing umbrella WHAM PMF file for {T_ref:.0f} K. Looked for: "
            f"{', '.join(str(path) for path in pmf_candidates)}"
        )

    return np.load(bin_centres_path), np.load(pmf_path)


def _umbrella_block_statistics(trajs: list[np.ndarray], phi0s: np.ndarray, cfg: dict, T_ref: float, n_blocks: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_samples = min(len(traj) for traj in trajs)
    block_size = n_samples // n_blocks
    if block_size < 20:
        centres, pmf = umbrella_wham(trajs, phi0s, cfg, T_ref)
        return centres, pmf, np.zeros_like(pmf), np.array([pmf])

    blocks = []
    centres = None
    for block_index in range(n_blocks):
        start = block_index * block_size
        stop = (block_index + 1) * block_size
        block_trajs = [np.asarray(traj[start:stop], dtype=float) for traj in trajs]
        centres, block_pmf = umbrella_wham(block_trajs, phi0s, cfg, T_ref)
        blocks.append(_fill_nan_linear(block_pmf))

    blocks = np.asarray(blocks, dtype=float)
    pmf_mean = np.nanmean(blocks, axis=0)
    pmf_std = np.nanstd(blocks, axis=0, ddof=1) if blocks.shape[0] > 1 else np.zeros_like(pmf_mean)
    return centres, pmf_mean, pmf_std, blocks


def _load_baseline_pmfs(results_dir: Path, T_ref: float, cfg: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    tag = _temperature_tag(T_ref)
    n_bins = int(cfg["simulation"]["n_bins"])
    trajectories_dir = results_dir / "trajectories"
    mc_path = trajectories_dir / f"mc_{int(round(T_ref))}.npy"
    md_path = trajectories_dir / f"md_{int(round(T_ref))}.npy"
    if not mc_path.exists() or not md_path.exists():
        raise FileNotFoundError(
            f"missing baseline trajectories for {T_ref:.0f} K. Expected {mc_path.name} and {md_path.name}"
        )

    mc_traj = np.load(mc_path)
    md_traj = np.load(md_path)
    mc_centres_rad, mc_pmf_k = boltzmann_pmf(mc_traj, T_ref, n_bins)
    md_centres_rad, md_pmf_k = boltzmann_pmf(md_traj, T_ref, n_bins)

    return (
        rad_to_degrees(mc_centres_rad),
        K_to_kJmol(mc_pmf_k),
        K_to_kJmol(md_pmf_k),
        mc_traj,
        md_traj,
    )


def _pmf_region_metrics(phi_deg: np.ndarray, pmf_kjmol: np.ndarray) -> tuple[float, float, np.ndarray]:
    pmf = _shift_to_zero(np.asarray(pmf_kjmol, dtype=float))
    finite = np.isfinite(pmf)
    if not finite.any():
        return np.nan, np.nan, pmf

    barrier = float(np.nanmax(pmf))
    trans_mask = (np.abs(phi_deg) > 150.0) & finite
    gauche_mask = (np.abs(phi_deg) > 40.0) & (np.abs(phi_deg) < 80.0) & finite
    if not trans_mask.any() or not gauche_mask.any():
        delta_f = np.nan
    else:
        delta_f = float(np.nanmin(pmf[gauche_mask]) - np.nanmin(pmf[trans_mask]))
    return barrier, delta_f, pmf


def _rmsd(reference: np.ndarray, comparison: np.ndarray) -> float:
    diff = np.asarray(reference, dtype=float) - np.asarray(comparison, dtype=float)
    finite = np.isfinite(diff)
    if not finite.any():
        return np.nan
    return float(np.sqrt(np.nanmean(diff[finite] ** 2)))


def _format_value(value: float, digits: int = 3) -> str:
    if not np.isfinite(value):
        return "nan"
    return f"{value:.{digits}f}"


def _format_markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [len(str(header)) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(str(cell)))

    def _row(cells: list[str]) -> str:
        return "| " + " | ".join(f"{cell:<{widths[i]}}" for i, cell in enumerate(cells)) + " |"

    separator = ["-" * width for width in widths]
    lines = [_row(headers), _row(separator)]
    lines.extend(_row(row) for row in rows)
    return "\n".join(lines)


def _plot_overlay(
    phi_deg: np.ndarray,
    remd_pmf: np.ndarray,
    remd_std: np.ndarray,
    umbrella_pmf: np.ndarray,
    umbrella_std: np.ndarray,
    baseline_pmf: np.ndarray,
    baseline_std: np.ndarray,
    out_path: Path,
    title: str,
    metrics_text: str,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_title(title, fontsize=13, color="#e6edf3")
    ax.set_xlabel("Backbone dihedral phi [deg]")
    ax.set_ylabel("F(phi) [kJ/mol]")

    series = [
        ("REMD", remd_pmf, remd_std, "#58a6ff"),
        ("Umbrella+WHAM", umbrella_pmf, umbrella_std, "#f78166"),
        ("Baseline MC/MD mean", baseline_pmf, baseline_std, "#56d364"),
    ]

    for label, values, spread, colour in series:
        lower = values - spread
        upper = values + spread
        ax.plot(phi_deg, values, label=label, color=colour, lw=2.0)
        ax.fill_between(phi_deg, lower, upper, color=colour, alpha=0.18)

    for xv, colour in [(180.0, "grey"), (-180.0, "grey"), (60.0, "steelblue"), (-60.0, "steelblue"), (0.0, "red")]:
        ax.axvline(xv, ls=":", c=colour, lw=1.0, alpha=0.45)

    ax.set_xlim(-180.0, 180.0)
    ax.set_ylim(bottom=0.0)
    ax.legend(fontsize=9)
    ax.text(
        0.98,
        0.98,
        metrics_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#0d1117", alpha=0.85, edgecolor="#30363d"),
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def run_crossvalidation(T_ref: float, remd_dir: Path, results_dir: Path, n_blocks: int = 5) -> tuple[str, Path, Path]:
    cfg = CFG
    temps, remd_dihed_deg, remd_energy_kjmol = load_remd_data(remd_dir)
    _require_matching_remd_temperature(temps, T_ref)

    umbrella_tag = _temperature_tag(T_ref)
    umbrella_bin_centres_rad, umbrella_pmf_k = _load_umbrella_pmf(results_dir, T_ref)
    umbrella_trajs = _load_umbrella_trajectories(results_dir, cfg, T_ref)
    umbrella_phi0s = _umbrella_window_centres(cfg)

    remd_centres_deg, remd_pmf_kjmol, _ = remd_wham_pmf(remd_dihed_deg, remd_energy_kjmol, temps, T_ref, n_bins=N_BINS)
    remd_centres_deg = np.asarray(remd_centres_deg, dtype=float)
    remd_pmf_kjmol = _shift_to_zero(remd_pmf_kjmol)
    remd_pmf_interp = _shift_to_zero(_interp_to_grid(remd_centres_deg, remd_pmf_kjmol, rad_to_degrees(umbrella_bin_centres_rad)))
    remd_centres_deg = rad_to_degrees(umbrella_bin_centres_rad)

    _, remd_block_mean_kj, remd_block_std_kj, _ = remd_block_pmf(remd_dihed_deg, remd_energy_kjmol, temps, T_ref, n_blocks=n_blocks, n_bins=N_BINS)
    remd_block_mean_kj = _shift_to_zero(_interp_to_grid(remd_centres_deg, remd_block_mean_kj, remd_centres_deg))
    remd_block_std_kj = _interp_to_grid(remd_centres_deg, remd_block_std_kj, remd_centres_deg)

    umbrella_block_centres_rad, umbrella_block_mean_k, umbrella_block_std_k, umbrella_blocks = _umbrella_block_statistics(
        umbrella_trajs, umbrella_phi0s, cfg, T_ref, n_blocks
    )
    umbrella_pmf_k = K_to_kJmol(umbrella_pmf_k)
    umbrella_block_mean_kj = K_to_kJmol(umbrella_block_mean_k)
    umbrella_block_std_kj = K_to_kJmol(umbrella_block_std_k)

    target_deg = rad_to_degrees(umbrella_bin_centres_rad)
    umbrella_pmf_kj = _shift_to_zero(_interp_to_grid(rad_to_degrees(umbrella_bin_centres_rad), umbrella_pmf_k, target_deg))
    umbrella_block_mean_kj = _shift_to_zero(_interp_to_grid(rad_to_degrees(umbrella_block_centres_rad), umbrella_block_mean_kj, target_deg))
    umbrella_block_std_kj = _interp_to_grid(rad_to_degrees(umbrella_block_centres_rad), umbrella_block_std_kj, target_deg)

    baseline_centres_deg, baseline_mc_kj, baseline_md_kj, mc_traj, md_traj = _load_baseline_pmfs(results_dir, T_ref, cfg)
    baseline_mc_interp = _shift_to_zero(_interp_to_grid(baseline_centres_deg, baseline_mc_kj, target_deg))
    baseline_md_interp = _shift_to_zero(_interp_to_grid(baseline_centres_deg, baseline_md_kj, target_deg))
    baseline_stack = np.vstack([baseline_mc_interp, baseline_md_interp])
    baseline_mean = np.nanmean(baseline_stack, axis=0)
    baseline_std = np.nanstd(baseline_stack, axis=0, ddof=0)
    baseline_mean = _shift_to_zero(baseline_mean)

    remd_barrier, remd_deltaf, remd_pmf_interp = _pmf_region_metrics(target_deg, remd_pmf_interp)
    umbrella_barrier, umbrella_deltaf, umbrella_pmf_kj = _pmf_region_metrics(target_deg, umbrella_pmf_kj)
    baseline_barrier, baseline_deltaf, baseline_mean = _pmf_region_metrics(target_deg, baseline_mean)

    remd_rmsd = _rmsd(remd_pmf_interp, umbrella_pmf_kj)
    baseline_rmsd = _rmsd(remd_pmf_interp, baseline_mean)
    umbrella_vs_baseline_rmsd = _rmsd(umbrella_pmf_kj, baseline_mean)

    barrier_diff = abs(umbrella_barrier - remd_barrier)
    deltaf_diff = abs(umbrella_deltaf - remd_deltaf)
    overall_pass = (
        np.isfinite(remd_rmsd)
        and remd_rmsd < PRIMARY_RMSD_TOL
        and barrier_diff < PRIMARY_BARRIER_TOL
        and deltaf_diff < PRIMARY_DELTAF_TOL
    )

    method_headers = ["Quantity", "Umbrella+WHAM", "REMD", "Baseline (MC/MD)", "Difference", "Pass/Fail"]
    method_rows = [
        [
            "Barrier height [kJ/mol]",
            _format_value(umbrella_barrier),
            _format_value(remd_barrier),
            _format_value(baseline_barrier),
            _format_value(barrier_diff),
            "PASS" if barrier_diff < PRIMARY_BARRIER_TOL else "FAIL",
        ],
        [
            "Gauche-trans DeltaF [kJ/mol]",
            _format_value(umbrella_deltaf),
            _format_value(remd_deltaf),
            _format_value(baseline_deltaf),
            _format_value(deltaf_diff),
            "PASS" if deltaf_diff < PRIMARY_DELTAF_TOL else "FAIL",
        ],
        [
            "PMF RMSD (umbrella vs REMD) [kJ/mol]",
            _format_value(remd_rmsd),
            "n/a",
            "n/a",
            _format_value(remd_rmsd),
            "PASS" if np.isfinite(remd_rmsd) and remd_rmsd < PRIMARY_RMSD_TOL else "FAIL",
        ],
    ]

    pairwise_headers = ["Pairwise comparison", "RMSD [kJ/mol]", "Pass/Fail"]
    pairwise_rows = [
        ["Umbrella vs REMD", _format_value(remd_rmsd), "PASS" if np.isfinite(remd_rmsd) and remd_rmsd < PRIMARY_RMSD_TOL else "FAIL"],
        ["Umbrella vs Baseline", _format_value(umbrella_vs_baseline_rmsd), "PASS" if np.isfinite(umbrella_vs_baseline_rmsd) and umbrella_vs_baseline_rmsd < PRIMARY_RMSD_TOL else "FAIL"],
        ["REMD vs Baseline", _format_value(baseline_rmsd), "PASS" if np.isfinite(baseline_rmsd) and baseline_rmsd < PRIMARY_RMSD_TOL else "FAIL"],
    ]

    metrics_text = "\n".join(
        [
            f"T_ref = {T_ref:.0f} K",
            f"RMSD(U,R) = {remd_rmsd:.3f} kJ/mol",
            f"|DeltaBarrier| = {barrier_diff:.3f} kJ/mol",
            f"|DeltaF| = {deltaf_diff:.3f} kJ/mol",
        ]
    )

    plots_dir = results_dir / "plots"
    report_dir = results_dir / "report"
    plots_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    plot_path = plots_dir / f"crossvalidation_{_temperature_tag(T_ref)}.png"
    report_path = report_dir / f"crossvalidation_{_temperature_tag(T_ref)}.txt"
    _plot_overlay(
        target_deg,
        remd_pmf_interp,
        remd_block_std_kj,
        umbrella_pmf_kj,
        umbrella_block_std_kj,
        baseline_mean,
        baseline_std,
        plot_path,
        f"Cross-validation: REMD vs Umbrella+WHAM vs Baseline ({T_ref:.0f} K)",
        metrics_text,
    )

    report_lines = [
        f"Cross-validation summary at {T_ref:.0f} K",
        "",
        f"REMD lowest replica temperature: {np.min(temps):.1f} K",
        f"Umbrella PMF source: {umbrella_tag}",
        f"Reference PMF grid: {len(target_deg)} bins",
        "",
        _format_markdown_table(method_headers, method_rows),
        "",
        _format_markdown_table(pairwise_headers, pairwise_rows),
        "",
        f"Primary pass criterion: RMSD < {PRIMARY_RMSD_TOL:.3f} kJ/mol, |DeltaBarrier| < {PRIMARY_BARRIER_TOL:.3f} kJ/mol, |DeltaF| < {PRIMARY_DELTAF_TOL:.3f} kJ/mol",
        f"Overall status: {'PASS' if overall_pass else 'FAIL'}",
    ]
    report_text = "\n".join(report_lines)
    report_path.write_text(report_text + "\n", encoding="utf-8")

    console_lines = [
        report_text,
        "",
        f"Overlay figure: {plot_path}",
        f"Report file: {report_path}",
    ]
    console_text = "\n".join(console_lines)
    print(console_text)
    return console_text, plot_path, report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cross-validate REMD against umbrella+WHAM")
    parser.add_argument(
        "--temperature",
        type=float,
        default=PRIMARY_T,
        help="Reference temperature in K (must match the lowest REMD replica and the umbrella/baseline outputs)",
    )
    parser.add_argument("--remd-dir", type=Path, default=ROOT / "remd" / "remd_output")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--n-blocks", type=int, default=5, help="Number of blocks used for uncertainty estimates")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_crossvalidation(args.temperature, args.remd_dir, args.results_dir, n_blocks=args.n_blocks)


if __name__ == "__main__":
    main()