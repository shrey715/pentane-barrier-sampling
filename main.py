"""
main.py — End-to-end pipeline for n-pentane barrier sampling.

Stages
------
  1. Baseline NVT MC + MD at 120 K and 250 K
  2. Umbrella sampling + WHAM at both temperatures
  3. REMD validation (requires OpenMM)

Each stage caches its outputs to results/ and skips recomputation
if those files already exist.

Usage
-----
  python main.py                         # run all three stages
  python main.py --skip-remd             # skip REMD (fast, no OpenMM needed)
  python main.py --skip-baseline --skip-umbrella   # REMD only
  python main.py --remd-steps 50000 --remd-replicas 8

"""
import argparse
import sys
from pathlib import Path

# Make src/, scripts/, and remd/ all importable from the project root
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "remd"))


def run_baselines():
    print("\n" + "=" * 65)
    print("  STAGE 1 — Baseline NVT MC + MD")
    print("=" * 65)
    import run_baseline
    run_baseline.main()


def run_umbrella():
    print("\n" + "=" * 65)
    print("  STAGE 2 — Umbrella Sampling + WHAM")
    print("=" * 65)
    import run_umbrella
    run_umbrella.main()


def run_remd(n_replicas: int, total_steps: int):
    print("\n" + "=" * 65)
    print("  STAGE 3 — REMD Validation (OpenMM)")
    print("=" * 65)
    from remd import run_remd as _run_remd
    _run_remd(
        n_replicas  = n_replicas,
        T_min       = 120.0,
        T_max       = 600.0,
        total_steps = total_steps,
        swap_freq   = 500,
        sample_every= 50,
        dt_ps       = 0.002,
        out_dir     = str(ROOT / "results" / "remd"),
        seed        = 42,
    )


def parse_args():
    p = argparse.ArgumentParser(
        description="n-Pentane barrier sampling pipeline: baselines → US/WHAM → REMD"
    )
    p.add_argument("--skip-baseline", action="store_true",
                   help="Skip Stage 1 (baseline MC + MD)")
    p.add_argument("--skip-umbrella", action="store_true",
                   help="Skip Stage 2 (umbrella sampling + WHAM)")
    p.add_argument("--skip-remd", action="store_true",
                   help="Skip Stage 3 (REMD) — useful if OpenMM is not installed")
    p.add_argument("--remd-replicas", type=int, default=12,
                   help="Number of REMD replicas (default: 12)")
    p.add_argument("--remd-steps", type=int, default=100_000,
                   help="Total MD steps per REMD run (default: 100 000)")
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 65)
    print("  n-Pentane Barrier Sampling Pipeline")
    print("=" * 65)

    if not args.skip_baseline:
        run_baselines()
    else:
        print("\nStage 1 skipped (--skip-baseline)")

    if not args.skip_umbrella:
        run_umbrella()
    else:
        print("\nStage 2 skipped (--skip-umbrella)")

    if not args.skip_remd:
        run_remd(args.remd_replicas, args.remd_steps)
    else:
        print("\nStage 3 skipped (--skip-remd)")

    print("\n" + "=" * 65)
    print("  Pipeline complete. All outputs in results/")
    print("=" * 65)


if __name__ == "__main__":
    main()
