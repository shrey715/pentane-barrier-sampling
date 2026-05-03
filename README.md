# Escaping the Pentane Barrier — Molecular Modeling Mini-Project

## Overview

This project models **n-pentane** in the **United-Atom (UA)** representation using the **TraPPE-UA** force field in full Cartesian 3D space. It demonstrates the conformational sampling problem ("pentane barrier") and compares baseline Monte Carlo / molecular dynamics against **umbrella sampling + WHAM**.

## Project Structure

```txt
project/
├── pyproject.toml          # uv project configuration
├── README.md               # This file
├── project.pdf             # Original assignment
├── src/pentane/            # Core simulation package
│   ├── forcefield.py       # TraPPE-UA torsion potential & parameters
│   ├── geometry.py         # Molecular geometry (NeRF algorithm)
│   ├── mc.py               # Metropolis Monte Carlo
│   ├── md.py               # NVT MD (Nosé-Hoover thermostat)
│   ├── umbrella.py         # Umbrella sampling windows
│   ├── wham.py             # WHAM post-processing
│   ├── units.py            # K <-> kJ/mol and angle conversions
│   ├── analysis.py         # Entropy, PMF, exploration metrics
│   └── plotting.py         # Publication-quality figure generation
├── scripts/
│   ├── run_baseline.py     # Baseline MC/MD pipeline
│   ├── run_umbrella.py     # Umbrella sampling + WHAM pipeline
│   └── run_crossvalidation.py  # REMD vs umbrella vs baseline comparison
├── tests/
│   └── test_crossvalidation.py  # Dihedral and unit sanity checks
└── results/                # Generated outputs
    ├── plots/              # All figures (PNG)
    └── report/             # Summary report (TXT)
```

## Quick Start

```bash
# Install dependencies with uv
uv sync

# Run the baseline pipeline
uv run python scripts/run_baseline.py

# Run umbrella sampling + WHAM
uv run python scripts/run_umbrella.py

# Cross-validate against REMD (requires matching lowest replica temperature)
uv run python scripts/run_crossvalidation.py --temperature 250
```

## Methods Compared

| Method | Type | Description |
|--------|------|-------------|
| **Metropolis MC** | Baseline | Full Cartesian fragment rotations with Boltzmann acceptance |
| **NVT MD** | Baseline | Full Cartesian velocity Verlet + Nosé-Hoover thermostat |
| **Umbrella sampling + WHAM** | Enhanced | Biased window sampling with histogram reweighting |

## Key Results

- At **120 K**, baseline methods remain trapped near the trans minimum.
- Umbrella sampling with WHAM reconstructs the unbiased PMF across all 36 bins.
- The full Cartesian model includes bond stretch, angle bend, torsion, and the allowed C1···C5 Lennard-Jones interaction.

## Cross-Validation

REMD is the independent OpenMM-backed reference. The cross-validation script compares the REMD PMF against the umbrella+WHAM PMF and the baseline MC/MD PMFs after converting everything to a common unit system.

The REMD ladder must include the comparison temperature as its lowest replica. For example, to compare at 250 K run:

```bash
uv run python remd/remd.py --T-min 250 --T-max 600 --replicas 8
uv run python scripts/run_baseline.py
uv run python scripts/run_umbrella.py
uv run python scripts/run_crossvalidation.py --temperature 250
```

For a 120 K comparison, rerun REMD with a ladder that starts at 120 K, for example `--T-min 120 --T-max 600 --replicas 12`.

The cross-validation report and overlay figure are written to `results/report/crossvalidation_<T>K.txt` and `results/plots/crossvalidation_<T>K.png`.

## Dependencies

- Python ≥ 3.10
- NumPy ≥ 1.24
- Matplotlib ≥ 3.7
- SciPy ≥ 1.10

## References

- Martin & Siepmann, J. Phys. Chem. B, 102, 2569 (1998) — TraPPE-UA force field
- Mundy et al., Faraday Discuss. 104, 123 (1996) — bond stretch and angle parameters
- Kumar et al., J. Comput. Chem. 13, 1011 (1992) — WHAM
