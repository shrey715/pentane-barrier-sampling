# Escaping the Pentane Barrier — Molecular Modeling Mini-Project

## Overview

This project models **n-pentane** in the **United-Atom (UA)** representation using the **TraPPE-UA** force field in full Cartesian 3D space. It demonstrates the conformational sampling problem ("pentane barrier") and compares baseline Monte Carlo / molecular dynamics against **umbrella sampling + WHAM**.

## Project Structure

```
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
│   ├── analysis.py         # Entropy, PMF, exploration metrics
│   └── plotting.py         # Publication-quality figure generation
├── scripts/
│   ├── run_baseline.py     # Baseline MC/MD pipeline
│   └── run_umbrella.py     # Umbrella sampling + WHAM pipeline
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

## Dependencies

- Python ≥ 3.10
- NumPy ≥ 1.24
- Matplotlib ≥ 3.7
- SciPy ≥ 1.10

## References

- Martin & Siepmann, J. Phys. Chem. B, 102, 2569 (1998) — TraPPE-UA force field
- Mundy et al., Faraday Discuss. 104, 123 (1996) — bond stretch and angle parameters
- Kumar et al., J. Comput. Chem. 13, 1011 (1992) — WHAM
# MoMoS_Project
