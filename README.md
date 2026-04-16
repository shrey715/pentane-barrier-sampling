# Escaping the Pentane Barrier — Molecular Modeling Mini-Project

## Overview

This project models **n-pentane** in the **United-Atom (UA)** representation using the **TraPPE-UA** force field. It demonstrates the conformational sampling problem ("pentane barrier") and shows how the **Wang-Landau** enhanced sampling algorithm overcomes it.

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
│   ├── wang_landau.py      # Wang-Landau flat-histogram sampling
│   ├── analysis.py         # Entropy, PMF, exploration metrics
│   └── plotting.py         # Publication-quality figure generation
├── scripts/
│   └── run_all.py          # Main driver: runs everything end-to-end
└── results/                # Generated outputs
    ├── plots/              # All figures (PNG)
    └── report/             # Summary report (TXT)
```

## Quick Start

```bash
# Install dependencies with uv
uv sync

# Run the complete pipeline
uv run python scripts/run_all.py
```

## Methods Compared

| Method | Type | Description |
|--------|------|-------------|
| **Metropolis MC** | Baseline | Random dihedral perturbation with Boltzmann acceptance |
| **NVT MD** | Baseline | Velocity Verlet + Nosé-Hoover thermostat |
| **Wang-Landau** | Enhanced | Flat-histogram MC with adaptive bias potential |

## Key Results

- At **120 K**, baseline methods are trapped near the trans minimum. Wang-Landau explores all 36 bins.
- At **250 K**, MC partially escapes barriers. Wang-Landau still achieves full coverage.
- The Wang-Landau PMF closely reproduces the exact torsion potential.

## Dependencies

- Python ≥ 3.10
- NumPy ≥ 1.24
- Matplotlib ≥ 3.7
- SciPy ≥ 1.10

## References

- Martin & Siepmann, J. Phys. Chem. B, 102, 2569 (1998) — TraPPE-UA force field
- Wang & Landau, Phys. Rev. Lett. 86, 2050 (2001) — Wang-Landau algorithm
