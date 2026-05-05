# Escaping the Pentane Barrier — Molecular Modeling Mini-Project

## Overview

This project models **n-pentane** in the **United-Atom (UA)** representation using the **TraPPE-UA** force field. It implements full 3D Cartesian simulations (MC and MD) to demonstrate the conformational trapping problem at low temperatures and reconstructs the unbiased free-energy landscape (PMF) using **Umbrella Sampling + WHAM**.

## Project Structure

```txt
project/
├── main.py                 # Unified entry point for the simulation pipeline
├── pyproject.toml          # uv project configuration
├── config/
│   └── trappe_ua.toml      # Force field and simulation parameters
├── src/pentane/            # Core analytical package
│   ├── analysis.py         # Entropy, early exploration scores, and PMF math
│   ├── plotting.py         # Publication-quality visualization suite
│   ├── forcefield.py       # TraPPE-UA potential implementation
│   ├── mc.py               # Metropolis MC (Cartesian displacements)
│   ├── md.py               # Velocity Verlet MD (Nosé-Hoover)
│   └── wham.py             # WHAM implementation
├── scripts/
│   ├── run_baseline.py     # Stage 1: Trapping diagnostics (MC/MD)
│   └── run_umbrella.py     # Stage 2: Enhanced sampling (Umbrella+WHAM)
├── report/
│   └── main.tex            # Final LaTeX research report
└── results/                # Generated figures and trajectory cache
    ├── trajectories/       # Cached .npy simulation data
    └── *.png               # Analysis plots (entropy, timeseries, PMF)
```

## Getting Started

The project uses the `uv` package manager for fast, reproducible environment management.

```bash
# Install dependencies
uv sync

# Run the complete research pipeline (Baseline + Umbrella)
uv run python main.py --skip-remd
```

## Key Research Features

- **Diagnostic Visualizations**: Automatically generates dihedral time series, probability distributions, and PMF overlays comparing 120 K vs 250 K.
- **Exploration Metrics**: Quantifies sampling efficiency using:
  - **Shannon Entropy $S(t)$**: Tracking conformational discovery over time.
  - **Early Exploration Score $E$**: Time-average of $S(t)$ to detect trapping.
- **Enhanced Sampling**: Implementation of 36-window umbrella sampling to resolve barriers (~1500–2200 K) that are inaccessible to baseline MD/MC.
- **Physical Rigor**: Models all internal degrees of freedom (bond stretch, angle bend, torsion) and long-range C1···C5 Lennard-Jones interactions.

## Core Results

- At **120 K**, baseline simulations are trapped in the *trans* state (crossings $\approx 0$).
- Umbrella Sampling recovers the full dihedral landscape, revealing the *gauche* minima and the rotation barriers.
- The **Early Exploration Score** quantitatively distinguishes the "staircase" discovery of Umbrella Sampling from the immediate plateau of trapped baseline simulations.

## References

- **Force Field**: Martin & Siepmann, *J. Phys. Chem. B* 102, 2569 (1998).
- **WHAM**: Kumar et al., *J. Comput. Chem.* 13, 1011 (1992).
- **Algorithm**: NeRF implementation for fast Cartesian mapping.
