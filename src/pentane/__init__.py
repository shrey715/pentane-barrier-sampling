"""
Pentane Molecular Modeling Package
==================================

United-Atom (UA) simulation of n-pentane using the TraPPE-UA force field.
Implements Metropolis Monte Carlo, NVT Molecular Dynamics (Nosé-Hoover),
and Wang-Landau enhanced sampling to study conformational sampling and
the "pentane barrier" problem.

Modules
-------
forcefield : TraPPE-UA torsion potential energy and forces
geometry   : Molecular geometry construction and dihedral calculation
mc         : Metropolis Monte Carlo simulation
md         : NVT Molecular Dynamics with Nosé-Hoover thermostat
wang_landau: Wang-Landau flat-histogram enhanced sampling
analysis   : Entropy, PMF, and exploration metrics
plotting   : Publication-quality figure generation
"""

__version__ = "1.0.0"
