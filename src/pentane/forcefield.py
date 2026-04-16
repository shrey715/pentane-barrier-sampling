"""
TraPPE-UA Force Field — Torsion Potential for n-Pentane
=======================================================

Implements the Ryckaert–Bellemans / OPLS torsion potential used in the
TraPPE-UA (Transferable Potentials for Phase Equilibria — United Atom)
force field for alkanes.

The torsion potential for the C1–C2–C3–C4 dihedral is:

    U(φ) = c₁(1 + cos φ) + c₂(1 − cos 2φ) + c₃(1 + cos 3φ)

Parameters are given in Kelvin (i.e., U / k_B), so k_B = 1 in these
natural units and β = 1/T.

References
----------
- M. G. Martin & J. I. Siepmann, J. Phys. Chem. B, 102, 2569 (1998).
  "Transferable Potentials for Phase Equilibria. 1. United-Atom
  Description of n-Alkanes."
"""

import numpy as np
from numpy.typing import ArrayLike

# ---------------------------------------------------------------------------
# TraPPE-UA torsion coefficients [K]
# ---------------------------------------------------------------------------
C1_K: float = 355.03   # coefficient for (1 + cos φ) term
C2_K: float = -68.19   # coefficient for (1 − cos 2φ) term
C3_K: float = 791.32   # coefficient for (1 + cos 3φ) term

# ---------------------------------------------------------------------------
# Bond geometry parameters
# ---------------------------------------------------------------------------
BOND_LENGTH: float = 1.54       # C–C bond length [Å]
BOND_ANGLE_DEG: float = 114.0   # C–C–C bond angle [degrees]
BOND_ANGLE_RAD: float = np.radians(BOND_ANGLE_DEG)

# ---------------------------------------------------------------------------
# Effective moment of inertia for dihedral rotation
# ---------------------------------------------------------------------------
MASS_CH3: float = 15.035   # united-atom CH₃ mass [amu]
MASS_CH2: float = 14.027   # united-atom CH₂ mass [amu]

# Perpendicular distance from the rotation axis (C2–C3 bond) to C4/C5
R_PERP: float = BOND_LENGTH * np.sin(BOND_ANGLE_RAD)  # ≈ 1.408 Å

# Effective moment of inertia for the rotating group [amu·Å²]
I_EFF: float = MASS_CH3 * R_PERP**2  # ≈ 29.75 amu·Å²

# ---------------------------------------------------------------------------
# Unit conversion constants
# ---------------------------------------------------------------------------
KB_KJ_MOL: float = 0.008314   # Boltzmann constant [kJ/(mol·K)]
UNIT_CONV: float = 0.01        # kJ/mol per amu·Å²/ps²


def torsion_energy(phi: ArrayLike) -> np.ndarray:
    """
    Compute the TraPPE-UA torsion potential energy.

    Parameters
    ----------
    phi : float or array_like
        Dihedral angle(s) in radians.

    Returns
    -------
    U : float or ndarray
        Torsion potential energy in Kelvin (U / k_B).

    Notes
    -----
    - Trans state (φ = ±π): U = 0 K (global minimum)
    - Gauche states (φ ≈ ±π/3): local minima
    - Eclipsed barriers (φ ≈ 0, ±2π/3): up to ~2292 K
    """
    phi = np.asarray(phi, dtype=np.float64)
    return (
        C1_K * (1.0 + np.cos(phi))
        + C2_K * (1.0 - np.cos(2.0 * phi))
        + C3_K * (1.0 + np.cos(3.0 * phi))
    )


def torsion_force(phi: ArrayLike) -> np.ndarray:
    """
    Compute the torsion torque τ = −dU/dφ.

    Parameters
    ----------
    phi : float or array_like
        Dihedral angle(s) in radians.

    Returns
    -------
    tau : float or ndarray
        Torque in Kelvin / radian (i.e., −dU/dφ with U in Kelvin).
    """
    phi = np.asarray(phi, dtype=np.float64)
    return (
        C1_K * np.sin(phi)
        - 2.0 * C2_K * np.sin(2.0 * phi)
        + 3.0 * C3_K * np.sin(3.0 * phi)
    )
