"""
forcefield.py — Cartesian TraPPE-UA energy model for n-pentane.

The full potential includes bond stretch, angle bend, torsion, and the only
allowed intramolecular Lennard-Jones interaction for n-pentane (C1···C5).

All energies are expressed in Kelvin with k_B = 1.
"""
import numpy as np

from pentane.config_loader import CFG
from pentane.geometry import calc_angle, calc_dihedral


_BOND = CFG.get("bond_stretch", {})
_ANG = CFG["angles"]
_TOR = CFG["torsion"]
_NB = CFG["nonbonded"]

K_R = float(_BOND.get("k_r_K_per_ang2", 452900.0))
R0 = float(_BOND.get("r0_ang", CFG["bonds"]["r_CC_ang"]))
K_TH = float(_ANG["k_theta_K"])
TH0 = float(np.radians(_ANG["theta0_deg"]))

C0 = float(_TOR.get("c0_K", 0.0))
C1 = float(_TOR["c1_K"])
C2 = float(_TOR["c2_K"])
C3 = float(_TOR["c3_K"])

EPS_CH3 = float(_NB["eps_CH3_K"])
SIG_CH3 = float(_NB["sigma_CH3_ang"])
EPS_CH2 = float(_NB["eps_CH2_K"])
SIG_CH2 = float(_NB["sigma_CH2_ang"])

EPS_15 = EPS_CH3
SIG_15 = SIG_CH3


def calc_bond(a: np.ndarray, b: np.ndarray) -> float:
    """Bond length between two Cartesian coordinates [Å]."""
    return float(np.linalg.norm(np.asarray(b, dtype=float) - np.asarray(a, dtype=float)))


def bond_energy(r: float) -> float:
    """Harmonic bond stretch energy [K]."""
    return 0.5 * K_R * (r - R0) ** 2


def angle_energy(theta: float) -> float:
    """Harmonic angle bend energy [K]."""
    return 0.5 * K_TH * (theta - TH0) ** 2


def torsion_energy(phi: float) -> float:
    """Three-term TraPPE/OPLS-style torsion potential [K]."""
    return (
        C0
        + C1 * (1.0 + np.cos(phi))
        + C2 * (1.0 - np.cos(2.0 * phi))
        + C3 * (1.0 + np.cos(3.0 * phi))
    )


def torsion_force(phi: float) -> float:
    """Generalized torsional force −dU/dφ [K/rad]."""
    return C1 * np.sin(phi) - 2.0 * C2 * np.sin(2.0 * phi) + 3.0 * C3 * np.sin(3.0 * phi)


def lj_energy(r: float, eps: float, sigma: float) -> float:
    """Lennard-Jones 12-6 energy [K]."""
    sr6 = (sigma / r) ** 6
    return 4.0 * eps * (sr6 * sr6 - sr6)


def total_energy(coords: np.ndarray) -> float:
    """Full Cartesian TraPPE-UA energy for n-pentane [K]."""
    c = np.asarray(coords, dtype=float)
    if c.shape != (5, 3):
        raise ValueError(f"expected coords with shape (5, 3), got {c.shape}")

    U = 0.0

    for i, j in ((0, 1), (1, 2), (2, 3), (3, 4)):
        U += bond_energy(calc_bond(c[i], c[j]))

    for i, j, k in ((0, 1, 2), (1, 2, 3), (2, 3, 4)):
        U += angle_energy(calc_angle(c[i], c[j], c[k]))

    for i, j, k, l in ((0, 1, 2, 3), (1, 2, 3, 4)):
        U += torsion_energy(calc_dihedral(c[i], c[j], c[k], c[l]))

    U += lj_energy(calc_bond(c[0], c[4]), EPS_15, SIG_15)
    return float(U)


def forces_numerical(coords: np.ndarray, h: float = 1e-5) -> np.ndarray:
    """Numerical Cartesian forces from central finite differences [K/Å]."""
    c = np.asarray(coords, dtype=float)
    forces = np.zeros_like(c)

    for i in range(c.shape[0]):
        for d in range(c.shape[1]):
            plus = c.copy()
            minus = c.copy()
            plus[i, d] += h
            minus[i, d] -= h
            forces[i, d] = -(total_energy(plus) - total_energy(minus)) / (2.0 * h)

    return forces
