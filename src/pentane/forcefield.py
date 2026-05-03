"""
forcefield.py — TraPPE-UA energy functions for n-pentane.

Energy contributions (TraPPE-UA, Martin & Siepmann 1998):
  • 3 × angle bending   : C1-C2-C3, C2-C3-C4, C3-C4-C5
  • 2 × torsion         : C1-C2-C3-C4, C2-C3-C4-C5
  • 1 × LJ 1-5          : C1···C5  (both CH3; only non-excluded pair)

Deliberately ABSENT:
  • Bond stretch  — TraPPE-UA rigid bonds; enforced by build_pentane()
  • 1-4 LJ        — absorbed into torsion coefficients per TraPPE-UA spec

All parameters read from CFG at import time — no magic numbers here.
"""
import numpy as np
from pentane.config_loader import CFG
from pentane.geometry import calc_angle, calc_dihedral

_a  = CFG["angles"]
_t  = CFG["torsion"]
_nb = CFG["nonbonded"]

K_THETA = _a["k_theta_K"]
THETA0  = np.radians(_a["theta0_deg"])

C0 = _t["c0_K"]
C1 = _t["c1_K"]
C2 = _t["c2_K"]
C3 = _t["c3_K"]

EPS15   = _nb["eps_CH3_K"]       # C1–C5 both CH3; combining rule trivial
SIG15   = _nb["sigma_CH3_ang"]


# ── Individual energy terms ─────────────────────────────────────────────────

def angle_energy(theta: float) -> float:
    """Harmonic angle energy [K].  U = (k/2)(θ − θ₀)²"""
    return 0.5 * K_THETA * (theta - THETA0) ** 2


def torsion_energy(phi: float) -> float:
    """
    Ryckaert-Bellemans torsion potential [K].
    U = c0 + c1(1+cosφ) + c2(1−cos2φ) + c3(1+cos3φ)
    """
    return (C0
            + C1 * (1.0 + np.cos(phi))
            + C2 * (1.0 - np.cos(2.0 * phi))
            + C3 * (1.0 + np.cos(3.0 * phi)))


def torsion_force(phi: float) -> float:
    """Generalised torsional force −dU_tors/dφ [K/rad]."""
    return -(
        -C1 * np.sin(phi)
        + 2.0 * C2 * np.sin(2.0 * phi)
        - 3.0 * C3 * np.sin(3.0 * phi)
    )


def lj_energy(r: float, eps: float, sigma: float) -> float:
    """Lennard-Jones 12-6 energy [K].  U = 4ε[(σ/r)¹²−(σ/r)⁶]"""
    sr6 = (sigma / r) ** 6
    return 4.0 * eps * (sr6 * sr6 - sr6)


# ── Full molecule energy ────────────────────────────────────────────────────

def total_energy(coords: np.ndarray) -> float:
    """
    Full TraPPE-UA potential energy for n-pentane [K, with k_B = 1].

    Parameters
    ----------
    coords : np.ndarray, shape (5, 3)
        Cartesian positions in Angstrom, ordered C1…C5.

    Returns
    -------
    U : float  [K]
    """
    c = coords
    U = 0.0

    # 3 angle bends
    U += angle_energy(calc_angle(c[0], c[1], c[2]))   # C1-C2-C3
    U += angle_energy(calc_angle(c[1], c[2], c[3]))   # C2-C3-C4
    U += angle_energy(calc_angle(c[2], c[3], c[4]))   # C3-C4-C5

    # 2 backbone torsions
    U += torsion_energy(calc_dihedral(c[0], c[1], c[2], c[3]))  # C1-C2-C3-C4
    U += torsion_energy(calc_dihedral(c[1], c[2], c[3], c[4]))  # C2-C3-C4-C5

    # 1-5 LJ: C1(CH3)···C5(CH3) — the only non-excluded pair
    r15 = float(np.linalg.norm(c[4] - c[0]))
    U += lj_energy(r15, EPS15, SIG15)

    return U
