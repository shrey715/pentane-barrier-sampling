"""
forcefield.py — Cartesian TraPPE-UA energy model for n-pentane.

The full potential includes bond stretch, angle bend, torsion, and the only
allowed intramolecular Lennard-Jones interaction for n-pentane (C1···C5).

All energies are expressed in Kelvin with k_B = 1.

Speed: when Numba is available, `forces_numba` is used in place of the
pure-Python finite-difference evaluator, giving ~20× speedup on MD.
"""
import math
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

# ── Numba JIT force evaluator ─────────────────────────────────────────────────
try:
    from numba import njit as _njit
    _NUMBA = True
except ImportError:  # pragma: no cover
    _NUMBA = False

if _NUMBA:
    # All constants baked in at compile time via closure-style module globals.
    # The function is fully self-contained: no Python calls inside the hot loop.
    _K_R   = K_R
    _R0    = R0
    _K_TH  = K_TH
    _TH0   = TH0
    _C0    = C0
    _C1    = C1
    _C2    = C2
    _C3    = C3
    _EPS15 = EPS_15
    _SIG15 = SIG_15

    @_njit(cache=True)
    def _nb_total_energy(c):
        """Numba-JIT TraPPE-UA total energy for n-pentane [K]. c shape (5,3)."""
        U = 0.0
        # -- Bonds (0-1, 1-2, 2-3, 3-4) --
        for i, j in ((0,1),(1,2),(2,3),(3,4)):
            dx = c[j,0]-c[i,0]; dy = c[j,1]-c[i,1]; dz = c[j,2]-c[i,2]
            r = math.sqrt(dx*dx + dy*dy + dz*dz)
            U += 0.5 * _K_R * (r - _R0)**2
        # -- Angles (0-1-2, 1-2-3, 2-3-4) --
        for i, j, k in ((0,1,2),(1,2,3),(2,3,4)):
            ux=c[i,0]-c[j,0]; uy=c[i,1]-c[j,1]; uz=c[i,2]-c[j,2]
            vx=c[k,0]-c[j,0]; vy=c[k,1]-c[j,1]; vz=c[k,2]-c[j,2]
            nu = math.sqrt(ux*ux+uy*uy+uz*uz)
            nv = math.sqrt(vx*vx+vy*vy+vz*vz)
            cos_th = (ux*vx+uy*vy+uz*vz)/(nu*nv)
            cos_th = max(-1.0, min(1.0, cos_th))
            theta = math.acos(cos_th)
            U += 0.5 * _K_TH * (theta - _TH0)**2
        # -- Torsions (0-1-2-3, 1-2-3-4) --
        for i, j, k, l in ((0,1,2,3),(1,2,3,4)):
            b1x=c[j,0]-c[i,0]; b1y=c[j,1]-c[i,1]; b1z=c[j,2]-c[i,2]
            b2x=c[k,0]-c[j,0]; b2y=c[k,1]-c[j,1]; b2z=c[k,2]-c[j,2]
            b3x=c[l,0]-c[k,0]; b3y=c[l,1]-c[k,1]; b3z=c[l,2]-c[k,2]
            n1x=b1y*b2z-b1z*b2y; n1y=b1z*b2x-b1x*b2z; n1z=b1x*b2y-b1y*b2x
            n2x=b2y*b3z-b2z*b3y; n2y=b2z*b3x-b2x*b3z; n2z=b2x*b3y-b2y*b3x
            nn1 = math.sqrt(n1x*n1x+n1y*n1y+n1z*n1z)
            nn2 = math.sqrt(n2x*n2x+n2y*n2y+n2z*n2z)
            nb2 = math.sqrt(b2x*b2x+b2y*b2y+b2z*b2z)
            cos_phi = (n1x*n2x+n1y*n2y+n1z*n2z)/(nn1*nn2)
            cos_phi = max(-1.0, min(1.0, cos_phi))
            mx=n1y*n2z-n1z*n2y; my=n1z*n2x-n1x*n2z; mz=n1x*n2y-n1y*n2x
            sin_phi = (mx*b2x+my*b2y+mz*b2z)/((nn1*nn2)*nb2)
            phi = math.atan2(sin_phi, cos_phi)
            U += _C0 + _C1*(1+math.cos(phi)) + _C2*(1-math.cos(2*phi)) + _C3*(1+math.cos(3*phi))
        # -- LJ C1···C5 --
        dx=c[4,0]-c[0,0]; dy=c[4,1]-c[0,1]; dz=c[4,2]-c[0,2]
        r = math.sqrt(dx*dx+dy*dy+dz*dz)
        sr6 = (_SIG15/r)**6
        U += 4.0 * _EPS15 * (sr6*sr6 - sr6)
        return U

    @_njit(cache=True)
    def _nb_forces(coords, h=1e-5):
        """Numba-JIT finite-difference forces [K/Å]. ~20× faster than pure Python."""
        forces = np.zeros((5, 3))
        for i in range(5):
            for d in range(3):
                plus  = coords.copy()
                minus = coords.copy()
                plus[i, d]  += h
                minus[i, d] -= h
                forces[i, d] = -(_nb_total_energy(plus) - _nb_total_energy(minus)) / (2.0*h)
        return forces

    def forces_numba(coords: np.ndarray) -> np.ndarray:
        """Public wrapper: Numba-JIT forces for n-pentane [K/Å]."""
        return _nb_forces(np.asarray(coords, dtype=np.float64))

else:  # pragma: no cover
    forces_numba = None  # caller must fall back to forces_numerical


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
