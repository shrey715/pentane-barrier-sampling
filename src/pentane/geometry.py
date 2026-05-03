"""
geometry.py — Build 3D coordinates and compute geometric quantities.

Public API
----------
build_pentane(phi1, phi2, cfg) -> np.ndarray  shape (5, 3) [Å]
calc_dihedral(a, b, c, d)     -> float  [rad], in (-π, π]
calc_angle(a, b, c)           -> float  [rad], in [0, π]

All bond lengths and equilibrium angles come from cfg — never hardcoded.
The dihedral convention is IUPAC: phi = 0 is cis, phi = ±π is trans.
build_pentane and calc_dihedral are consistent — i.e.
    calc_dihedral(*build_pentane(phi, phi, cfg)[[0,1,2,3]]) ≈ phi
"""
import numpy as np


def _place_atom(
    A: np.ndarray, B: np.ndarray, C: np.ndarray,
    r_new: float, theta: float, phi_iupac: float
) -> np.ndarray:
    """
    Place atom D at fixed bond length r_new from C, with:
      • bond angle angle(B, C, D) = theta
      • IUPAC dihedral dihedral(A, B, C, D) = phi_iupac

    Standard z-matrix to Cartesian formula (Parsons et al. 2005).
    phi = 0 → cis (A and D on same side); phi = ±π → trans.
    """
    bc      = C - B
    bc_norm = bc / np.linalg.norm(bc)

    ab = B - A
    n  = np.cross(ab, bc)
    n_l = np.linalg.norm(n)
    n  = n / n_l if n_l > 1e-12 else np.array([0.0, 0.0, 1.0])

    # In-plane perpendicular to bc (defines phi = 0 direction = cis)
    m = np.cross(n, bc_norm)

    # Deviation from linear chain direction
    sin_th = np.sin(np.pi - theta)   # > 0
    cos_th = np.cos(np.pi - theta)   # sign tells us which side of C

    d_hat = (
        cos_th    * bc_norm
        + sin_th * np.cos(phi_iupac) * m
        - sin_th * np.sin(phi_iupac) * n
    )
    return C + r_new * d_hat


def build_pentane(phi1: float, phi2: float, cfg: dict) -> np.ndarray:
    """
    Build n-pentane Cartesian coordinates from IUPAC dihedrals.

    phi1 : C1-C2-C3-C4 dihedral [rad], IUPAC convention (π = trans)
    phi2 : C2-C3-C4-C5 dihedral [rad], IUPAC convention (π = trans)
    cfg  : loaded trappe_ua.toml dict

    Returns
    -------
    coords : np.ndarray, shape (5, 3), [Å]
        Cartesian positions of C1…C5. Bond lengths and angles enforced
        exactly from cfg. calc_dihedral(coords[0:4]) recovers phi1.
    """
    r   = cfg["bonds"]["r_CC_ang"]
    th  = np.radians(cfg["angles"]["theta0_deg"])

    # C1 at origin, C2 along +x
    C1 = np.array([0.0, 0.0, 0.0])
    C2 = np.array([r, 0.0, 0.0])

    # C3: place in xy-plane with correct angle at C2
    C3 = C2 + r * np.array([np.cos(np.pi - th), np.sin(np.pi - th), 0.0])

    # C4 and C5: use z-matrix placement with IUPAC dihedral convention
    C4 = _place_atom(C1, C2, C3, r, th, phi1)
    C5 = _place_atom(C2, C3, C4, r, th, phi2)

    return np.array([C1, C2, C3, C4, C5])


def calc_dihedral(a: np.ndarray, b: np.ndarray,
                  c: np.ndarray, d: np.ndarray) -> float:
    """
    IUPAC dihedral angle for atoms a-b-c-d.

    Returns
    -------
    phi : float
        Dihedral in radians, range (-π, π].
        phi = 0 is cis; phi = ±π is trans.
    """
    b1 = b - a
    b2 = c - b
    b3 = d - c

    n1 = np.cross(b1, b2)
    n1_l = np.linalg.norm(n1)
    n1 = n1 / n1_l if n1_l > 1e-12 else n1

    n2 = np.cross(b2, b3)
    n2_l = np.linalg.norm(n2)
    n2 = n2 / n2_l if n2_l > 1e-12 else n2

    m1 = np.cross(n1, b2 / np.linalg.norm(b2))
    return float(np.arctan2(np.dot(m1, n2), np.dot(n1, n2)))


def calc_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """
    Bond angle at atom b, for the sequence a-b-c.

    Returns
    -------
    theta : float  [rad], in [0, π].
    """
    ba = a - b
    bc = c - b
    cos_t = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return float(np.arccos(np.clip(cos_t, -1.0, 1.0)))
