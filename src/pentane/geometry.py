"""
Molecular Geometry — n-Pentane Construction and Dihedral Calculation
====================================================================

Provides functions to:
1. Build an all-atom (united-atom sites) geometry of n-pentane from
   internal coordinates (bond lengths, bond angles, dihedral angles)
   using the Natural Extension Reference Frame (NeRF) algorithm.
2. Compute the dihedral angle from four Cartesian positions.
3. Verify bond-length correctness of a generated structure.

The NeRF algorithm places each new atom relative to the three preceding
atoms using local cylindrical coordinates defined by the bond vector,
bond angle, and dihedral angle.
"""

import numpy as np

from pentane.forcefield import BOND_LENGTH, BOND_ANGLE_RAD


def _add_atom(
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    length: float,
    theta: float,
    phi: float,
) -> np.ndarray:
    """
    Place a new atom D given three preceding atoms (A, B, C) and the
    internal coordinates (bond length, bond angle, dihedral angle).

    Uses the Natural Extension Reference Frame (NeRF) algorithm to
    define a local coordinate system from the B->C bond vector and the
    A-B-C plane normal.

    Parameters
    ----------
    A, B, C : ndarray, shape (3,)
        Cartesian positions of the three reference atoms.
    length : float
        Bond length C-D [Angstrom].
    theta : float
        Bond angle B-C-D [radians].
    phi : float
        Dihedral angle A-B-C-D [radians].

    Returns
    -------
    D : ndarray, shape (3,)
        Cartesian position of the new atom.
    """
    BC = C - B
    bc_hat = BC / np.linalg.norm(BC)

    AB = B - A
    n1 = np.cross(AB, BC)
    norm_n1 = np.linalg.norm(n1)

    # Handle collinear case: pick an arbitrary perpendicular direction
    if norm_n1 < 1e-10:
        n1 = np.array([0.0, 0.0, 1.0])
    else:
        n1 = n1 / norm_n1

    n2 = np.cross(bc_hat, n1)

    # Displacement vector in the local frame
    d = length * (
        np.cos(np.pi - theta) * bc_hat
        + np.sin(np.pi - theta) * (np.cos(phi) * n2 + np.sin(phi) * n1)
    )

    return C + d


def build_pentane(
    phi1: float = np.radians(180.0),
    phi2: float = np.radians(180.0),
) -> np.ndarray:
    """
    Construct the Cartesian coordinates of n-pentane (5 UA sites) from
    two backbone dihedral angles.

    The molecule is built sequentially:
      C1 at origin -> C2 along +x -> C3 in the xy-plane -> C4 via phi1 -> C5 via phi2

    Parameters
    ----------
    phi1 : float, optional
        Dihedral angle C1-C2-C3-C4 [radians]. Default: pi (trans).
    phi2 : float, optional
        Dihedral angle C2-C3-C4-C5 [radians]. Default: pi (trans).

    Returns
    -------
    coords : ndarray, shape (5, 3)
        Cartesian coordinates of the five united-atom sites [Angstrom].
    """
    C1 = np.array([0.0, 0.0, 0.0])
    C2 = np.array([BOND_LENGTH, 0.0, 0.0])
    C3 = C2 + BOND_LENGTH * np.array([
        np.cos(np.pi - BOND_ANGLE_RAD),
        np.sin(np.pi - BOND_ANGLE_RAD),
        0.0,
    ])

    C4 = _add_atom(C1, C2, C3, BOND_LENGTH, BOND_ANGLE_RAD, phi1)
    C5 = _add_atom(C2, C3, C4, BOND_LENGTH, BOND_ANGLE_RAD, phi2)

    return np.array([C1, C2, C3, C4, C5])


def calc_dihedral(coords: np.ndarray) -> float:
    """
    Compute the dihedral angle defined by four sequential atom positions.

    Uses the atan2-based formula that gives the signed dihedral angle
    in the range [-pi, pi].

    Parameters
    ----------
    coords : ndarray, shape (4, 3)
        Cartesian coordinates of four atoms defining the dihedral.

    Returns
    -------
    phi : float
        Dihedral angle [radians], in the range [-pi, pi].
    """
    b1 = coords[1] - coords[0]
    b2 = coords[2] - coords[1]
    b3 = coords[3] - coords[2]

    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)

    n1 = n1 / np.linalg.norm(n1)
    n2 = n2 / np.linalg.norm(n2)

    m1 = np.cross(n1, b2 / np.linalg.norm(b2))

    return float(np.arctan2(np.dot(m1, n2), np.dot(n1, n2)))


def verify_bonds(
    coords: np.ndarray,
    expected: float = BOND_LENGTH,
    tol: float = 1e-10,
) -> list:
    """
    Verify that all consecutive bond lengths match the expected value.

    Parameters
    ----------
    coords : ndarray, shape (N, 3)
        Cartesian coordinates of N atoms.
    expected : float
        Expected bond length [Angstrom].
    tol : float
        Tolerance for bond-length deviation.

    Returns
    -------
    bonds : list of (i, j, length) tuples
        Raises AssertionError if any bond deviates beyond tolerance.
    """
    bonds = []
    for i in range(len(coords) - 1):
        dist = float(np.linalg.norm(coords[i + 1] - coords[i]))
        bonds.append((i, i + 1, dist))
        assert abs(dist - expected) < tol, (
            f"Bond C{i+1}-C{i+2} = {dist:.6f} A, expected {expected:.4f} A"
        )
    return bonds
