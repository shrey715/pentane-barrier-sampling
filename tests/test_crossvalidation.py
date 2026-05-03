import numpy as np
import pytest

from pentane.config_loader import CFG
from pentane.forcefield import total_energy
from pentane.geometry import build_pentane, calc_dihedral
from pentane.units import K_to_kJmol, kJmol_to_K


def _angles_match(a: float, b: float, atol: float = 1e-10) -> bool:
    return abs(a - b) < atol or abs(abs(a - b) - 2.0 * np.pi) < atol


def _remd_dihedral_deg(pos, i, j, k, l):
    b1 = pos[j] - pos[i]
    b2 = pos[k] - pos[j]
    b3 = pos[l] - pos[k]
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    nn1, nn2 = np.linalg.norm(n1), np.linalg.norm(n2)
    if nn1 < 1e-12 or nn2 < 1e-12:
        return 0.0
    n1u, n2u = n1 / nn1, n2 / nn2
    m = np.cross(n1u, b2 / np.linalg.norm(b2))
    return np.degrees(np.arctan2(np.dot(m, n2u), np.dot(n1u, n2u)))


def test_dihedral_convention_consistency():
    """build_pentane(phi, pi) should recover phi through calc_dihedral."""
    for phi_deg in [-180, -120, -60, 0, 60, 120, 180]:
        phi_rad = np.radians(phi_deg)
        coords = build_pentane(phi_rad, np.pi, CFG)
        recovered = calc_dihedral(coords[0], coords[1], coords[2], coords[3])
        assert _angles_match(recovered, phi_rad)


def test_dihedral_convention_matches_remd():
    """The package dihedral and the REMD-style dihedral must agree."""
    for phi_deg in [-170, -60, 0, 60, 170]:
        phi_rad = np.radians(phi_deg)
        coords = build_pentane(phi_rad, np.pi, CFG)
        pkg_result = np.degrees(calc_dihedral(coords[0], coords[1], coords[2], coords[3]))
        remd_result = _remd_dihedral_deg(coords, 0, 1, 2, 3)
        assert abs(pkg_result - remd_result) < 1e-8


def test_trans_is_global_minimum():
    """Trans should be lower in energy than gauche."""
    e_trans = total_energy(build_pentane(np.pi, np.pi, CFG))
    e_gauche = total_energy(build_pentane(np.radians(60), np.pi, CFG))
    assert e_trans < e_gauche


def test_unit_roundtrip():
    """K -> kJ/mol -> K should be identity."""
    values = np.array([0.0, 100.0, 500.0, 2292.0])
    assert np.allclose(kJmol_to_K(K_to_kJmol(values)), values, rtol=1e-12)


def test_energy_matches_expected_scale():
    """All-trans energy should be finite and near the expected minimum."""
    coords = build_pentane(np.pi, np.pi, CFG)
    energy_k = total_energy(coords)
    assert np.isfinite(energy_k)
    assert abs(energy_k) < 50.0
