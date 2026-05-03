"""
units.py - Unit conversion helpers between K and kJ/mol conventions.
"""
import numpy as np


kB_kJ_per_mol = 8.31446261815324e-3


def K_to_kJmol(value: float | np.ndarray) -> float | np.ndarray:
    """Convert a value from Kelvin to kJ/mol."""
    return np.asarray(value) * kB_kJ_per_mol


def kJmol_to_K(value: float | np.ndarray) -> float | np.ndarray:
    """Convert a value from kJ/mol to Kelvin."""
    return np.asarray(value) / kB_kJ_per_mol


def degrees_to_rad(value: float | np.ndarray) -> float | np.ndarray:
    """Convert degrees to radians."""
    return np.radians(value)


def rad_to_degrees(value: float | np.ndarray) -> float | np.ndarray:
    """Convert radians to degrees."""
    return np.degrees(value)
