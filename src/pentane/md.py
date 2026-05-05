"""
md.py — Cartesian NVT molecular dynamics for n-pentane.

The integrator advances full Cartesian coordinates with a Nosé-Hoover
thermostat. The order parameter reported to the rest of the pipeline is the
C1-C2-C3-C4 dihedral extracted from the Cartesian trajectory.
"""
import numpy as np

from pentane.config_loader import CFG
from pentane.forcefield import forces_numerical, forces_numba
from pentane.geometry import build_all_trans, calc_dihedral

# Prefer the Numba-JIT evaluator (~20× faster); fall back to pure-Python.
_forces = forces_numba if forces_numba is not None else forces_numerical

# 1 u expressed in K·ps²/Å² under k_B = 1.
_U_TO_KPSA2 = 1.20272


def _center_of_mass(coords: np.ndarray, masses: np.ndarray) -> np.ndarray:
    return np.average(coords, axis=0, weights=masses)


def _center_of_mass_velocity(velocities: np.ndarray, masses: np.ndarray) -> np.ndarray:
    return np.average(velocities, axis=0, weights=masses)


def run_md(T: float, cfg: dict, seed: int = None) -> np.ndarray:
    """Run full Cartesian Nosé-Hoover MD and return the phi1 trajectory."""
    rng = np.random.default_rng(seed)
    n = int(cfg["simulation"]["n_steps"])
    dt = float(cfg["simulation"]["dt_ps"])
    tau_T = float(cfg["simulation"]["tau_T_ps"])

    masses_cfg = cfg.get("masses", {})
    masses_u = np.array([
        masses_cfg.get("m_CH3_u", 15.035),
        masses_cfg.get("m_CH2_u", 14.027),
        masses_cfg.get("m_CH2_u", 14.027),
        masses_cfg.get("m_CH2_u", 14.027),
        masses_cfg.get("m_CH3_u", 15.035),
    ], dtype=float)
    masses = masses_u * _U_TO_KPSA2
    masses_3d = masses[:, None]

    pos = build_all_trans(cfg)
    pos = pos - _center_of_mass(pos, masses)

    vel = rng.normal(0.0, np.sqrt(T / masses)[:, None], size=pos.shape)
    vel = vel - _center_of_mass_velocity(vel, masses)

    dof = 3 * pos.shape[0] - 3
    Q = dof * T * tau_T ** 2
    xi = 0.0

    forces = _forces(pos)
    traj = np.empty(n, dtype=float)

    for i in range(n):
        kinetic = 0.5 * np.sum(masses_3d * vel * vel)
        xi += 0.5 * dt * ((2.0 * kinetic - dof * T) / Q)

        acc = forces / masses_3d
        vel += 0.5 * dt * (acc - xi * vel)
        pos += dt * vel

        pos = pos - _center_of_mass(pos, masses)

        forces = _forces(pos)
        acc = forces / masses_3d
        vel += 0.5 * dt * (acc - xi * vel)

        # Second ξ half-kick: must use kinetic energy *before* COM removal so
        # that the thermostat sees the full kinetic energy consistent with
        # dof = 3N - 3.  (Itoh, Morishita & Okumura, J. Chem. Phys. 139,
        # 064103, 2013 — correct MTK operator-splitting order.)
        kinetic = 0.5 * np.sum(masses_3d * vel * vel)
        xi += 0.5 * dt * ((2.0 * kinetic - dof * T) / Q)
        vel = vel - _center_of_mass_velocity(vel, masses)

        traj[i] = calc_dihedral(pos[0], pos[1], pos[2], pos[3])
    return traj
