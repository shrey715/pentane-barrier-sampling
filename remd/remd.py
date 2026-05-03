import argparse
import sys
import time
from pathlib import Path

import numpy as np
import numpy.random as npr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import logsumexp

try:
    import openmm
    import openmm.unit as unit
    from openmm import (LangevinMiddleIntegrator, CustomBondForce,
                        CustomAngleForce, CustomTorsionForce,
                        CustomNonbondedForce, Context, Platform, Vec3)
except ImportError:
    print("ERROR: OpenMM not installed.  pip install openmm")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
#  PHYSICAL CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
kB_kJ = 8.31446261815324e-3          # kJ mol-1 K-1

# ─────────────────────────────────────────────────────────────────────────────
#  TraPPE-UA PARAMETERS  (n-pentane, 5 united atoms)
# ─────────────────────────────────────────────────────────────────────────────
N_ATOMS     = 5
MASSES_AMU  = np.array([15.035, 14.027, 14.027, 14.027, 15.035])

EPS_K    = np.array([98.0,  46.0,  46.0,  46.0,  98.0])      # eps/kB  [K]
SIGMA_NM = np.array([0.375, 0.395, 0.395, 0.395, 0.375])     # sigma  [nm]
EPS_KJ   = EPS_K * kB_kJ

BOND_PAIRS    = [(0,1),(1,2),(2,3),(3,4)]
BOND_R0_NM    = 0.154
KB_BOND_KJ    = kB_kJ * 452900.0 * 100.0   # K/Ang2 -> kJ/mol/nm2

ANGLE_TRIPLES = [(0,1,2),(1,2,3),(2,3,4)]
THETA0_RAD    = np.radians(114.0)
K_THETA_KJ    = kB_kJ * 62500.0            # kJ/mol/rad2

# TraPPE OPLS torsion  U = sum ci f(theta)   [kJ/mol]
TORSION_QUADS = [(0,1,2,3),(1,2,3,4)]
C_KJ = np.array([0.0, 355.03, -68.19, 791.32]) * kB_kJ

# 1-2, 1-3, 1-4 ALL excluded for TraPPE n-alkanes; only 1-5 (atoms 0<->4) kept.
EXCLUDED_PAIRS = set()
for _b in BOND_PAIRS:
    EXCLUDED_PAIRS.add((min(_b), max(_b)))
for _a in ANGLE_TRIPLES:
    EXCLUDED_PAIRS.add((min(_a[0], _a[2]), max(_a[0], _a[2])))
for _t in TORSION_QUADS:
    EXCLUDED_PAIRS.add((min(_t[0], _t[3]), max(_t[0], _t[3])))

LJ_CUTOFF_NM = 1.0


# ─────────────────────────────────────────────────────────────────────────────
#  TEMPERATURE LADDER
# ─────────────────────────────────────────────────────────────────────────────

def make_temperature_ladder(T_min, T_max, n_replicas):
    return np.geomspace(T_min, T_max, n_replicas)


# ─────────────────────────────────────────────────────────────────────────────
#  BUILD OPENMM SYSTEM
# ─────────────────────────────────────────────────────────────────────────────

def build_system() -> openmm.System:
    """
    TraPPE-UA n-pentane force field.

    FIX: CustomTorsionForce exposes the torsion angle as 'theta' (built-in).
    Using 'phi' raises  OpenMMException: Unknown variable in expression: phi.
    """
    system = openmm.System()
    for m in MASSES_AMU:
        system.addParticle(m * unit.amu)

    # Harmonic bonds
    bf = CustomBondForce("0.5*k*(r-r0)^2")
    bf.addPerBondParameter("k");  bf.addPerBondParameter("r0")
    for (i, j) in BOND_PAIRS:
        bf.addBond(i, j, [KB_BOND_KJ, BOND_R0_NM])
    system.addForce(bf)

    # Harmonic angles
    af = CustomAngleForce("0.5*k*(theta-theta0)^2")
    af.addPerAngleParameter("k");  af.addPerAngleParameter("theta0")
    for (i, j, k) in ANGLE_TRIPLES:
        af.addAngle(i, j, k, [K_THETA_KJ, THETA0_RAD])
    system.addForce(af)

    # OPLS torsion — 'theta' is the OpenMM built-in variable name here
    tf = CustomTorsionForce(
        "c0 + c1*(1+cos(theta)) + c2*(1-cos(2*theta)) + c3*(1+cos(3*theta))"
    )
    tf.addPerTorsionParameter("c0");  tf.addPerTorsionParameter("c1")
    tf.addPerTorsionParameter("c2");  tf.addPerTorsionParameter("c3")
    for (i, j, k, l) in TORSION_QUADS:
        tf.addTorsion(i, j, k, l, list(C_KJ))
    system.addForce(tf)

    # LJ  (Lorentz-Berthelot, CutoffNonPeriodic, no PBC)
    lj = CustomNonbondedForce(
        "4*eps*((sig/r)^12-(sig/r)^6);"
        "eps=sqrt(eps1*eps2); sig=0.5*(sig1+sig2)"
    )
    lj.addPerParticleParameter("eps");  lj.addPerParticleParameter("sig")
    lj.setCutoffDistance(LJ_CUTOFF_NM * unit.nanometer)
    lj.setNonbondedMethod(CustomNonbondedForce.CutoffNonPeriodic)
    for i in range(N_ATOMS):
        lj.addParticle([EPS_KJ[i], SIGMA_NM[i]])
    for (i, j) in EXCLUDED_PAIRS:
        lj.addExclusion(i, j)
    system.addForce(lj)

    return system


# ─────────────────────────────────────────────────────────────────────────────
#  INITIAL CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

def build_initial_positions_nm():
    theta_turn = np.radians(180.0 - np.degrees(THETA0_RAD))
    pos = np.zeros((N_ATOMS, 3))
    pos[1] = [BOND_R0_NM, 0.0, 0.0]
    bond = np.array([BOND_R0_NM, 0.0, 0.0])
    sign = 1.0
    for idx in range(2, N_ATOMS):
        c, s = np.cos(sign * theta_turn), np.sin(sign * theta_turn)
        rot = np.array([[c, -s], [s, c]])
        bxy = rot @ bond[:2]
        bond = np.array([bxy[0], bxy[1], 0.0])
        pos[idx] = pos[idx-1] + bond
        sign *= -1.0
    return [Vec3(*p) for p in pos] * unit.nanometer


# ─────────────────────────────────────────────────────────────────────────────
#  REPLICA CONTEXTS
# ─────────────────────────────────────────────────────────────────────────────

def create_replicas(system, temps, dt_ps, seed):
    for pname in ["CUDA", "OpenCL", "CPU"]:
        try:
            platform = Platform.getPlatformByName(pname); break
        except Exception:
            continue
    print(f"  OpenMM platform: {platform.getName()}")

    init_pos = build_initial_positions_nm()
    replicas = []
    for i, T in enumerate(temps):
        integ = LangevinMiddleIntegrator(
            T * unit.kelvin, 1.0 / unit.picosecond, dt_ps * unit.picoseconds)
        integ.setRandomNumberSeed(seed + i * 100)
        ctx = Context(system, integ, platform)
        rng_l = npr.default_rng(seed + i)
        perturb = rng_l.normal(0, 0.002, (N_ATOMS, 3))
        perturbed = [Vec3(*(np.array(p.value_in_unit(unit.nanometer)) + perturb[k]))
                     * unit.nanometer for k, p in enumerate(init_pos)]
        ctx.setPositions(perturbed)
        ctx.setVelocitiesToTemperature(T * unit.kelvin, seed + i + 50)
        replicas.append(ctx)
    return replicas


# ─────────────────────────────────────────────────────────────────────────────
#  OBSERVABLES
# ─────────────────────────────────────────────────────────────────────────────

def get_positions_nm(ctx):
    return np.array(ctx.getState(getPositions=True)
                      .getPositions(asNumpy=True)
                      .value_in_unit(unit.nanometer))


def dihedral_deg(pos, i, j, k, l):
    b1 = pos[j]-pos[i]; b2 = pos[k]-pos[j]; b3 = pos[l]-pos[k]
    n1 = np.cross(b1, b2);  n2 = np.cross(b2, b3)
    nn1, nn2 = np.linalg.norm(n1), np.linalg.norm(n2)
    if nn1 < 1e-12 or nn2 < 1e-12:
        return 0.0
    n1u, n2u = n1/nn1, n2/nn2
    cos_phi = np.clip(np.dot(n1u, n2u), -1.0, 1.0)
    m = np.cross(n1u, b2/np.linalg.norm(b2))
    return np.degrees(np.arctan2(np.dot(m, n2u), cos_phi))


def get_potential_kJ(ctx):
    return (ctx.getState(getEnergy=True)
               .getPotentialEnergy()
               .value_in_unit(unit.kilojoule_per_mole))


# ─────────────────────────────────────────────────────────────────────────────
#  REMD SWAP  (Metropolis)
# ─────────────────────────────────────────────────────────────────────────────

def attempt_swap(ctx_i, ctx_j, T_i, T_j, rng):
    E_i, E_j   = get_potential_kJ(ctx_i), get_potential_kJ(ctx_j)
    bi, bj     = 1.0/(kB_kJ*T_i), 1.0/(kB_kJ*T_j)
    delta = (bi - bj) * (E_j - E_i)
    if delta >= 0.0 or rng.random() < np.exp(delta):
        pi = ctx_i.getState(getPositions=True).getPositions()
        pj = ctx_j.getState(getPositions=True).getPositions()
        ctx_i.setPositions(pj);  ctx_j.setPositions(pi)
        ctx_i.setVelocitiesToTemperature(T_i * unit.kelvin)
        ctx_j.setVelocitiesToTemperature(T_j * unit.kelvin)
        return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
#  ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def histogram(dihedrals, n_bins=36):
    bins = np.linspace(-180, 180, n_bins + 1)
    d = np.asarray(dihedrals, dtype=float)
    if d.size == 0:
        return 0.5*(bins[:-1]+bins[1:]), np.zeros(n_bins)
    counts, _ = np.histogram(d, bins=bins)
    tot = counts.sum()
    return 0.5*(bins[:-1]+bins[1:]), counts/tot if tot > 0 else np.zeros(n_bins, float)


def pmf_from_probs(centers, probs, T):
    """F(phi) = -kBT ln P(phi), shifted so min = 0.  NaN where P = 0."""
    p = np.asarray(probs, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        f = np.where(p > 0, -kB_kJ * T * np.log(p), np.nan)
    f -= np.nanmin(f)
    return f


def wham_pmf(all_dihed, all_energ, temps, T_ref,
             n_bins=36, stride=1, max_iter=5000, tol=1e-10):
    """WHAM reweighting of all replicas -> unbiased PMF at T_ref."""
    temps = np.asarray(temps, float)
    beta  = 1.0 / (kB_kJ * temps)
    n_rep = len(temps)

    E_list, phi_list = [], []
    N_k = np.zeros(n_rep, float)
    for k in range(n_rep):
        E_k   = np.asarray(all_energ[k],  float)[::stride]
        phi_k = np.asarray(all_dihed[k],  float)[::stride]
        if E_k.size == 0:
            continue
        E_list.append(E_k);  phi_list.append(phi_k)
        N_k[k] = E_k.size

    if not E_list:
        return np.linspace(-175, 175, n_bins), np.full(n_bins, np.nan), np.zeros(n_bins)

    E_n   = np.concatenate(E_list)
    phi_n = np.concatenate(phi_list)
    logN  = np.log(np.maximum(N_k, 1.0))

    # Iterative WHAM
    f_k = np.zeros(n_rep)
    for _ in range(max_iter):
        log_d = logsumexp(logN[None,:] + f_k[None,:] - beta[None,:]*E_n[:,None], axis=1)
        f_new = -logsumexp(-beta[None,:]*E_n[:,None] - log_d[:,None], axis=0)
        f_new -= f_new[0]
        if np.max(np.abs(f_new - f_k)) < tol:
            f_k = f_new; break
        f_k = 0.5*f_k + 0.5*f_new

    log_d   = logsumexp(logN[None,:] + f_k[None,:] - beta[None,:]*E_n[:,None], axis=1)
    beta_r  = 1.0/(kB_kJ*float(T_ref))
    log_w   = -beta_r*E_n - log_d
    log_w  -= logsumexp(log_w)
    w       = np.exp(log_w)

    bins = np.linspace(-180.0, 180.0, n_bins+1)
    counts, _ = np.histogram(phi_n, bins=bins, weights=w)
    s = counts.sum()
    probs = counts/s if s > 0 else counts
    centers = 0.5*(bins[:-1]+bins[1:])
    return centers, pmf_from_probs(centers, probs, float(T_ref)), probs


def _interp_nan(y):
    """Replace NaN in y with linear interpolation from neighbours."""
    y = y.copy()
    nans = np.isnan(y)
    if nans.all():
        return y
    x = np.arange(len(y))
    y[nans] = np.interp(x[nans], x[~nans], y[~nans])
    return y


# ─────────────────────────────────────────────────────────────────────────────
#  CONVERGENCE DIAGNOSTICS
# ─────────────────────────────────────────────────────────────────────────────

def autocorr_time(x, max_lag_frac=0.5):
    """
    Integrated autocorrelation time via the Sokal windowed estimator.
    Returns (tau, N_eff).  tau is clamped to [1, N/2].
    """
    x = np.asarray(x, float)
    N = len(x)
    if N < 20:
        return 1.0, N
    max_lag = max(1, int(N * max_lag_frac))
    x = x - x.mean()
    # FFT-based normalised ACF
    f   = np.fft.rfft(x, n=2*N)
    acf = np.fft.irfft(f * np.conj(f))[:N].real
    if acf[0] == 0:
        return 1.0, N
    acf /= acf[0]
    # Sokal window: stop at first M where tau < M/5
    tau = 1.0
    for M in range(1, max_lag):
        tau += 2.0 * acf[M]
        if M >= 5 * tau:
            break
    tau   = float(np.clip(tau, 1.0, N // 2))
    N_eff = max(1, int(N / (2.0 * tau)))
    return tau, N_eff


def block_pmf(all_dihed, all_energ, temps, T_ref, n_blocks=5, n_bins=36):
    """
    Split trajectories into n_blocks equal windows.
    Return (centers, mean_pmf, std_pmf, pmf_blocks).
    NaN bins are interpolated before computing statistics.
    """
    n_rep = len(temps)
    n_s   = min(len(d) for d in all_dihed)
    bs    = n_s // n_blocks
    if bs < 20:
        # Not enough data for blocks — return full PMF with zero uncertainty
        centers, pmf_full, _ = wham_pmf(all_dihed, all_energ, temps, T_ref, n_bins=n_bins)
        return centers, pmf_full, np.zeros_like(pmf_full), np.array([pmf_full])

    blocks = []
    for b in range(n_blocks):
        s, e = b*bs, (b+1)*bs
        bd = [np.array(all_dihed[k][s:e]) for k in range(n_rep)]
        be = [np.array(all_energ[k][s:e]) for k in range(n_rep)]
        centers, pmf_b, _ = wham_pmf(bd, be, temps, T_ref, n_bins=n_bins)
        blocks.append(_interp_nan(pmf_b))

    blocks    = np.array(blocks)            # (n_blocks, n_bins)
    pmf_mean  = np.mean(blocks, axis=0)
    pmf_std   = np.std(blocks,  axis=0, ddof=1)
    return centers, pmf_mean, pmf_std, blocks


def running_pmf(all_dihed, all_energ, temps, T_ref, n_cp=8, n_bins=36):
    """Cumulative PMF at n_cp checkpoints."""
    n_rep = len(temps)
    n_s   = min(len(d) for d in all_dihed)
    min_pts = max(50, n_s // n_cp)
    cps   = np.unique(np.linspace(min_pts, n_s, n_cp, dtype=int))
    fracs = cps / n_s
    pmfs  = []
    for cp in cps:
        sd = [np.array(all_dihed[k][:cp]) for k in range(n_rep)]
        se = [np.array(all_energ[k][:cp]) for k in range(n_rep)]
        centers, pmf_cp, _ = wham_pmf(sd, se, temps, T_ref, n_bins=n_bins)
        pmfs.append(pmf_cp)
    return fracs, np.array(pmfs), centers


def replica_diffusion_report(rth, temps):
    n_rep = len(temps)
    print("\n  Replica diffusion (fraction of T-ladder visited):")
    for i in range(n_rep):
        visited = len(set(rth[i])) / n_rep
        filled  = round(visited * n_rep)
        bar = "X" * filled + "." * (n_rep - filled)
        print(f"    Replica {i:2d} ({temps[i]:5.0f} K start): [{bar}]  {visited*100:.0f}%")


# ─────────────────────────────────────────────────────────────────────────────
#  PLOTTING
# ─────────────────────────────────────────────────────────────────────────────

def _annotate_pmf_ax(ax, pmf_wham):
    ymax = np.nanmax(pmf_wham[np.isfinite(pmf_wham)]) if np.any(np.isfinite(pmf_wham)) else 20
    for xv, col in [(180,'grey'),(-180,'grey'),(60,'steelblue'),(-60,'steelblue'),(0,'red')]:
        ax.axvline(xv, ls=':', c=col, lw=1.0, alpha=0.5)
    ax.text( 170, ymax*0.05, 'trans',     ha='right',  fontsize=8, color='grey')
    ax.text(-170, ymax*0.05, 'trans',     ha='left',   fontsize=8, color='grey')
    ax.text(  60, ymax*0.55, 'gauche+',  ha='center', fontsize=8, color='steelblue')
    ax.text( -60, ymax*0.55, 'gauche-',  ha='center', fontsize=8, color='steelblue')
    ax.text(   0, ymax*0.88, 'eclipsed', ha='center', fontsize=8, color='red')
    ax.set_xlim(-180, 180)
    ax.set_ylim(bottom=0)


def make_plots(all_dihed, all_energ, temps, swap_rates, rth, out_dir: Path):
    out_dir.mkdir(exist_ok=True)
    n_rep  = len(temps)
    COLORS = plt.cm.plasma(np.linspace(0.1, 0.9, n_rep))
    T_ref  = float(temps[0])
    N_BINS = 36

    # 1. Dihedral time series
    sel = [0, n_rep//2, n_rep-1]
    fig, axes = plt.subplots(len(sel), 1, figsize=(12, 8), sharex=True)
    for ax, ri in zip(axes, sel):
        ax.plot(all_dihed[ri], lw=0.4, color=COLORS[ri], alpha=0.8)
        for yv, ls, col in [(180,'--','grey'),(-180,'--','grey'),
                             (60,':','steelblue'),(-60,':','salmon')]:
            ax.axhline(yv, ls=ls, c=col, lw=0.8)
        ax.set_ylabel(f"phi [deg]\nT={temps[ri]:.0f}K", fontsize=9)
        ax.set_ylim(-185, 185)
    axes[-1].set_xlabel("Sample index", fontsize=10)
    fig.suptitle("REMD - Backbone Dihedral Time Series", fontsize=12)
    fig.tight_layout(); fig.savefig(out_dir/"remd_dihedral_timeseries.png", dpi=150); plt.close(fig)

    # 2. Dihedral histograms
    ncols = 4; nrows = int(np.ceil(n_rep/ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3*nrows))
    for idx in range(nrows*ncols):
        ax = axes.flat[idx]
        if idx < n_rep:
            c, p = histogram(all_dihed[idx], N_BINS)
            ax.bar(c, p, width=10, color=COLORS[idx], alpha=0.85, edgecolor='k', lw=0.3)
            ax.set_title(f"T={temps[idx]:.0f}K", fontsize=8)
            ax.set_xlabel("phi [deg]", fontsize=7); ax.set_ylabel("P", fontsize=7)
            ax.set_xlim(-180, 180); ax.tick_params(labelsize=7)
        else:
            ax.set_visible(False)
    fig.suptitle("REMD - Dihedral Distributions", fontsize=12)
    fig.tight_layout(); fig.savefig(out_dir/"remd_dihedral_histograms.png", dpi=150); plt.close(fig)

    # 3. Main PMF with uncertainty band
    centers, pmf_wham, _ = wham_pmf(all_dihed, all_energ, temps, T_ref, n_bins=N_BINS)
    centers_b, pmf_mean, pmf_std, _ = block_pmf(all_dihed, all_energ, temps, T_ref,
                                                  n_blocks=5, n_bins=N_BINS)
    c0, p0 = histogram(all_dihed[0], N_BINS)
    pmf0   = pmf_from_probs(c0, p0, T_ref)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(centers, pmf_wham, lw=2.5, color='steelblue', label=f"WHAM PMF @ {T_ref:.0f} K")
    lo = np.where(np.isfinite(pmf_mean-pmf_std), pmf_mean-pmf_std, np.nan)
    hi = np.where(np.isfinite(pmf_mean+pmf_std), pmf_mean+pmf_std, np.nan)
    ax.fill_between(centers_b, lo, hi, alpha=0.25, color='steelblue', label="+/-1sigma (5 blocks)")
    ax.plot(c0, pmf0, lw=1.5, ls='--', color='coral',
            label=f"Direct hist @ {T_ref:.0f} K (replica 0)")
    _annotate_pmf_ax(ax, pmf_wham)
    ax.set_xlabel("Backbone dihedral phi [deg]", fontsize=12)
    ax.set_ylabel("F(phi) [kJ/mol]", fontsize=12)
    ax.set_title(f"PMF - n-Pentane REMD (TraPPE-UA)\n"
                 f"F(phi)=-kBT ln P(phi), min=0, T_ref={T_ref:.0f} K", fontsize=11)
    ax.legend(fontsize=10)
    fig.tight_layout(); fig.savefig(out_dir/"remd_pmf.png", dpi=150); plt.close(fig)
    print(f"  PMF -> {out_dir/'remd_pmf.png'}")

    # 4. Running PMF convergence
    fracs, pmf_list, centers_r = running_pmf(all_dihed, all_energ, temps, T_ref,
                                              n_cp=8, n_bins=N_BINS)
    CMAP = plt.cm.viridis(np.linspace(0.1, 0.95, len(fracs)))
    fig, ax = plt.subplots(figsize=(10, 5))
    for frac, pmf_cp, col in zip(fracs, pmf_list, CMAP):
        ax.plot(centers_r, pmf_cp, lw=1.5, color=col, alpha=0.85, label=f"{frac*100:.0f}%")
    ax.set_xlabel("phi [deg]", fontsize=12); ax.set_ylabel("F(phi) [kJ/mol]", fontsize=12)
    ax.set_title("Convergence - Running PMF (curves collapse when converged)", fontsize=11)
    ax.legend(fontsize=8, ncol=2, title="Data used")
    ax.set_xlim(-180, 180); ax.set_ylim(bottom=0)
    fig.tight_layout(); fig.savefig(out_dir/"remd_pmf_convergence.png", dpi=150); plt.close(fig)
    print(f"  Convergence -> {out_dir/'remd_pmf_convergence.png'}")

    # 5. Block-average spread
    _, _, _, pmf_blocks = block_pmf(all_dihed, all_energ, temps, T_ref,
                                     n_blocks=5, n_bins=N_BINS)
    fig, ax = plt.subplots(figsize=(10, 5))
    for b, pmf_b in enumerate(pmf_blocks):
        ax.plot(centers_b, pmf_b, lw=1.2, alpha=0.7, label=f"Block {b+1}")
    ax.plot(centers_b, pmf_mean, lw=2.5, color='k', label="Mean")
    ax.fill_between(centers_b, np.maximum(0, pmf_mean-pmf_std), pmf_mean+pmf_std,
                    alpha=0.2, color='k', label="+/-1sigma")
    ax.set_xlabel("phi [deg]", fontsize=12); ax.set_ylabel("F(phi) [kJ/mol]", fontsize=12)
    ax.set_title("Block-average PMF (5 blocks) - spread = statistical uncertainty", fontsize=11)
    ax.legend(fontsize=9); ax.set_xlim(-180, 180); ax.set_ylim(bottom=0)
    fig.tight_layout(); fig.savefig(out_dir/"remd_pmf_blocks.png", dpi=150); plt.close(fig)

    # 6. Swap acceptance rates (colour-coded)
    if swap_rates:
        fig, ax = plt.subplots(figsize=(9, 4))
        bars = ax.bar(range(len(swap_rates)), np.array(swap_rates)*100,
                      edgecolor='k', lw=0.5)
        for bar, r in zip(bars, swap_rates):
            bar.set_color('green' if 0.20 <= r <= 0.40 else
                          'orange' if r <= 0.70 else 'red')
        ax.set_xticks(range(len(swap_rates)))
        ax.set_xticklabels(
            [f"{temps[i]:.0f}->{temps[i+1]:.0f}" for i in range(len(temps)-1)],
            rotation=45, fontsize=8)
        ax.axhline(20, ls='--', c='green',  lw=1.2, label='20% lower target')
        ax.axhline(40, ls='--', c='orange', lw=1.2, label='40% upper target')
        ax.set_ylabel("Swap acceptance [%]", fontsize=10)
        ax.set_title("REMD Swap Acceptance (green=good, orange=ok, red=too high/low)", fontsize=10)
        ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(out_dir/"remd_swap_rates.png", dpi=150); plt.close(fig)

    # 7. Energy distributions
    fig, ax = plt.subplots(figsize=(10, 5))
    for i in range(n_rep):
        E = np.asarray(all_energ[i], float)
        if E.size < 2: continue
        bins = np.linspace(np.nanmin(E)-1, np.nanmax(E)+1, 60)
        c_e, edges = np.histogram(E, bins=bins, density=True)
        ax.plot(0.5*(edges[:-1]+edges[1:]), c_e, lw=1.2, color=COLORS[i],
                alpha=0.8, label=f"{temps[i]:.0f}K")
    ax.set_xlabel("Potential energy [kJ/mol]", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("Energy Distributions (good overlap -> efficient swaps)", fontsize=11)
    if n_rep <= 12:
        ax.legend(fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(out_dir/"remd_energy_distributions.png", dpi=150); plt.close(fig)

    # 8. Replica temperature walk
    fig, ax = plt.subplots(figsize=(12, 5))
    show = min(n_rep, 4)
    for i in range(show):
        ax.plot(rth[i], lw=0.8, alpha=0.8, label=f"Replica {i}")
    ax.set_yticks(range(n_rep))
    ax.set_yticklabels([f"{t:.0f}K" for t in temps], fontsize=8)
    ax.set_xlabel("Sample", fontsize=10); ax.set_ylabel("Temperature rung", fontsize=10)
    ax.set_title("Replica Temperature Walk (good mixing = diffuses across all rungs)", fontsize=11)
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out_dir/"remd_replica_walk.png", dpi=150); plt.close(fig)

    print(f"  All plots -> {out_dir}/")


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN REMD LOOP
# ─────────────────────────────────────────────────────────────────────────────

def run_remd(
    n_replicas   : int   = 12,
    T_min        : float = 120.0,
    T_max        : float = 600.0,
    total_steps  : int   = 100_000,
    swap_freq    : int   = 500,
    sample_every : int   = 50,      # record dihedral/energy every N steps (NEW)
    dt_ps        : float = 0.002,
    out_dir      : str   = "remd_output",
    seed         : int   = 42,
):
    print(f"\n{'='*65}")
    print(f"  REMD - n-Pentane (TraPPE-UA)  |  Backend: OpenMM")
    print(f"  Replicas : {n_replicas}  |  T: {T_min:.0f}-{T_max:.0f} K  (geometric)")
    print(f"  Steps    : {total_steps:,}  |  dt: {dt_ps*1000:.1f} fs")
    print(f"  Swap every {swap_freq} steps  |  sample every {sample_every} steps")
    print(f"{'='*65}\n")

    if sample_every > swap_freq:
        print(f"  WARNING: sample_every ({sample_every}) > swap_freq ({swap_freq}); "
              f"setting sample_every = swap_freq")
        sample_every = swap_freq

    if swap_freq % sample_every != 0:
        sample_every = swap_freq
        print(f"  WARNING: swap_freq not divisible by sample_every; "
              f"setting sample_every = swap_freq")

    temps = make_temperature_ladder(T_min, T_max, n_replicas)
    print("  Temperature ladder [K]:", "  ".join(f"{t:.1f}" for t in temps))

    system   = build_system()
    replicas = create_replicas(system, temps, dt_ps, seed)

    all_dihed = [[] for _ in range(n_replicas)]
    all_energ = [[] for _ in range(n_replicas)]
    rth       = [[] for _ in range(n_replicas)]   # temperature-rung history
    cur_rung  = list(range(n_replicas))

    swap_n   = np.zeros(n_replicas-1, dtype=int)
    swap_att = np.zeros(n_replicas-1, dtype=int)

    n_blocks     = total_steps // swap_freq
    steps_inner  = swap_freq // sample_every
    rng = npr.default_rng(seed)
    t0  = time.perf_counter()

    print(f"\n  Running {n_blocks} swap-blocks x {swap_freq} steps "
          f"({steps_inner} samples/block/replica) ...\n")

    for block in range(n_blocks):
        # Inner loop: sample every sample_every steps
        for step in range(steps_inner):
            for i, ctx in enumerate(replicas):
                ctx.getIntegrator().step(sample_every)
                pos = get_positions_nm(ctx)
                all_dihed[i].append(dihedral_deg(pos, 0, 1, 2, 3))
                all_energ[i].append(get_potential_kJ(ctx))
                rth[i].append(cur_rung[i])

        # REMD swaps (alternating parity)
        parity = block % 2
        for i in range(parity, n_replicas-1, 2):
            swap_att[i] += 1
            if attempt_swap(replicas[i], replicas[i+1], temps[i], temps[i+1], rng):
                swap_n[i] += 1
                cur_rung[i], cur_rung[i+1] = cur_rung[i+1], cur_rung[i]

        # Progress
        if (block+1) % max(1, n_blocks//10) == 0:
            elapsed  = time.perf_counter() - t0
            rate_str = " ".join(
                f"{swap_n[j]/swap_att[j]*100:.0f}%" if swap_att[j] > 0 else "n/a"
                for j in range(n_replicas-1))
            avg_E = np.mean([all_energ[j][-1] for j in range(n_replicas)])
            print(f"  Block {block+1:>5}/{n_blocks}  "
                  f"|  <E>={avg_E:+.1f} kJ/mol  "
                  f"|  elapsed {elapsed:.1f}s  "
                  f"|  swap%: [{rate_str}]")

    elapsed = time.perf_counter() - t0
    n_samples = len(all_dihed[0])
    print(f"\n  Finished in {elapsed:.1f}s  ({elapsed/60:.2f} min)")
    print(f"  Samples per replica: {n_samples}")

    # Swap summary
    swap_rates = [swap_n[i]/swap_att[i] if swap_att[i] > 0 else 0.0
                  for i in range(n_replicas-1)]
    print("\n  Swap acceptance rates:")
    for i in range(n_replicas-1):
        flag = "OK" if 0.20 <= swap_rates[i] <= 0.40 else ("~OK" if swap_rates[i] <= 0.70 else "TOO HIGH - widen T range")
        print(f"    {temps[i]:.0f}->{temps[i+1]:.0f}K : {swap_rates[i]*100:.1f}%  {flag}")
    if all(r > 0.60 for r in swap_rates):
        print("\n  *** All swap rates >60%: temperature spacing is too narrow.")
        print("      Replicas are nearly identical -> poor enhanced sampling.")
        print("      Recommended fix: widen T_min-T_max, or reduce n_replicas.")

    # ── Convergence diagnostics ───────────────────────────────────────────────
    print("\n  -- Convergence diagnostics ------------------------------------------")

    print(f"\n  Autocorrelation time of phi per replica  (N_samples = {n_samples}):")
    for i in range(n_replicas):
        tau, N_eff = autocorr_time(all_dihed[i])
        ok = "OK" if N_eff >= 200 else "need more samples"
        print(f"    Replica {i:2d} ({temps[i]:5.0f}K): "
              f"tau_act = {tau:6.1f} samples  N_eff = {N_eff:6d}  [{ok}]")

    replica_diffusion_report(rth, temps)

    T_ref = float(temps[0])
    centers_b, pmf_mean, pmf_std, _ = block_pmf(all_dihed, all_energ, temps, T_ref, n_blocks=5)
    max_dev = float(np.nanmax(pmf_std))
    print(f"\n  Block-average PMF (5 blocks) max pointwise sigma: {max_dev:.3f} kJ/mol")
    if max_dev < 1.0:
        print("    OK - PMF well converged (sigma < 1 kJ/mol)")
    elif max_dev < 3.0:
        print("    WARNING - Moderate uncertainty; consider more steps")
    else:
        print("    NOT CONVERGED - Run more steps or increase sample_every")

    # ── Output ────────────────────────────────────────────────────────────────
    out_path = Path(out_dir)
    make_plots(all_dihed, all_energ, temps, swap_rates, rth, out_path)

    out_path.mkdir(exist_ok=True)
    for i in range(n_replicas):
        fname = out_path / f"replica_{i:02d}_T{temps[i]:.0f}K.csv"
        np.savetxt(fname,
                   np.column_stack([all_dihed[i], all_energ[i]]),
                   delimiter=",", header="dihedral_deg,energy_kJmol", comments="")
    print(f"\n  CSV files -> {out_dir}/")

    return all_dihed, all_energ, temps, swap_rates


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="REMD n-pentane (TraPPE-UA) - OpenMM")
    p.add_argument("--replicas",      type=int,   default=12)
    p.add_argument("--T-min",         type=float, default=120.0,
                   help="Lowest replica T [K] (default 120)")
    p.add_argument("--T-max",         type=float, default=600.0)
    p.add_argument("--steps",         type=int,   default=100_000)
    p.add_argument("--swap-freq",     type=int,   default=500)
    p.add_argument("--sample-every",  type=int,   default=50,
                   help="Record dihedral/energy every N steps (must divide swap-freq)")
    p.add_argument("--dt",            type=float, default=0.002)
    p.add_argument("--out-dir",       type=str,   default="remd_output")
    p.add_argument("--seed",          type=int,   default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_remd(
        n_replicas   = args.replicas,
        T_min        = args.T_min,
        T_max        = args.T_max,
        total_steps  = args.steps,
        swap_freq    = args.swap_freq,
        sample_every = args.sample_every,
        dt_ps        = args.dt,
        out_dir      = args.out_dir,
        seed         = args.seed,
    )
