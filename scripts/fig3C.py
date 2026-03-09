#!/usr/bin/env python
# coding: utf-8

import os
import csv
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, AutoMinorLocator, MaxNLocator, FixedLocator
from scipy.optimize import minimize_scalar
from scipy import stats
from qutip import *
import sys
from pathlib import Path

# 让 Python 能找到仓库根目录
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from paths import DATA
# ==========================================================
# Config: files in same folder
# ==========================================================
HERE = os.path.dirname(os.path.abspath(__file__))

IQ_EXP_CSV  = DATA / "I_q_exp.csv"         # header: Time_us,I_exp,I_err
POP_EXP_CSV = DATA / "bursting_population.csv"  # no header: time,population,error

# bursting logic split points (in us, from your simulation segmentation)
SPLIT_POINTS_BURSTING_US = [25, 35, 60, 70, 95, 105, 120]

# ==========================================================
# Matplotlib: Arial
# ==========================================================
mpl.rcParams.update({
    'font.family': 'Arial',
    'font.sans-serif': ['Arial'],
    'mathtext.fontset': 'custom',
    'mathtext.rm': 'Arial',
    'mathtext.it': 'Arial:italic',
    'mathtext.bf': 'Arial:bold',
    'axes.unicode_minus': False,

    'font.size': 20,
    'axes.titlesize': 30,
    'axes.labelsize': 22,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 18,

    'axes.linewidth': 2,
    'grid.alpha': 0.3,
})

# ==========================================================
# 1) CSV Readers (experiment)
# ==========================================================
def read_Iq_exp_csv(path: str):
    """CSV with header: Time_us,I_exp,I_err"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")

    t, I, Ierr = [], [], []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        required = {"Time_us", "I_exp", "I_err"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ValueError(f"I_q_exp.csv header must contain {required}, got {reader.fieldnames}")

        for row in reader:
            t.append(float(row["Time_us"]))
            I.append(float(row["I_exp"]))
            Ierr.append(float(row["I_err"]))

    t = np.asarray(t, dtype=float)
    I = np.asarray(I, dtype=float)
    Ierr = np.asarray(Ierr, dtype=float)

    order = np.argsort(t)
    return t[order], I[order], Ierr[order]


def read_population_csv_noheader(path: str, delimiter=","):
    """
    CSV without header: time,population,error
    (error optional; not used for unweighted fit)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")

    raw = np.loadtxt(path, delimiter=delimiter)
    if raw.ndim != 2 or raw.shape[1] < 2:
        raise ValueError("population_exp.csv must have >=2 columns: time,population,(error)")

    t = raw[:, 0].astype(float)
    y = raw[:, 1].astype(float)
    yerr = raw[:, 2].astype(float) if raw.shape[1] >= 3 else None

    order = np.argsort(t)
    t = t[order]
    y = y[order]
    if yerr is not None:
        yerr = yerr[order]
    return t, y, yerr

# ==========================================================
# 2) Gamma fitting utilities (piecewise, snapped boundaries)
# ==========================================================
def _estimate_baseline_C_global(t, y, lower=0.0):
    """Estimate baseline C globally by minimizing SSE of linear fit in log(y-C)."""
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    ymin = float(np.min(y))

    def sse_for_C(C):
        if np.any(y - C <= 0):
            return 1e18
        ly = np.log(y - C)
        A = np.vstack([np.ones_like(t), t]).T
        coef, *_ = np.linalg.lstsq(A, ly, rcond=None)
        pred = A @ coef
        return float(np.sum((ly - pred) ** 2))

    upper = max(lower + 1e-12, ymin * 0.999)
    if upper <= lower:
        return 0.0

    res = minimize_scalar(sse_for_C, bounds=(lower, upper), method="bounded")
    return float(res.x) if res.success else 0.0


def _snap_to_grid(exp_t, split_points_theory):
    exp_t = np.asarray(exp_t, dtype=float)
    snapped = []
    for s in split_points_theory:
        idx = int(np.argmin(np.abs(exp_t - float(s))))
        snapped.append(float(exp_t[idx]))
    return sorted(set(snapped))


def fit_piecewise_gamma_from_population(
    pop_t, pop_y, split_points_theory, target_t, *,
    min_points=2,
    verbose=True
):
    """
    Piecewise gamma fit with OVERLAPPED boundaries:
    - Each segment uses CLOSED interval [t0, t1], so boundary points belong to both neighbor segments.
    - Baseline C is estimated ONCE globally.
    - min_points default=2 (two points define a line).
    Returns: gamma_fit_on_target_t, info
    """
    pop_t = np.asarray(pop_t, dtype=float)
    pop_y = np.asarray(pop_y, dtype=float)
    target_t = np.asarray(target_t, dtype=float)

    if pop_t.size < 2:
        raise ValueError("population data too short.")

    # 1) Global baseline C
    C_global = _estimate_baseline_C_global(pop_t, pop_y, lower=0.0)
    if np.any(pop_y - C_global <= 0):
        raise ValueError(
            f"Global C={C_global} makes (y-C)<=0. "
            "Check population values or adjust baseline."
        )
    ly_all = np.log(pop_y - C_global)

    # 2) snapped boundaries
    snapped = _snap_to_grid(pop_t, split_points_theory)
    boundaries = [float(pop_t.min())] + snapped + [float(pop_t.max())]
    boundaries = sorted(set(boundaries))

    segments = []
    for k in range(len(boundaries) - 1):
        t0, t1 = boundaries[k], boundaries[k + 1]

        # ✅ CLOSED interval for BOTH ends -> boundary points are shared by adjacent segments
        mask = (pop_t >= t0) & (pop_t <= t1)
        idxs = np.where(mask)[0]
        n = int(len(idxs))

        if n < min_points:
            # keep the segment but mark gamma as NaN (we'll fill later)
            segments.append({
                "t0": float(t0), "t1": float(t1),
                "n": n,
                "gamma": np.nan,
                "gamma_err": np.nan,
                "R2": np.nan,
            })
            continue

        t_seg = pop_t[idxs]
        ly_seg = ly_all[idxs]

        if n == 2:
            slope = (ly_seg[1] - ly_seg[0]) / (t_seg[1] - t_seg[0] + 1e-15)
            intercept = ly_seg[0] - slope * t_seg[0]
            gamma = float(-slope)
            segments.append({
                "t0": float(t0), "t1": float(t1),
                "n": n,
                "gamma": gamma,
                "gamma_err": np.nan,
                "R2": 1.0,
            })
        else:
            slope, intercept, r_value, p_value, std_err = stats.linregress(t_seg, ly_seg)
            gamma = float(-slope)
            segments.append({
                "t0": float(t0), "t1": float(t1),
                "n": n,
                "gamma": gamma,
                "gamma_err": float(std_err) if np.isfinite(std_err) else np.nan,
                "R2": float(r_value**2),
            })

    # 3) Fill NaN gammas by nearest available segment (forward then backward)
    gammas = np.array([seg["gamma"] for seg in segments], dtype=float)
    # forward fill
    last = np.nan
    for i in range(len(gammas)):
        if np.isfinite(gammas[i]):
            last = gammas[i]
        else:
            gammas[i] = last
    # backward fill (for leading NaNs)
    last = np.nan
    for i in range(len(gammas) - 1, -1, -1):
        if np.isfinite(gammas[i]):
            last = gammas[i]
        else:
            gammas[i] = last
    # if still NaN (all NaN), set 0
    gammas = np.where(np.isfinite(gammas), gammas, 0.0)
    for seg, g in zip(segments, gammas):
        seg["gamma"] = float(g)

    # 4) Build gamma(t) on target_t (piecewise-constant, priority: first matching segment)
    # 4) Build gamma(t) on target_t (piecewise-constant, priority: RIGHT segment at overlap)
    gamma_fit = np.full_like(target_t, np.nan, dtype=float)

    for seg in segments:
        m = (target_t >= seg["t0"]) & (target_t <= seg["t1"])
        # ✅ overwrite: later segments take precedence at overlap
        gamma_fit[m] = seg["gamma"]

# fill any remaining NaNs (numerical edges)
    if np.any(~np.isfinite(gamma_fit)):
        gamma_fit[target_t < boundaries[0]] = segments[0]["gamma"]
        gamma_fit[target_t > boundaries[-1]] = segments[-1]["gamma"]
        gamma_fit[~np.isfinite(gamma_fit)] = segments[-1]["gamma"]

    info = {
        "C_global": float(C_global),
        "split_points_theory": [float(x) for x in split_points_theory],
        "split_points_snapped": snapped,
        "boundaries": boundaries,
        "segments": segments,
    }

    if verbose:
        print("\n=== Piecewise gamma fit (overlapped boundaries) ===")
        print("C_global =", info["C_global"])
        print("theory splits :", info["split_points_theory"])
        print("snapped splits:", info["split_points_snapped"])
        for i, seg in enumerate(info["segments"], 1):
            print(f"seg{i}: [{seg['t0']:.1f},{seg['t1']:.1f}]  n={seg['n']}  "
                  f"gamma={seg['gamma']:.6e}  R2={seg['R2']}")
    return gamma_fit, info


# ==========================================================
# 3) QuTiP simulation: bursting mode (all in-script)
# ==========================================================
def simulate_bursting(total_t=160.0, measure_times=161):
    """
    Implements your bursting J2 schedule:
      i<=25: J2=0
      25<i<=35: active
      35<i<60: 0
      60<=i<70: active
      70<=i<95: 0
      95<=i<105: active
      105<=i<120: 0
      120<=i<end: active
    Returns:
      t (len=measure_times-1),
      I_sim (len=measure_times-1),
      gamma_sim (len=measure_times-1),
      V_sim (len=measure_times-1),
      signal_sim (len=measure_times-1),
      bias
    """
    total_t = float(total_t)
    measure_times = int(measure_times)

    tlist2 = np.linspace(0, total_t, measure_times)
    dt = total_t / (measure_times - 1)

    dim = 7
    Delta = -5.23 * 2 * np.pi
    beta = 0.0
    Gamma = 19.6 * 2 * np.pi

    options = Options(nsteps=1500000)
    basis_states = [basis(dim, i) for i in range(dim)]

    I_th = 0.2
    bias = -0.001

    # operators
    Sx = basis_states[1] * basis_states[0].dag() + basis_states[0] * basis_states[1].dag()
    Sy = -1j * basis_states[1] * basis_states[0].dag() + 1j * basis_states[0] * basis_states[1].dag()
    Sz = basis_states[1] * basis_states[1].dag() - basis_states[0] * basis_states[0].dag()
    Iden = basis_states[1] * basis_states[1].dag() + basis_states[0] * basis_states[0].dag()
    Iden_excited = sum(basis_states[k] * basis_states[k].dag() for k in [2, 3, 4, 5, 6])

    def U(theta, phi):
        return (np.cos(theta/2) * Iden
                - 1j * np.sin(theta/2) * (np.cos(phi) * Sx + np.sin(phi) * Sy)
                + Iden_excited)

    gamma_train = -bias / I_th
    Dt = np.linspace(0, dt, 130)
    delta1 = 1.0

    # initial state
    a = b = c = d = 1
    psi0 = (a*basis_states[1] + b*basis_states[0] + c*basis_states[2] + d*basis_states[3]).unit()

    p0 = [float(b**2/(a**2+b**2+c**2+d**2))]
    p1 = [float(a**2/(a**2+b**2+c**2+d**2))]
    p2 = [float(c**2/(a**2+b**2+c**2+d**2))]

    # Hamiltonian parts
    H_offset = (-Delta) * basis_states[6] * basis_states[6].dag() + (Delta) * basis_states[4] * basis_states[4].dag()
    H_structure1 = basis_states[1] * basis_states[5].dag() + basis_states[5] * basis_states[1].dag()
    H_structure2 = (basis_states[2] * basis_states[4].dag() + basis_states[4] * basis_states[2].dag()
                    + basis_states[3] * basis_states[6].dag() + basis_states[6] * basis_states[3].dag())

    collapse_ops = [
        *[np.sqrt(Gamma / 3) * basis_states[k] * basis_states[5].dag() for k in [2, 1, 3]],
        *[np.sqrt(Gamma / 3) * basis_states[k] * basis_states[4].dag() for k in [1, 2, 0]],
        *[np.sqrt(Gamma / 3) * basis_states[k] * basis_states[6].dag() for k in [3, 0, 1]]
    ]

    def build_H(J1, J2):
        return H_offset + J1 * H_structure1 + J2 * H_structure2

    # storage
    states = psi0
    Sigy = [float(np.real(expect(Sy, psi0)))]
    I_sim = []
    gamma_sim = []
    V_sim = []
    signal_sim = []

    # J1 fixed by trained gamma
    J1 = float((3 * Gamma * gamma_train / 8) ** 0.5)
    J2_list = [0.0]

    for i in range(measure_times - 1):
        H2 = build_H(J1, J2_list[i])

        result = mesolve(
            H2, states, Dt, collapse_ops,
            [Sx, Sy, Sz,
             basis_states[0] * basis_states[0].dag(),
             basis_states[1] * basis_states[1].dag(),
             basis_states[2] * basis_states[2].dag()],
            options=options
        )
        result1 = mesolve(H2, states, Dt, collapse_ops, [], options=options)

        # measurement rotation for Sy
        Uy = U(np.pi/2, -delta1*(i+1))
        rho_y = Uy * result1.states[-1] * Uy.dag()
        Sy_val = float(np.real(rho_y[1, 1] - rho_y[0, 0]))
        Sigy.append(Sy_val)

        # populations
        p0.append(float(result.expect[3][-1]))
        p1.append(float(result.expect[4][-1]))
        p2.append(float(result.expect[5][-1]))

        # gamma1 from p1
        denom = p1[i] if p1[i] != 0 else 1e-12
        g1 = -(p1[i+1] - p1[i]) / (dt * denom)
        gamma_sim.append(float(g1))

        # Iq(sim)
        Iq = -0.75 * Sigy[i]
        I_sim.append(float(Iq))

        # Vq(sim)
        Vq = Iq * g1
        V_sim.append(float(Vq))
        signal_sim.append(float(np.sign(Vq + bias)))

        # bursting schedule for J2
        def J2_active_value(idx):
            denom2 = (1 - p0[idx] - p1[idx])
            if denom2 <= 1e-12:
                denom2 = 1e-12
            return float((2 * J1**2 * p1[idx] * (Gamma**2 + 4*Delta**2) / (denom2 * Gamma**2))**0.5)

        if i <= 25:
            J2_list.append(0.0)
        elif 25 < i <= 35:
            J2_list.append(J2_active_value(i+1))
        elif 35 < i < 60:
            J2_list.append(0.0)
        elif 60 <= i < 70:
            J2_list.append(J2_active_value(i+1))
        elif 70 <= i < 95:
            J2_list.append(0.0)
        elif 95 <= i < 105:
            J2_list.append(J2_active_value(i+1))
        elif 105 <= i < 120:
            J2_list.append(0.0)
        else:
            # 120 <= i < end
            if i < (measure_times - 2):
                J2_list.append(J2_active_value(i+1))
            else:
                J2_list.append(J2_list[-1])

        states = result1.states[-1]

    t = tlist2[:measure_times - 1]
    return t, np.asarray(I_sim), np.asarray(gamma_sim), np.asarray(V_sim), np.asarray(signal_sim), bias

# ==========================================================
# 4) Main: bursting pipeline
# ==========================================================
def main():
    # --- simulate bursting ---
    t_sim, I_sim, gamma_sim, V_sim, signal_sim, bias = simulate_bursting(total_t=160.0, measure_times=161)

    # --- experimental Iq (CSV) ---
    t_Iexp, I_exp_raw, I_err_raw = read_Iq_exp_csv(IQ_EXP_CSV)

    # interpolate I(exp) onto simulation grid
    I_exp = np.interp(t_sim, t_Iexp, I_exp_raw, left=I_exp_raw[0], right=I_exp_raw[-1])
    I_err = np.interp(t_sim, t_Iexp, I_err_raw, left=I_err_raw[0], right=I_err_raw[-1])

    # --- experimental population (CSV) -> fit piecewise gamma (snapped) ---
    pop_t, pop_y, pop_yerr = read_population_csv_noheader(POP_EXP_CSV, delimiter=",")
    print(pop_t)
    gamma_fit, fit_info = fit_piecewise_gamma_from_population(
        pop_t, pop_y,
        split_points_theory=SPLIT_POINTS_BURSTING_US,
        target_t=t_sim,
        verbose=True
    )

    # --- experimental Vq from fitted gamma ---
    V_exp = I_exp * gamma_fit
    V_exp_err = np.abs(gamma_fit) * I_err
    signal_exp = np.sign(V_exp + bias)

    # ==========================================================
    # Plot
    # ==========================================================
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 7), dpi=200, sharex=True,
        gridspec_kw={'hspace': 0, 'height_ratios': [1.5, 1]}
    )

    ax1.plot(t_sim, V_sim,
             color="#080808", linewidth=4, alpha=0.95, zorder=3, solid_capstyle='round',
             label="Simulation")

    ax1.errorbar(t_sim, V_exp, yerr=V_exp_err,
                 fmt='o', color="#EE053F", markersize=6,
                 markeredgecolor="#8A1830", markeredgewidth=2,
                 capsize=6, capthick=2, elinewidth=1, zorder=4,
                 label="Experiment (CSV)")

    ax1.axhline(y=-bias, color="#EE0B0B", linestyle='--', linewidth=4)

    ax2.plot(t_sim, signal_sim,
             color="#080808", linewidth=4, alpha=0.95, zorder=3, solid_capstyle='round')

    ax2.scatter(t_sim, signal_exp,
                zorder=4, s=40, color="#F8083C",
                edgecolors="#8A1830", linewidths=2)

    ax2.set_xlabel('Time (μs)', fontweight='bold', labelpad=10)
    ax1.set_ylabel(r'$\mathbf{V}_\mathbf{q}$', fontweight='bold', labelpad=7)

    ax1.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax1.yaxis.set_major_locator(MaxNLocator(4))
    ax1.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax1.ticklabel_format(style='sci', axis='y', scilimits=(-2, 2))
    ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.6)
    ax1.set_facecolor('#f8f9fa')
    # ax1.legend(frameon=False, loc="upper right")

    ax2.set_ylabel('Signal', fontweight='bold', labelpad=22)
    ax2.yaxis.set_major_locator(FixedLocator([-1, 1]))
    ax2.set_ylim(-1.1, 1.1)
    ax2.grid(True, alpha=0.3, linestyle='-', linewidth=0.6)
    ax2.set_facecolor('#f8f9fa')

    ax1.spines['bottom'].set_visible(False)
    ax1.tick_params(bottom=False, labelbottom=False)

    ax2.xaxis.set_major_locator(MaxNLocator(6))
    ax2.xaxis.set_minor_locator(AutoMinorLocator(2))

    for time_point in t_sim[::max(1, len(t_sim)//8)]:
        ax1.axvline(x=time_point, color='gray', linestyle=':', alpha=0.4, linewidth=0.8)
        ax2.axvline(x=time_point, color='gray', linestyle=':', alpha=0.4, linewidth=0.8)

    fig.patch.set_facecolor('white')
    plt.tight_layout()
    plt.show()

    # ==========================================================
    # Save outputs (optional but useful for paper)
    # ==========================================================
    # out_dir = HERE
    # pd.DataFrame({
    #     "Time_us": t_sim,
    #     "gamma_exp_fit": gamma_fit
    # }).to_csv(os.path.join(out_dir, "bursting_gamma_exp_fit_piecewise.csv"), index=False, encoding="utf-8-sig")

    # pd.DataFrame({
    #     "Time_us": t_sim,
    #     "I_exp_interp": I_exp,
    #     "I_err_interp": I_err,
    #     "gamma_exp_fit": gamma_fit,
    #     "Vq_exp": V_exp,
    #     "Vq_exp_err": V_exp_err,
    #     "I_sim": I_sim,
    #     "gamma_sim": gamma_sim,
    #     "Vq_sim": V_sim,
    #     "signal_exp": signal_exp,
    #     "signal_sim": signal_sim
    # }).to_csv(os.path.join(out_dir, "bursting_Vq_compare_exp_sim.csv"), index=False, encoding="utf-8-sig")

    # pd.DataFrame([{
    #     "split_points_theory": str(fit_info["split_points_theory"]),
    #     "split_points_snapped": str(fit_info["split_points_snapped"]),
    #     "boundaries": str(fit_info["boundaries"]),
    #     "segments": str(fit_info["segments"])
    # }]).to_csv(os.path.join(out_dir, "bursting_gamma_fit_summary.csv"), index=False, encoding="utf-8-sig")

    # print("\nSaved outputs in:", out_dir)
    # print(" - bursting_gamma_exp_fit_piecewise.csv")
    # print(" - bursting_Vq_compare_exp_sim.csv")
    # print(" - bursting_gamma_fit_summary.csv")


if __name__ == "__main__":
    main()
