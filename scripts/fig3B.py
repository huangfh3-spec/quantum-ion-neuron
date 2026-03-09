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
# Config: files in the same folder as this script
# ==========================================================
# HERE = os.path.dirname(os.path.abspath(__file__))

IQ_EXP_CSV   = DATA / "I_q_exp.csv"          # header: Time_us,I_exp,I_err
POP_EXP_CSV  = DATA / "phasic_population.csv"   # no header: time,population,error

# "theoretical" split from simulation logic
SPLIT_T_US_THEORY = 55.0

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
# 1) Read experimental CSVs
# ==========================================================
def read_Iq_exp(path: str):
    """header: Time_us,I_exp,I_err"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")

    t, I, Ierr = [], [], []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t.append(float(row["Time_us"]))
            I.append(float(row["I_exp"]))
            Ierr.append(float(row["I_err"]))

    t = np.array(t, dtype=float)
    I = np.array(I, dtype=float)
    Ierr = np.array(Ierr, dtype=float)

    order = np.argsort(t)
    return t[order], I[order], Ierr[order]

def read_population_noheader(path: str):
    """
    no header: time,population,error
    (error column is read but not used in unweighted fit)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")

    raw = np.loadtxt(path, delimiter=",")  # if not comma-separated: delimiter=None
    if raw.ndim != 2 or raw.shape[1] < 2:
        raise ValueError("population_exp.csv must have >=2 columns: time, population (3rd error optional)")

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
# 2) Fit piecewise gamma from population
#    model: log(y - C) = a + b t, gamma = -b
#    (unweighted fit)
# ==========================================================
def estimate_baseline_C(t, y, lower=0.0):
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

def fit_gamma_single_segment(t_seg, y_seg):
    C = estimate_baseline_C(t_seg, y_seg, lower=0.0)
    if np.any(y_seg - C <= 0):
        raise ValueError("Estimated C makes (y-C)<=0 in this segment. Check population data.")
    ly = np.log(y_seg - C)
    slope, intercept, r_value, p_value, std_err = stats.linregress(t_seg, ly)
    gamma = -slope
    return gamma, C, r_value**2, std_err

def find_nearest_time_point(t_arr, t0):
    idx = int(np.argmin(np.abs(t_arr - t0)))
    return idx, float(t_arr[idx])

def piecewise_gamma_from_population(pop_t, pop_y, split_t_theory, target_t):
    """
    split_t_theory: e.g. 55us (from simulation logic)
    BUT: use nearest experimental time point pop_t[idx] as actual boundary.
    """
    if pop_t.size < 6:
        raise ValueError("population data too short for piecewise fit.")

    idx_split, split_t_exp = find_nearest_time_point(pop_t, split_t_theory)

    # segment definition anchored to experimental sampling:
    # seg1: t <= split_t_exp (includes idx_split)
    # seg2: t >  split_t_exp
    mask1 = pop_t <= split_t_exp
    mask2 = pop_t >  split_t_exp

    if mask1.sum() < 3 or mask2.sum() < 3:
        raise ValueError(
            f"After splitting at nearest experimental point t={split_t_exp}, "
            f"one segment has <3 points. Please check population_exp.csv time coverage."
        )

    gamma1, C1, r2_1, gerr1 = fit_gamma_single_segment(pop_t[mask1], pop_y[mask1])
    gamma2, C2, r2_2, gerr2 = fit_gamma_single_segment(pop_t[mask2], pop_y[mask2])

    gamma_fit = np.where(target_t <= split_t_exp, gamma1, gamma2).astype(float)

    info = {
        "split_theory": float(split_t_theory),
        "split_exp": float(split_t_exp),
        "split_idx": int(idx_split),
        "seg1": {"gamma": gamma1, "C": C1, "R2": r2_1, "gamma_err": gerr1,
                 "t_range": (float(pop_t[mask1].min()), float(pop_t[mask1].max())),
                 "n": int(mask1.sum())},
        "seg2": {"gamma": gamma2, "C": C2, "R2": r2_2, "gamma_err": gerr2,
                 "t_range": (float(pop_t[mask2].min()), float(pop_t[mask2].max())),
                 "n": int(mask2.sum())},
    }
    return gamma_fit, info

# ==========================================================
# 3) QuTiP simulation (all in-script)
# ==========================================================
def run_simulation(total_t=160, measure_times=161):
    tlist2 = np.linspace(0, total_t, measure_times)
    dt = total_t / (measure_times - 1)

    dim = 7
    Delta = -5.23 * 2 * np.pi
    beta = 0
    Gamma = 19.6 * 2 * np.pi

    options = Options(nsteps=1500000)
    basis_states = [basis(dim, i) for i in range(dim)]

    I_th = 0.2
    bias = -0.001

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
    delta1 = 1

    # initial state
    a = b = c = d = 1
    psi0 = (a*basis_states[1] + b*basis_states[0] + c*basis_states[2] + d*basis_states[3]).unit()

    p0_0 = b**2/(a**2+b**2+c**2+d**2)
    p1_0 = a**2/(a**2+b**2+c**2+d**2)
    p2_0 = c**2/(a**2+b**2+c**2+d**2)

    H_offset = (-Delta) * basis_states[6] * basis_states[6].dag() + (Delta) * basis_states[4] * basis_states[4].dag()
    H_structure1 = basis_states[1] * basis_states[5].dag() + basis_states[5] * basis_states[1].dag()
    H_structure2 = (basis_states[2] * basis_states[4].dag() + basis_states[4] * basis_states[2].dag()
                    + basis_states[3] * basis_states[6].dag() + basis_states[6] * basis_states[3].dag())

    collapse_ops = [
        *[np.sqrt(Gamma / 3) * basis_states[k] * basis_states[5].dag() for k in [2, 1, 3]],
        *[np.sqrt(Gamma / 3) * basis_states[k] * basis_states[4].dag() for k in [1, 2, 0]],
        *[np.sqrt(Gamma / 3) * basis_states[k] * basis_states[6].dag() for k in [3, 0, 1]]
    ]

    def build_Hamiltonian_J2(J1, J2):
        return H_offset + J1 * H_structure1 + J2 * H_structure2

    # storage
    states = psi0
    p0 = [p0_0]
    p1 = [p1_0]
    p2 = [p2_0]

    Sigy = [float(np.real(expect(Sy, psi0)))]
    I1 = []         # Iq(sim) each step
    gamma1 = []     # gamma(sim) each step
    v_m = []
    signal = []

    # initial J1 (scalar)
    J1 = float((3 * Gamma * gamma_train / 8) ** 0.5)
    J2_list = [0.0]

    total_t = float(total_t)
    measure_times = int(measure_times)

    for i in range(measure_times - 1):
        H2 = build_Hamiltonian_J2(J1, J2_list[i])

        result = mesolve(
            H2, states, Dt, collapse_ops,
            [Sx, Sy, Sz,
             basis_states[0] * basis_states[0].dag(),
             basis_states[1] * basis_states[1].dag(),
             basis_states[2] * basis_states[2].dag()],
            options=options
        )
        result1 = mesolve(H2, states, Dt, collapse_ops, [], options=options)

        Uy = U(np.pi/2, -delta1*(i+1))
        rho_y = Uy * result1.states[-1] * Uy.dag()
        Sy_val = float(np.real(rho_y[1, 1] - rho_y[0, 0]))
        Sigy.append(Sy_val)

        # populations
        p0.append(float(result.expect[3][-1]))
        p1.append(float(result.expect[4][-1]))
        p2.append(float(result.expect[5][-1]))

        # gamma(sim) from p1
        denom = p1[i] if p1[i] != 0 else 1e-12
        g1 = -(p1[i+1] - p1[i]) / (dt * denom)
        gamma1.append(float(g1))

        # Iq(sim)
        Iq = -0.75 * Sigy[i]
        I1.append(float(Iq))

        # Vq(sim)
        v_m.append(float(Iq * g1))
        signal.append(float(np.sign(v_m[-1] - (-bias))))  # sign(v + bias) equivalently sign(v - (-bias))

        
        if i <= 55:
            J2_list.append(0.0)
        else:
            denom2 = (1 - p0[i+1] - p1[i+1])
            if denom2 <= 1e-12:
                denom2 = 1e-12
            J2_next = (2 * J1**2 * p1[i+1] * (Gamma**2 + 4*Delta**2) / (denom2 * Gamma**2))**0.5
            J2_list.append(float(J2_next))

        states = result1.states[-1]

    t = tlist2[:measure_times - 1]  # 160 points
    return t, np.array(I1), np.array(gamma1), np.array(v_m), np.array(signal), float(bias)

# ==========================================================
# Main
# ==========================================================
# --- Run simulation ---
t_sim, I_sim, gamma_sim, v_sim, signal_sim, bias = run_simulation(total_t=160, measure_times=161)

# --- Load experimental Iq ---
t_Iexp, I_exp, I_err = read_Iq_exp(IQ_EXP_CSV)

# interpolate Iq(exp) onto simulation grid
I_exp_itp = np.interp(t_sim, t_Iexp, I_exp, left=I_exp[0], right=I_exp[-1])
I_err_itp = np.interp(t_sim, t_Iexp, I_err, left=I_err[0], right=I_err[-1])

# --- Load population(exp) and fit piecewise gamma(exp), with boundary snapped to nearest experimental point ---
pop_t, pop_y, pop_yerr = read_population_noheader(POP_EXP_CSV)
gamma_exp_fit, fit_info = piecewise_gamma_from_population(pop_t, pop_y, SPLIT_T_US_THEORY, t_sim)

# print("\n=== Piecewise gamma(exp) from population (boundary snapped to nearest exp point) ===")
# print(f"theory split: {fit_info['split_theory']:.3f} us")
# print(f"exp split   : {fit_info['split_exp']:.3f} us (idx={fit_info['split_idx']})")
# print(f"seg1 n={fit_info['seg1']['n']}, t_range={fit_info['seg1']['t_range']}, gamma={fit_info['seg1']['gamma']:.8e}, R2={fit_info['seg1']['R2']:.4f}")
# print(f"seg2 n={fit_info['seg2']['n']}, t_range={fit_info['seg2']['t_range']}, gamma={fit_info['seg2']['gamma']:.8e}, R2={fit_info['seg2']['R2']:.4f}")

# --- Build Vq(exp) ---
v_exp = I_exp_itp * gamma_exp_fit
v_exp_err = np.abs(gamma_exp_fit) * I_err_itp
signal_exp = np.sign(v_exp + bias)

# ==========================================================
# Plot
# ==========================================================
fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(10, 7), dpi=200, sharex=True,
    gridspec_kw={'hspace': 0, 'height_ratios': [1.5, 1]}
)

ax1.plot(t_sim, v_sim, color="#0D6412", linewidth=4, alpha=0.95, zorder=3, solid_capstyle='round', label="Simulation")
ax1.errorbar(t_sim, v_exp, yerr=v_exp_err, fmt='o', color="#EE053F", markersize=6,
             markeredgecolor="#8A1830", markeredgewidth=2,
             capsize=6, capthick=2, elinewidth=1, zorder=4, label="Experiment (CSV)")

ax1.axhline(y=-bias, color="#EE0B0B", linestyle='--', linewidth=4)

ax2.plot(t_sim, signal_sim, color="#0D6412", linewidth=4, alpha=0.95, zorder=3, solid_capstyle='round')
ax2.scatter(t_sim, signal_exp, zorder=4, s=40, color="#F8083C", edgecolors="#8A1830", linewidths=2)

ax2.set_xlabel('Time (μs)', fontweight='bold', labelpad=10)
ax1.set_ylabel(r'$\mathbf{V}_\mathbf{q}$', fontweight='bold', labelpad=7)

ax1.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
ax1.yaxis.set_major_locator(MaxNLocator(5))
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
# Save outputs
# ==========================================================
# out_dir = HERE

# pd.DataFrame({
#     "Time_us": t_sim,
#     "gamma_exp_fit": gamma_exp_fit
# }).to_csv(os.path.join(out_dir, "gamma_exp_fit_piecewise.csv"), index=False, encoding="utf-8-sig")

# pd.DataFrame({
#     "Time_us": t_sim,
#     "I_exp_interp": I_exp_itp,
#     "I_err_interp": I_err_itp,
#     "gamma_exp_fit": gamma_exp_fit,
#     "Vq_exp": v_exp,
#     "Vq_exp_err": v_exp_err,
#     "I_sim": I_sim,
#     "gamma_sim": gamma_sim,
#     "Vq_sim": v_sim,
#     "signal_exp": signal_exp,
#     "signal_sim": signal_sim
# }).to_csv(os.path.join(out_dir, "Vq_compare_exp_sim.csv"), index=False, encoding="utf-8-sig")

# pd.DataFrame([{
#     "split_theory_us": fit_info["split_theory"],
#     "split_exp_us": fit_info["split_exp"],
#     "split_exp_idx": fit_info["split_idx"],

#     "seg1_gamma": fit_info["seg1"]["gamma"],
#     "seg1_gamma_err": fit_info["seg1"]["gamma_err"],
#     "seg1_C": fit_info["seg1"]["C"],
#     "seg1_R2": fit_info["seg1"]["R2"],
#     "seg1_n": fit_info["seg1"]["n"],
#     "seg1_tmin": fit_info["seg1"]["t_range"][0],
#     "seg1_tmax": fit_info["seg1"]["t_range"][1],

#     "seg2_gamma": fit_info["seg2"]["gamma"],
#     "seg2_gamma_err": fit_info["seg2"]["gamma_err"],
#     "seg2_C": fit_info["seg2"]["C"],
#     "seg2_R2": fit_info["seg2"]["R2"],
#     "seg2_n": fit_info["seg2"]["n"],
#     "seg2_tmin": fit_info["seg2"]["t_range"][0],
#     "seg2_tmax": fit_info["seg2"]["t_range"][1],
# }]).to_csv(os.path.join(out_dir, "gamma_fit_summary.csv"), index=False, encoding="utf-8-sig")

# print("\nSaved outputs in:", out_dir)
# print(" - gamma_exp_fit_piecewise.csv")
# print(" - Vq_compare_exp_sim.csv")
# print(" - gamma_fit_summary.csv")
