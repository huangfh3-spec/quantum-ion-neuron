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

# 
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from paths import DATA
# =======================
# Matplotlib (Arial)
# =======================
mpl.rcParams.update({
    'font.family': 'Arial',
    'font.sans-serif': ['Arial'],
    'mathtext.fontset': 'custom',
    'mathtext.rm': 'Arial',
    'mathtext.it': 'Arial:italic',
    'mathtext.bf': 'Arial:bold',
    'axes.unicode_minus': False,
})

# =======================
# Paths (same folder)
# =======================
# HERE = os.path.dirname(os.path.abspath(__file__))

# 1) 布居数数据：无表头三列：Time_us, population, error

POP_PATH= DATA / "spiking_population.csv"  
# 2) Iq 实验数据：有表头 Time_us,I_exp,I_err
Iq_exp_path = DATA / "I_q_exp.csv"

# =======================
# Read CSVs
# =======================
def read_population_noheader_3col(path: str):
    """
    无表头三列：time, population, error
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Population CSV not found: {path}")
    raw = np.loadtxt(path, delimiter=",")  # 若是空格/Tab分隔，改 delimiter=None
    if raw.ndim != 2 or raw.shape[1] < 3:
        raise ValueError("布居数CSV必须为无表头三列：Time_us, population, error")
    t = raw[:, 0].astype(float)
    y = raw[:, 1].astype(float)
    yerr = raw[:, 2].astype(float)
    order = np.argsort(t)
    return t[order], y[order], yerr[order]

def read_Iq_exp(path: str):
    """
    有表头：Time_us,I_exp,I_err
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"I_q_exp.csv not found: {path}")
    t, I, Ierr = [], [], []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t.append(float(row["Time_us"]))
            I.append(float(row["I_exp"]))
            Ierr.append(float(row["I_err"]))
    return np.array(t), np.array(I), np.array(Ierr)

# =======================
# Fit gamma from population (single-segment, unweighted)
# log(y - C) = a + b t, gamma = -b
# =======================
def estimate_baseline_C(t, y, lower=0.0):
    ymin = y.min()

    def sse_for_C(C):
        if np.any(y - C <= 0):
            return 1e18
        ly = np.log(y - C)
        A = np.vstack([np.ones_like(t), t]).T
        coef, *_ = np.linalg.lstsq(A, ly, rcond=None)
        pred = A @ coef
        return np.sum((ly - pred) ** 2)

    upper = max(lower + 1e-12, ymin * 0.999)
    if upper <= lower:
        return 0.0

    res = minimize_scalar(sse_for_C, bounds=(lower, upper), method="bounded")
    return float(res.x) if res.success else 0.0

pop_t, pop_y, pop_err = read_population_noheader_3col(POP_PATH)

C = estimate_baseline_C(pop_t, pop_y, lower=0.0)
if np.any(pop_y - C <= 0):
    raise ValueError("C 估计导致 y-C 非正。请检查布居数数据或手动限制 C。")

ly = np.log(pop_y - C)
slope, intercept, r_value, p_value, std_err = stats.linregress(pop_t, ly)
gamma_fit_value = -slope
gamma_fit_err = std_err  # 斜率标准误 -> gamma标准误

# print("\n" + "="*70)
# print("Single-segment fit from population:")
# print(f"C = {C:.8e}")
# print(f"slope b = {slope:.8e} ± {std_err:.8e}")
# print(f"gamma = {-slope:.8e} ± {std_err:.8e}")
# print(f"R^2 = {r_value**2:.6f}")
# print("="*70)

# =======================
# Load Iq experiment and build Vq_exp = I_exp * gamma
# =======================
time_exp, I_exp, I_err = read_Iq_exp(Iq_exp_path)


total_t = 160
measure_times = 161
tlist2 = np.linspace(0, total_t, measure_times)
dt = total_t / (measure_times - 1)


target_t = tlist2[:measure_times-1]  # 160 points
I_exp_interp = np.interp(target_t, time_exp, I_exp, left=I_exp[0], right=I_exp[-1])
I_err_interp = np.interp(target_t, time_exp, I_err, left=I_err[0], right=I_err[-1])

gamma_fit = np.full_like(target_t, fill_value=gamma_fit_value, dtype=float)

v_test = I_exp_interp * gamma_fit
v_err = np.abs(gamma_fit) * I_err_interp  

signal_fit = np.sign(v_test + (-0.001))  

# =======================



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
    # Qobj -> OK
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
p3_0 = d**2/(a**2+b**2+c**2+d**2)

H_structure1 = basis_states[1] * basis_states[5].dag() + basis_states[5] * basis_states[1].dag()

collapse_ops = [
    *[np.sqrt(Gamma / 3) * basis_states[k] * basis_states[5].dag() for k in [2, 1, 3]],
    *[np.sqrt(Gamma / 3) * basis_states[k] * basis_states[4].dag() for k in [1, 2, 0]],
    *[np.sqrt(Gamma / 3) * basis_states[k] * basis_states[6].dag() for k in [3, 0, 1]]
]

def build_Hamiltonian(J1):
    return J1 * H_structure1

# storage
p0 = [p0_0]
p1 = [p1_0]
p2 = [p2_0]
p3 = [p3_0]

Sigx_list = [expect(Sx, psi0)]
Sigy_list = [expect(Sy, psi0)]

I1 = [-0.75 * expect(Sy, psi0)]
gamma1 = []

states = psi0

J1 = float((3 * Gamma * gamma_train / 8) ** 0.5)  # scalar

for i in range(measure_times - 1):
    H2 = build_Hamiltonian(J1)

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
    Ux = U(np.pi/2, -delta1*(i+1) - np.pi/2)

    rho_y = Uy * result1.states[-1] * Uy.dag()
    rho_x = Ux * result1.states[-1] * Ux.dag()

    # ✅ ensure real
    Sigx_list.append(float(np.real(rho_x[1, 1] - rho_x[0, 0])))
    Sigy_list.append(float(np.real(rho_y[1, 1] - rho_y[0, 0])))

    p0.append(result.expect[3][-1])
    p1.append(result.expect[4][-1])
    p2.append(result.expect[5][-1])

    # gamma1 from population p1
    denom = float(p1[i]) if float(p1[i]) != 0 else 1e-12
    gamma1.append(-(float(p1[i+1]) - float(p1[i])) / (dt * denom))

    I1.append(-0.75 * Sigy_list[-1])

    states = result1.states[-1]

# build Vq_sim and signals on same time grid (160 points)
t = target_t
v_m = [I1[i] * gamma1[i] for i in range(measure_times - 1)]
signal = [np.sign(v_m[i] + bias) for i in range(measure_times - 1)]

# now define signal_fit using the fitted gamma + experimental I
signal_fit = [np.sign(v_test[i] + bias) for i in range(measure_times - 1)]

# =======================
# Plot
# =======================
mpl.rcParams.update({
    'font.size': 20,
    'axes.labelsize': 22,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 18,
    'axes.linewidth': 2,
    'grid.alpha': 0.3,
})

fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(10, 7), dpi=200, sharex=True,
    gridspec_kw={'hspace': 0, 'height_ratios': [1.5, 1]}
)

ax1.plot(
    t, v_m,
    color="#062BCE", linewidth=4, alpha=0.95, zorder=3,
    solid_capstyle='round', label='Simulation'
)
ax1.errorbar(
    t, v_test, yerr=v_err,
    fmt='o', color="#EE053F", markersize=6,
    markeredgecolor="#8A1830", markeredgewidth=2,
    capsize=6, capthick=2, elinewidth=1, zorder=4,
    label='Experiment'
)
ax1.axhline(y=-bias, color="#EE0B0B", linestyle='--', linewidth=4)

ax2.plot(
    t, signal,
    color="#062BCE", linewidth=4, alpha=0.95, zorder=3,
    solid_capstyle='round'
)
ax2.scatter(
    t, signal_fit,
    zorder=4, s=40, color="#F8083C", edgecolors="#8A1830", linewidths=2
)

ax2.set_xlabel('Time (μs)', fontweight='bold', labelpad=10)
ax1.set_ylabel(r'$\mathbf{V}_\mathbf{q}$', fontweight='bold', labelpad=7)

ax1.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
ax1.yaxis.set_major_locator(MaxNLocator(5))
ax1.yaxis.set_minor_locator(AutoMinorLocator(2))
ax1.ticklabel_format(style='sci', axis='y', scilimits=(-2, 2))
ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.6)
ax1.set_facecolor('#f8f9fa')
# ax1.legend(frameon=False, loc='upper right')

ax2.set_ylabel('Signal', fontweight='bold', labelpad=22)
ax2.yaxis.set_major_locator(FixedLocator([-1, 1]))
ax2.set_ylim(-1.1, 1.1)
ax2.grid(True, alpha=0.3, linestyle='-', linewidth=0.6)
ax2.set_facecolor('#f8f9fa')

ax1.spines['bottom'].set_visible(False)
ax1.tick_params(bottom=False, labelbottom=False)

ax2.xaxis.set_major_locator(MaxNLocator(6))
ax2.xaxis.set_minor_locator(AutoMinorLocator(3))

fig.patch.set_facecolor('white')
plt.tight_layout()
plt.show()

# =======================
# Save gamma + Vq_exp (optional)
# =======================
# out_dir = HERE
# pd.DataFrame([{
#     "C": C,
#     "gamma": gamma_fit_value,
#     "gamma_err": gamma_fit_err,
#     "R2": r_value**2
# }]).to_csv(os.path.join(out_dir, "single_fit_result.csv"), index=False, encoding="utf-8-sig")

# pd.DataFrame({
#     "Time_us": t,
#     "I_exp_interp": I_exp_interp,
#     "I_err_interp": I_err_interp,
#     "gamma_fit": gamma_fit,
#     "Vq_exp": v_test,
#     "Vq_exp_err": v_err
# }).to_csv(os.path.join(out_dir, "Vq_exp_from_fit.csv"), index=False, encoding="utf-8-sig")

# print(f"\nSaved:")
# print(f" - {os.path.join(out_dir, 'single_fit_result.csv')}")
# print(f" - {os.path.join(out_dir, 'Vq_exp_from_fit.csv')}")
