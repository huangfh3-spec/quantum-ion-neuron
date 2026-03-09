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
I_th=0.2
bias=0.001


gamma_T=bias/I_th
# =======================
# Paths (same folder)
# # =======================
# HERE = os.path.dirname(os.path.abspath(__file__))

# # 1) 布居数数据：无表头三列：Time_us, population, error
# POP_PATH = os.path.join(HERE, "spiking_population.csv")   
POP_PATH =  DATA /"spiking_population.csv"
# 2) Iq 实验数据：有表头 Time_us,I_exp,I_err
Iq_exp_path =  DATA /"I_q_exp.csv"
Iq_sim_path =  DATA /"I_q_sim.csv"
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
def read_Iq_sim(path: str):
    
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)

        # 情况1：有表头且包含 Time_us
        if "Time_us" in header:
            idx_t = header.index("Time_us")
            # 第二列作为I_sim，或者你也可以改成 header.index("I_sim")
            idx_i = 1 if len(header) > 1 else None
            if idx_i is None:
                raise ValueError("Error: No suitable column for I_sim found in the header.")
            t, I = [], []
            for row in reader:
                t.append(float(row[idx_t]))
                I.append(float(row[idx_i]))
            return np.array(t), np.array(I)

        # 情况2：两列无固定表头（回退）
        t, I = [], []
        for row in reader:
            if len(row) < 2:
                continue
            t.append(float(row[0]))
            I.append(float(row[1]))
        return np.array(t), np.array(I)


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
time_sim, I_sim = read_Iq_sim(Iq_sim_path)
V_sim =I_sim *gamma_T 

tlist2 = np.linspace(time_sim[0], time_sim[-1], 161)  # 160 points


target_t = tlist2[:160]  # 160 points
I_exp_interp = np.interp(target_t, time_exp, I_exp, left=I_exp[0], right=I_exp[-1])
I_err_interp = np.interp(target_t, time_exp, I_err, left=I_err[0], right=I_err[-1])

gamma_fit = np.full_like(target_t, fill_value=gamma_fit_value, dtype=float)

v_test = I_exp_interp * gamma_fit
v_err = np.abs(gamma_fit) * I_err_interp  






# =======================
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
    'axes.labelsize': 35,
    'xtick.labelsize': 30,
    'ytick.labelsize': 30,
    'legend.fontsize': 20,
    'figure.titlesize': 20,

    'axes.linewidth': 2,
    'grid.alpha': 0.3,
})

fig, ax = plt.subplots(figsize=(10, 5), dpi=200)

ax.plot(
    time_sim, V_sim,
    linewidth=2.5,
    alpha=0.95,
    zorder=1,
    color='red',
    solid_capstyle='round',
   
)

# 
ax.errorbar(
    target_t, v_test, yerr=v_err,
    fmt='o',
    color='red',
    mfc='white',
    mec='red',
    mew=2,
    ms=5,
    alpha=1,
    capsize=4,
    capthick=0.8,
    zorder=4,
    
)


def theory_envelope(t, gamma):
    t = np.asarray(t, dtype=float)
    gamma = float(gamma)
    return (3.0/8.0) * gamma * np.exp(-(3.0/4.0) * gamma * t)


env_sim = theory_envelope(time_sim, gamma_T)


env_exp = theory_envelope(target_t, gamma_fit_value)


ax.plot(
    time_sim, env_sim,
    linewidth=3.0,
    alpha=0.95,
    zorder=2,
    color='red',
    linestyle='-',
    
)

ax.plot(
    target_t, env_exp,
    linewidth=3.0,
    alpha=0.95,
    zorder=3,
    color='blue',
    linestyle='-',
    
)

ax.axhline(
    y=bias,
    linestyle='--',
    linewidth=4,
    label='bias',
    color='r',
)

ax.set_xlabel('Time(μs)', fontweight='bold', labelpad=5)
ax.set_ylabel(r'$\mathbf{V}_{\mathbf{q}}$', fontweight='bold', labelpad=5)

ax.grid(True, alpha=0.4, linestyle='-', linewidth=0.8)
ax.grid(True, which='minor', alpha=0.2, linestyle='--', linewidth=0.5)

ax.yaxis.set_major_locator(MaxNLocator(6))
ax.yaxis.set_minor_locator(AutoMinorLocator(2))
ax.ticklabel_format(style='sci', axis='y', scilimits=(-2, 2))

ax.yaxis.get_offset_text().set_fontfamily('Arial')
ax.yaxis.get_offset_text().set_fontweight('bold')

ax.legend(loc='upper right', prop={'weight': 'bold', 'family': 'Arial'}, frameon=False)

ax.set_facecolor('#f8f9fa')
fig.patch.set_facecolor('white')

ax.tick_params(axis='both', which='major', length=6, width=2)
ax.tick_params(axis='both', which='minor', length=4, width=1)
# plt.savefig(os.path.join(HERE, "Fig.2D.svg"), dpi=300, bbox_inches='tight')
plt.tight_layout()
plt.show()
