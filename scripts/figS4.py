import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, AutoMinorLocator
from qutip import *
import sys
from pathlib import Path

# 让 Python 能找到仓库根目录
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from paths import DATA
# =======================
# Matplotlib
# =======================
plt.rcParams.update({
    'font.size': 30,
    'axes.titlesize': 30,
    'axes.labelsize': 35,
    'xtick.labelsize': 30,
    'ytick.labelsize': 30,
    'legend.fontsize': 35,
    'figure.titlesize': 20,
    'font.family': 'serif',
    'font.serif': ['Arial', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'axes.linewidth': 2,
    'grid.alpha': 0.3
})
# =======================
# Path
# =======================
HERE = os.path.dirname(os.path.abspath(__file__))
POP_PATH = DATA / "rest_population.csv"

# =======================
# Read population CSV
# 无表头三列: time, population, error
# =======================
def read_population_noheader_3col(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Population CSV not found: {path}")

    raw = np.loadtxt(path, delimiter=",")

    if raw.ndim != 2 or raw.shape[1] < 3:
        raise ValueError("布居数文件必须是无表头三列: Time_us, population, error")

    t = raw[:, 0].astype(float)
    pop = raw[:, 1].astype(float)
    err = raw[:, 2].astype(float)

    order = np.argsort(t)
    return t[order], pop[order], err[order]

pop_t, pop_y, pop_err = read_population_noheader_3col(POP_PATH)

# =======================


dim = 7
Delta = -5.23 * 2 * np.pi
beta = 0
Gamma = 19.6 * 2 * np.pi

total_t = 160
measure_times = 161
tlist2 = np.linspace(0, total_t, measure_times)
dt = total_t / (measure_times - 1)

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
H_offset = (-Delta) * basis_states[6] * basis_states[6].dag() + (Delta) * basis_states[4] * basis_states[4].dag()
collapse_ops = [
    *[np.sqrt(Gamma / 3) * basis_states[k] * basis_states[5].dag() for k in [2, 1, 3]],
    *[np.sqrt(Gamma / 3) * basis_states[k] * basis_states[4].dag() for k in [1, 2, 0]],
    *[np.sqrt(Gamma / 3) * basis_states[k] * basis_states[6].dag() for k in [3, 0, 1]]
]
H_structure2 = (basis_states[2] * basis_states[4].dag() + basis_states[4] * basis_states[2].dag()
                    + basis_states[3] * basis_states[6].dag() + basis_states[6] * basis_states[3].dag())

def build_Hamiltonian_J2(J1, J2):
    return H_offset + J1 * H_structure1 + J2 * H_structure2
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
J2=[(2 * J1**2 * p1[0] * (Gamma**2 + 4*Delta**2) / ((1 - p0[0] - p1[0]) * Gamma**2))**0.5]
for i in range(measure_times - 1):
    H2 = build_Hamiltonian_J2(J1, J2[i])

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
  
    denom2 = (1 - p0[i+1] - p1[i+1])
    if denom2 <= 1e-12:
        denom2 = 1e-12
    J2_next = (2 * J1**2 * p1[i+1] * (Gamma**2 + 4*Delta**2) / (denom2 * Gamma**2))**0.5
    J2.append(float(J2_next))
    states = result1.states[-1]

# =======================
# Plot population
# =======================
mpl.rcParams.update({
    'font.size': 20,
    'axes.labelsize': 22,
    'legend.fontsize': 18,
    'axes.linewidth': 2
})

fig, ax = plt.subplots(figsize=(12, 7), dpi=200)

ax.plot(tlist2, p1, 
        color="#12C1E0",  # 专业橙色
        linewidth=7, 
        label='Simulation ', 
        alpha=0.95, 
        zorder=3,
        solid_capstyle='round')
# 绘制实验数据点（带误差条）
ax.errorbar(pop_t, pop_y, yerr=pop_err, 
            fmt='o', color="#EE053F", markersize=15,  markeredgecolor="#8A1830",markeredgewidth=4, 
            capsize=10, capthick=2, elinewidth=2,zorder=4,
            label='Experiment')
ax.set_xlabel('Time(μs)', fontweight='bold', labelpad=5)

# 设置坐标轴标签
ax.set_ylim([0.20, 0.30])  # 假设布居数在-1到1之间
ax.set_ylabel(r'$\mathbf{|1\rangle \ }$Population',fontweight='bold', labelpad=5)

# 设置合理的数据显示范围（根据实际数据调整）


# 设置图例
legend = ax.legend(
    loc='upper right',
    prop={'weight': 'bold'},
    frameon=True,
    fancybox=True,
    shadow=True,
    framealpha=0.95,
    edgecolor='gray',
    facecolor='white',
    borderpad=0.5,      # 边框与内容的距离，减小
    labelspacing=0.5,   # label之间的垂直距离，减小
    fontsize=25,        # 字体大小可适当减小
    handlelength=1.2,   # 图例线条长度
    handletextpad=0.5 )  # 线条和文字的距离
ax.yaxis.set_major_locator(MaxNLocator(6))  # 最多显示6个主要刻度
ax.yaxis.set_minor_locator(AutoMinorLocator(2))  # 每个主刻度间有2个次刻度   
ax.ticklabel_format(style='sci', axis='y', scilimits=(-2,2))
ax.grid(True, alpha=0.4, linestyle='-', linewidth=0.8)
ax.grid(True, which='minor', alpha=0.2, linestyle='--', linewidth=0.5)
ax.set_facecolor('#f8f9fa')
fig.patch.set_facecolor('white')
ax.tick_params(axis='both', which='major', length=6, width=2)
ax.tick_params(axis='both', which='minor', length=4, width=1)
# 确保布局紧凑
plt.tight_layout()
# plt.savefig('spiking_population.svg', format='svg', bbox_inches='tight')
# plt.savefig('spiking_voltage.svg', format='svg', bbox_inches='tight')
plt.show()

# 如需保存图片，放在 show() 前
# fig.savefig(os.path.join(HERE, "population_compare.svg"), bbox_inches='tight')
# fig.savefig(os.path.join(HERE, "population_compare.pdf"), bbox_inches='tight')