import os
import csv
import numpy as np
from qutip import *
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib.ticker import ScalarFormatter, MaxNLocator, AutoMinorLocator
import sys
from pathlib import Path

# 让 Python 能找到仓库根目录
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from paths import DATA
# ====================== 全局参数 ========================
measure_times = 66
tlist2 = np.linspace(0, 65, measure_times)

bias = -0.0016
startJ2 = 45
endJ2 = 65
dt = 1

dim = 7
Delta = 5.223 * 2 * np.pi
beta = 0
Gamma = 19.6 * 2 * np.pi
options = Options(nsteps=1500000)
basis_states = [basis(dim, i) for i in range(dim)]
I_th = 0.2
gamma_train = -bias / I_th
Dt = np.linspace(0, dt, 130)
delta1 = 1


# try:
#     HERE = os.path.dirname(os.path.abspath(__file__))
# except NameError:
#     HERE = os.getcwd()

# ===================== 算符定义 =====================
Sx = basis_states[1] * basis_states[0].dag() + basis_states[0] * basis_states[1].dag()
Sy = -1j * basis_states[1] * basis_states[0].dag() + 1j * basis_states[0] * basis_states[1].dag()
Sz = basis_states[1] * basis_states[1].dag() - basis_states[0] * basis_states[0].dag()
Iden = basis_states[1] * basis_states[1].dag() + basis_states[0] * basis_states[0].dag()
Iden_excited = (
    basis_states[2] * basis_states[2].dag()
    + basis_states[3] * basis_states[3].dag()
    + basis_states[4] * basis_states[4].dag()
    + basis_states[5] * basis_states[5].dag()
    + basis_states[6] * basis_states[6].dag()
)

def U(theta, phi):
    return (
        np.cos(theta / 2) * Iden
        - 1j * np.sin(theta / 2) * (np.cos(phi) * Sx + np.sin(phi) * Sy)
        + Iden_excited
    )

# ===================== 读取实验数据 =====================
def read_three_column_csv(path):
    """
    默认读取三列数据：x, y, yerr
    自动跳过表头
    """
    x, y, yerr = [], [], []
    with open(path, 'r', newline='') as file:
        reader = csv.reader(file)
        next(reader)  # 跳过标题行
        for row in reader:
            if len(row) < 3:
                continue
            x.append(float(row[0]))
            y.append(float(row[1]))
            yerr.append(float(row[2]))
    return np.array(x), np.array(y), np.array(yerr)

# 波形实验数据
def exp_waveform_data(filename="I_high_waveform.csv"):
    path = DATA / filename
    return read_three_column_csv(path)

# 布居数实验数据
def exp_population_data(filename="High Low data.csv"):
    path = DATA / filename
    return read_three_column_csv(path)

# ===================== 指数衰减拟合 =====================
def exp_decay_with_offset(t, gamma):
    """
    P(t) = A * exp(-gamma * t) + C
    gamma 即拟合得到的衰减系数
    """
    return 0.25 * np.exp(-gamma * t) 

def fit_population_decay(time_pop, pop, pop_err=None):
    """
    对布居数实验数据做指数衰减拟合，返回拟合参数和协方差矩阵
    """
    # 初值尽量稳一点
    # A0 = max(pop) - min(pop)
    gamma0 = 0.005
    # C0 = min(pop)

    p0 = [gamma0]

 
    
    popt, pcov = curve_fit(
            exp_decay_with_offset,
            time_pop,
            pop,
            p0=p0,
            maxfev=100000
        )
    return popt, pcov

# ====================== 初始态 ======================
a = 1
b = 1
c = 1
d = 1
theta = 0

psi0 = (
    a * basis_states[1]
    + np.exp(1j * theta) * b * basis_states[0]
    + c * basis_states[2]
    + d * basis_states[3]
).unit()

p0_0 = b**2 / (a**2 + b**2 + c**2 + d**2)
p1_0 = a**2 / (a**2 + b**2 + c**2 + d**2)
p2_0 = c**2 / (a**2 + b**2 + c**2 + d**2)
p3_0 = d**2 / (a**2 + b**2 + c**2 + d**2)

# ====================== 七能级系统定义 ====================
H_structure1 = basis_states[1] * basis_states[5].dag() + basis_states[5] * basis_states[1].dag()
H_structure2 = (
    basis_states[2] * basis_states[4].dag()
    + basis_states[4] * basis_states[2].dag()
    + basis_states[3] * basis_states[6].dag()
    + basis_states[6] * basis_states[3].dag()
)
H_offset = (
    
     (Delta) * basis_states[6] * basis_states[6].dag()
    + (-Delta) * basis_states[4] * basis_states[4].dag()
    
)
collapse_ops = [
    *[np.sqrt(Gamma / 3) * basis_states[k] * basis_states[5].dag() for k in [2, 1, 3]],
    *[np.sqrt(Gamma / 3) * basis_states[k] * basis_states[4].dag() for k in [1, 2, 0]],
    *[np.sqrt(Gamma / 3) * basis_states[k] * basis_states[6].dag() for k in [3, 0, 1]]
]


def build_Hamiltonian(J1, J2):
    return J1 * H_structure1 + J2 * H_structure2

# ====================== 动力学 ======================
I1_a = [-0.75 * expect(Sy, psi0)]
Sigx = [expect(Sx, psi0)]
Sigy = [expect(Sy, psi0)]
gamma1_a = []
states = psi0

p2_a = [p2_0]
p0_a = [p0_0]
p1_a = [p1_0]
p3_a = [p3_0]

J1_a = (3 * (Gamma) * 0.0072 / 8) ** 0.5
J2_a = [0]

sampling_factor = 4
high_res_t = []
v_a = []
signal_a = []

for i in range(measure_times - 1):
    t_measure = np.array([0.25 * dt, 0.5 * dt, 0.75 * dt, dt])

    H2 = build_Hamiltonian(J1_a, 0)

    result = mesolve(
        H2, states, t_measure, collapse_ops,
        [Sx, Sy, Sz,
         basis_states[0] * basis_states[0].dag(),
         basis_states[1] * basis_states[1].dag(),
         basis_states[2] * basis_states[2].dag()],
        options=options
    )

    result1 = mesolve(H2, states, [0, dt], collapse_ops, [], options=options)
    result2 = mesolve(H2, states, t_measure, collapse_ops, [], options=options)

    for j in range(sampling_factor):
        current_time = tlist2[i] + t_measure[j] if i > 0 else t_measure[j]
        high_res_t.append(current_time)

        Uy = U(np.pi / 2, -delta1 * (i + (j + 1) * 0.25))
        Ux = U(np.pi / 2, -delta1 * (i + (j + 1) * 0.25) - np.pi / 2)

        rho_y = Uy * result2.states[j] * Uy.dag()
        rho_x = Ux * result2.states[j] * Ux.dag()

        current_I1 = -0.75 * float(np.real(rho_y[1, 1] - rho_y[0, 0]))
        current_p1 = float(np.real(result.expect[4][j]))

        if j > 0:
            dt_high_res = t_measure[j] - t_measure[j - 1]
            p1_prev = float(np.real(result.expect[4][j - 1]))
            current_gamma1 = -(current_p1 - p1_prev) / (dt_high_res * p1_prev) if p1_prev != 0 else 0
        else:
            current_gamma1 = gamma1_a[-1] if gamma1_a else 0

        current_v = current_I1 * current_gamma1
        current_signal = np.sign(current_v + bias)

        v_a.append(current_v)
        signal_a.append(current_signal)

    p0_a.append(result.expect[3][-1])
    p1_a.append(result.expect[4][-1])
    p2_a.append(result.expect[5][-1])
    I1_a.append(-0.75 * result.expect[1][-1])

    gamma1_a.append(-(p1_a[i + 1] - p1_a[i]) / (dt * p1_a[i]) if p1_a[i] != 0 else 0)
    states = result1.states[-1]

# ====================== 读取实验数据并实时拟合 ======================
# 1) 读取 I_high 波形实验数据
time, I_high, I_sem = exp_waveform_data("I_high_waveform.csv")

# 2) 读取布居数实验数据
time_pop, pop_exp, pop_err = exp_population_data("High Low data.csv")

# 3) 指数衰减拟合，提取 gamma_fit
popt, pcov = fit_population_decay(time_pop, pop_exp, pop_err)
gamma_fit = popt[0]
# gamma_fit_err = np.sqrt(np.diag(pcov))[0] if pcov is not None else np.nan
gamma_fit_err = np.sqrt(pcov[0, 0])
print(f"Fitted decay coefficient gamma = {gamma_fit:.8f} ± {gamma_fit_err:.8f}")

# 4) 用实时拟合得到的 gamma_fit 代替原来手写的 0.006234
v_high = gamma_fit * I_high
v_err = gamma_fit * I_sem


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
# 创建图形和主图
fig = plt.figure(figsize=(16, 7), dpi=200)
gs = fig.add_gridspec(2, 2, width_ratios=[3, 1], height_ratios=[1, 0.4])
ax_main = fig.add_subplot(gs[:, 0])  # 主图占据左半部分

# 使用高分辨率数据绘图
line = ax_main.plot(high_res_t, v_a, 
                    color="#F0850A",
                    linewidth=4,
                    label='High γ  , Low G', 
                    alpha=0.95, 
                    zorder=3,
                    solid_capstyle='round')
ax_main.errorbar(time, v_high, yerr=v_err, 
            fmt='o', color="#EE053F", markersize=8,  markeredgecolor="#8A1830",markeredgewidth=2, 
            capsize=6, capthick=2, elinewidth=1,zorder=4,
            )
ax_main.axhline(y=-bias, color="#EE0B0B", linestyle='--', linewidth=4, label='Bias')

# 在主图中标记81-100μs的观察窗口（使用更显眼的颜色）
observation_window = ax_main.axvspan(startJ2, endJ2, alpha=0.5, color='grey', 
                                     edgecolor='grey', linewidth=2, linestyle='--',
                                     )

# 添加连接线到子图的指示
# 在观察窗口的上下边缘添加标记
# ax_main.axvline(x=81, ymin=0.1, ymax=0.9, color='red', linewidth=2, linestyle=':')
# ax_main.axvline(x=100, ymin=0.1, ymax=0.9, color='red', linewidth=2, linestyle=':')

# 添加观察窗口文本标注
# ax_main.text(90.5, ax_main.get_ylim()[0] + 0.05 * (ax_main.get_ylim()[1] - ax_main.get_ylim()[0]), 
#              '81-100 μs', fontsize=24, fontweight='bold', 
#              ha='center', va='bottom', color='red',
#              bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='red', alpha=0.9))

# 设置主图坐标轴和格式
ax_main.set_xlabel('Time(μs)', fontweight='bold', labelpad=5)
ax_main.set_ylabel('V', fontweight='bold', labelpad=5)

formatter = ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((-2, 2))
ax_main.yaxis.set_major_formatter(formatter)
ax_main.yaxis.set_major_locator(MaxNLocator(6))
ax_main.yaxis.set_minor_locator(AutoMinorLocator(2))
ax_main.ticklabel_format(style='sci', axis='y', scilimits=(-2,2))

ax_main.grid(True, alpha=0.4, linestyle='-', linewidth=0.8)
ax_main.grid(True, which='minor', alpha=0.2, linestyle='--', linewidth=0.5)
ax_main.set_facecolor('#f8f9fa')

ax_main.tick_params(axis='both', which='major', length=6, width=2)
ax_main.tick_params(axis='both', which='minor', length=4, width=1)

# 创建放大子图（放在右上角）
ax_inset = fig.add_subplot(gs[0, 1])  # 右上角

# 提取81-100μs范围内的数据
start_idx = next(i for i, t in enumerate(high_res_t) if t >= startJ2)
end_idx = next(i for i, t in enumerate(high_res_t) if t >= endJ2)
window_t = high_res_t[start_idx:end_idx]
window_va = v_a[start_idx:end_idx]

# 在放大子图中绘制
ax_inset.plot(window_t, window_va, color="#F0850A", linewidth=4, alpha=0.95)
ax_inset.axhline(y=-bias, color="#EE0B0B", linestyle='--', linewidth=3)
ax_inset.errorbar(time, v_high, yerr=v_err, 
            fmt='o', color="#EE053F", markersize=8,  markeredgecolor="#8A1830",markeredgewidth=2, 
            capsize=6, capthick=2, elinewidth=1,zorder=4,
            )
# 计算数据的y值范围，从0开始
y_min = 0
y_max = max(window_va) * 1.1  # 增加10%的余量

# 设置放大子图的纵坐标范围
ax_inset.set_ylim(y_min, y_max)
# 设置放大子图的样式
# ax_inset.set_title('Zoomed View (81-100μs)', fontsize=26, fontweight='bold', pad=10)
ax_inset.set_xlabel('Time(μs)', fontsize=22, fontweight='bold')
ax_inset.set_ylabel('V', fontsize=22, fontweight='bold')

# 为放大子图添加灰色背景框，使其与主图区分
ax_inset.set_facecolor('#f0f0f0')
for spine in ax_inset.spines.values():
    spine.set_linewidth(2)
    spine.set_color('gray')

# 设置放大子图的刻度
ax_inset.tick_params(axis='both', which='major', labelsize=20)
ax_inset.tick_params(axis='both', which='minor', labelsize=16)

# 添加网格
ax_inset.grid(True, alpha=0.4, linestyle='-', linewidth=0.8)

# 调整放大子图的x轴范围，使其正好显示81-100μs
ax_inset.set_xlim(startJ2, endJ2)

# 为放大子图添加刻度线指示器（连接主图和子图的视觉指示）
# 在主图中添加指向子图的箭头
import matplotlib.patches as patches

# 创建连接线
# 选择主图中观察窗口中间的一个点作为连接起点
# con1 = patches.ConnectionPatch(xyA=(90, ax_main.get_ylim()[0] + 0.7*(ax_main.get_ylim()[1]-ax_main.get_ylim()[0])), 
#                                xyB=(81, ax_inset.get_ylim()[1]*0.95), 
#                                coordsA="data", coordsB="data",
#                                axesA=ax_main, axesB=ax_inset,
#                                arrowstyle="->", shrinkA=5, shrinkB=5,
#                                mutation_scale=20, fc="red", ec="red", alpha=0.6)
# fig.add_artist(con1)

# 添加图例（只需要一个，放在主图的合适位置）
legend = ax_main.legend(
    loc='lower left',  # 调整位置避免与子图重叠
    prop={'weight': 'bold'},
    frameon=True,
    fancybox=True,
    shadow=True,
    framealpha=0.95,
    edgecolor='gray',
    facecolor='white',
    borderpad=0.5,
    labelspacing=0.5,
    fontsize=24,
    handlelength=1.2,
    handletextpad=0.5)

fig.patch.set_facecolor('white')
plt.tight_layout()

# import os

# HERE = os.path.dirname(os.path.abspath(__file__))
# save_path = os.path.join(HERE, "10_voltage.svg")

# plt.savefig(save_path, bbox_inches='tight')
plt.show()