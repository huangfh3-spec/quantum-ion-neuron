import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter

# ======================================================
# 全局字体与风格：全部 Arial
# ======================================================
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial'],
    'mathtext.fontset': 'custom',
    'mathtext.rm': 'Arial',
    'mathtext.it': 'Arial:italic',
    'mathtext.bf': 'Arial:bold',

    'font.size': 26,
    'axes.titlesize': 28,
    'axes.labelsize': 35,
    'xtick.labelsize': 26,
    'ytick.labelsize': 26,
    'legend.fontsize': 24,
    'axes.linewidth': 2,
})
measure_times=260
tlist2=np.linspace(0,65,measure_times)

start=45
end=65
dt=1
delta1=1
# 1) 固定仿真时间轴与读出窗口

win = (start, end)  # 读出窗口 (μs)
win_mask = (tlist2 >= win[0]) & (tlist2 <= win[1])
b = 1.6e-3  # XOR里用的bias

# 2) 扫描范围（用 gamma_T 归一化）
gammaT = 0.005 # 归一化; 最终你用真实 gamma_T_eff 替换
gamma_grid = np.linspace(1*gammaT, 2*gammaT, 100)
G_grid     = np.linspace(0.0*gammaT, 3*gammaT,100)

C = np.zeros((len(G_grid), len(gamma_grid)), dtype=int)
D = np.zeros((len(G_grid), len(gamma_grid)), dtype=float)
def simulate_Vq(gamma,G_eff):  
    v=-3/4*1/2*np.sin(delta1*tlist2)*np.exp(-3*gamma/4*tlist2)*(gamma-G_eff)

    return v


for i, G in enumerate(G_grid):
    for j, gamma in enumerate(gamma_grid):
        Vq = simulate_Vq(gamma, G)
        vmax = Vq[win_mask].max()
        D[i, j] = vmax - b
        C[i, j] = 1 if vmax >= b else -1    




# ------------------ 归一化坐标 ------------------
gamma_n = gamma_grid / gammaT
G_n     = G_grid / gammaT
X, Y = np.meshgrid(gamma_n, G_n, indexing="xy")

# ------------------ 固定 G = 0.5 γT ------------------
G_fixed_norm = 2.3
iG = np.argmin(np.abs(G_n - G_fixed_norm))
D_1d = D[iG, :]

# ------------------ 科学计数格式 ------------------
sci_formatter = ScalarFormatter(useMathText=True)
sci_formatter.set_powerlimits((-2, 2))

# ======================================================
# Figure with subplots
# ======================================================
fig, axes = plt.subplots(
    1, 2,
    figsize=(20, 8),
    gridspec_kw={'width_ratios': [1.3, 1]}
)

# ======================================================
# (a) 二维决策图
# ======================================================
ax = axes[0]

pcm = ax.pcolormesh(X, Y, D, shading="auto", cmap="viridis")

ax.contour(
    gamma_n, G_n, D,
    levels=[0],
    colors="b",
    linewidths=5
)

# 决策边界图例代理
boundary_proxy = Line2D(
    [0], [0], color='b', lw=5,
    label=r"Decision boundary ($\kappa=0$)"
)

# 四个实验点（归一化坐标）
points = {
    "A": (1.44, 0.0),
    "B": (1.44, 2.3),
    "C": (1.0, 0.0),
    "D": (1.0, 2.3),
}

for label, (gx, Gy) in points.items():
    ax.scatter(
        gx, Gy,
        s=140,
        c="red",
        edgecolors="black",
        linewidths=2,
        zorder=5
    )
    ax.text(
        gx + 0.03, Gy + 0.05,
        label,
        fontsize=24,
        fontweight="bold"
    )

# 坐标轴（直立体 + 加粗）
ax.set_xlabel(r"$\mathrm{\gamma_{eff}/\gamma_T}$", fontweight="bold")
ax.set_ylabel(r"$\mathrm{G_{eff}/\gamma_T}$", fontweight="bold")
# ax.set_title("(a) 2D decision map")

# 刻度加粗
ax.tick_params(axis='both', which='major', width=2, length=8)
for tick in ax.get_xticklabels() + ax.get_yticklabels():
    tick.set_fontweight('bold')

# colorbar（科学计数）
cbar = fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label(r"$\kappa$",
               rotation=270, labelpad=30,
               fontweight="bold")
cbar.formatter = sci_formatter
cbar.update_ticks()

ax.legend(handles=[boundary_proxy], loc="upper right", frameon=True)

# ======================================================
# (b) 固定 G 的一维激活图
# ======================================================
ax = axes[1]

ax.plot(
    gamma_n, D_1d,
    lw=4,
    color="black",
    label=rf"$\mathrm{{G/\gamma_T={G_n[iG]:.2f}}}$"
)

ax.axhline(
    0,
    ls="--",
    lw=3,
    color="b",
    label=r"Threshold ($\kappa=0$)"
)

# XOR 中使用的 γ 点
ax.scatter(
    [1.0, 1.44],
    [
        D_1d[np.argmin(np.abs(gamma_n - 1.0))],
        D_1d[np.argmin(np.abs(gamma_n - 1.44))]
    ],
    s=120,
    c="red",
    zorder=5
)

ax.set_xlabel(r"$\mathrm{\gamma_{eff}/\gamma_T}$", fontweight="bold")
ax.set_ylabel(r"$\mathrm{\kappa}$", fontweight="bold")
# ax.set_title(r"(b) 1D activation (fixed $\mathrm{G/\gamma_T=0.5}$)")

# y 轴科学计数
ax.yaxis.set_major_formatter(sci_formatter)

# 刻度加粗
ax.tick_params(axis='both', which='major', width=2, length=8)
for tick in ax.get_xticklabels() + ax.get_yticklabels():
    tick.set_fontweight('bold')

ax.legend(frameon=True)
ax.grid(alpha=0.3)

# ======================================================
# 输出
# ======================================================
plt.tight_layout()
# import os

# HERE = os.path.dirname(os.path.abspath(__file__))
# save_path = os.path.join(HERE, "Fig4A_B.svg")

# plt.savefig(save_path, bbox_inches='tight')
plt.show()
