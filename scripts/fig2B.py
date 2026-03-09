#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import os
import numpy as np
from qutip import *
from scipy.signal import argrelmax, argrelmin
from matplotlib.ticker import MaxNLocator, AutoMinorLocator
import matplotlib as mpl
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


# 让 Python 找到仓库根目录
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from paths import DATA

# ====================== 全局参数 ========================
dim = 7  # 七能级系统
total_t = 160
delta = 0.040
Delta = -2.615 * 2 * np.pi  # 角频率
dt =step=4
N=total_t//dt
measure_times =int(total_t // dt + 1)
print(measure_times)
Gamma = 19.6 * 2 * np.pi  # 线频率
options = Options(nsteps=1500000)  # 增加积分步数
basis_states = [basis(dim, i) for i in range(dim)]
Dt = np.linspace(0, dt, 80)
tlist2 = np.linspace(0, total_t, measure_times)
gamma_real1 = [0]

beta = 0

V1 = []
I1 = []

V2 = []
I2 = []

V3 = []
I3 = []
t = []

Sx1 = [1]
Sy1 = [0]


Sx11 = []
Sy11 = []



P0 = [1/2]
P1 = [1/2]
P2 = [0]
P3 = [0]


# 量子比特算子定义
Sigx = basis_states[1] * basis_states[0].dag() + basis_states[0] * basis_states[1].dag()
Sigy = -1j * basis_states[1] * basis_states[0].dag() + 1j * basis_states[0] * basis_states[1].dag()
Sigz = basis_states[1] * basis_states[1].dag() - basis_states[0] * basis_states[0].dag()
Iden= basis_states[1] * basis_states[1].dag() + basis_states[0] * basis_states[0].dag()
Iden_excited=basis_states[2] * basis_states[2].dag() + basis_states[3] * basis_states[3].dag()+basis_states[4] * basis_states[4].dag()+basis_states[5] * basis_states[5].dag()+basis_states[6] * basis_states[6].dag()

#微波旋转算符定义  (theta 旋转角度  phi 旋转轴)
def U(theta , phi):
    return (np.cos(theta/2)*Iden-1j*np.sin(theta/2)*(np.cos(phi)*Sigx+np.sin(phi)*Sigy)+Iden_excited)



# ====================== 七能级系统定义 ====================


H_structure1 = basis_states[1] * basis_states[5].dag() + basis_states[5] * basis_states[1].dag()

H_structure2 = (
    basis_states[2] * basis_states[4].dag()
    + basis_states[4] * basis_states[2].dag()
    + basis_states[3] * basis_states[6].dag()
    + basis_states[6] * basis_states[3].dag()
)

J1 = [0]

# 耗散项 (示例，根据实际情况调整)
collapse_ops = [
    *[
        np.sqrt(Gamma / 3) * basis_states[k] * basis_states[5].dag()
        for k in [2, 1, 3]
    ],
    *[
        np.sqrt(Gamma / 3) * basis_states[k] * basis_states[4].dag()
        for k in [1, 2, 0]
    ],
    *[
        np.sqrt(Gamma / 3) * basis_states[k] * basis_states[6].dag()
        for k in [3, 0, 1]
    ],
]


def build_Hamiltonian(J1_val):
    return   J1_val * H_structure1 
tomo_x=[1]
tomo_y=[0.5]
J2=[]
states = (basis_states[1]+basis_states[0] ).unit()

for i in range(measure_times - 1): 
      #i 从0-159
    H2 = build_Hamiltonian(J1[i])
    result = mesolve(
        H2,
        states,
        Dt,
        collapse_ops,
        [
            Sigx,
            Sigy,
            Sigz,
            basis_states[0] * basis_states[0].dag(),
            basis_states[1] * basis_states[1].dag(),
            basis_states[2] * basis_states[2].dag(),
            basis_states[3] * basis_states[3].dag()
           
        ],
        options=options,
    )
    result1 = mesolve(H2, states, Dt, collapse_ops, [], options=options)
    


    # #微波脉冲测量Sy，Sx
    rho_y= U(np.pi/2,-delta*4*(i+1))*result1.states[-1]*U(np.pi/2,-delta*4*(i+1)).dag()

    rho_x= U(np.pi/2,-delta*4*(i+1)-np.pi/2)*result1.states[-1]*U(np.pi/2,-delta*4*(i+1)-np.pi/2).dag()



    Sx1.append(rho_x[1,1]-rho_x[0,0])
    Sy1.append(rho_y[1,1]-rho_y[0,0])
       #/ (result.expect[3][-1] + result.expect[4][-1])

    # tomo_y.append((result.expect[1][-1]+1)/2)
    # tomo_x.append((result.expect[0][-1]+1)/2)


    
    P0.append(result.expect[3][-1])
    P1.append(result.expect[4][-1])
    P2.append(result.expect[5][-1])
    P3.append(result.expect[6][-1])

    gamma_real1.append(gamma_real1[i] +0.5 * Sy1[i] * dt * 0.0004)

    J1.append(

         (3*(Gamma)*gamma_real1[i+1]/8)**0.5)
    
    
    



    states = result1.states[-1]
v_m=[]
# 截断到 measure_times 长度
t=[]   #160长度
t2=tlist2[:measure_times-1]
# J1 = J1[:measure_times-1]
# gamma_real1 = gamma_real1[:measure_times]
dy_dt=[]
I_m=[]
# 计算 I1, V1, aomrf, t
print(len(Sx1))
sigmax=Sx1
sigmay=Sy1


for i in range(len(Sx1) - 1):
    I_m.append(-0.75 * Sy1[i])
    # V1.append(((Sy1[i + 1] - Sy1[i]) / dt - delta* Sx1[i]))
    v_m.append(I_m[i]*gamma_real1[i])
    dy_dt.append((Sy1[i + 1] - Sy1[i]) / dt)
    J2.append(beta*J1[i])
    t.append(tlist2[i])
Sx1 = Sx1[:N]
Sy1 = Sy1[:N]
t=tlist2[:N]
dy_dt_sim = (np.array(Sy1[2:]) - np.array(Sy1[:-2])) / (2*dt)
Sy1 = Sy1[1:N-1]
Sx1 = Sx1[1:N-1]
t=tlist2[1:N-1]
V1 = dy_dt_sim - 0.04 * np.array(Sx1)
I1 = -0.75 * np.array(Sy1)



# import os

# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Sx_path = os.path.join(BASE_DIR, "data", "S_x.csv")
# Sy_path = os.path.join(BASE_DIR, "data", "S_y.csv")


sx_path = DATA / "S_x.csv"
sy_path = DATA / "S_y.csv"

# sx = pd.read_csv(sx_path)
# sy = pd.read_csv(sy_path)
# 读取 CSV
df = pd.read_csv(sx_path, header=None)

# 提取两列
Sx_exp  = df.iloc[:, 0].values   # 第一列
Sx_err = df.iloc[:, 1].values   # 第二列


df = pd.read_csv(sy_path, header=None)

# 提取两列
Sy_exp  = df.iloc[:, 0].values   # 第一列
Sy_err = df.iloc[:, 1].values   # 第二列


leng=len(Sx_exp)
Sx_exp=Sx_exp[1:(leng-1)]
Sx_err=Sx_err[1:(leng-1)]


#V_q and I_q from experiment
dy_dt = (Sy_exp[2:] - Sy_exp[:-2]) / (2*step)

dy_dt_sem=(Sy_err[2:]**2 + Sy_err[:-2]**2)**0.5/(2*step)
Sy_err=Sy_err[1:(leng-1)]
I_sem=0.75*Sy_err
V_sem=np.sqrt(dy_dt_sem**2+(0.04*Sx_err)**2)*0.5

Sy_exp=Sy_exp[1:(leng-1)]
V = dy_dt - 0.04 * Sx_exp
I = -0.75 * Sy_exp



import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from matplotlib.ticker import ScalarFormatter, AutoMinorLocator, MaxNLocator

# 设置全局绘图参数 - 使用Arial字体
plt.rcParams['font.family'] = 'Arial'
mpl.rcParams['font.family'] = 'Arial'
plt.rcParams.update({
    'font.size': 15,
    'axes.titlesize': 30,
    'axes.labelsize': 25,
    'xtick.labelsize': 25,
    'ytick.labelsize': 25,
    'legend.fontsize': 20,
    'figure.titlesize': 20,
    'font.family': 'Arial',  # 改为Arial字体
    'mathtext.fontset': 'custom',  # 自定义数学字体
    'mathtext.it': 'Arial:italic',  # 数学斜体使用Arial
    'mathtext.bf': 'Arial:bold',    # 数学粗体使用Arial
    'mathtext.rm': 'Arial',         # 数学常规使用Arial
    'axes.linewidth': 2,
    'grid.alpha': 0.3
})

def create_enhanced_iv_plot(I,V, I_sem, V_sem, I1, V1, save_path='I-V_enhanced.svg'):


    # 创建图形和坐标轴
    fig, ax = plt.subplots(figsize=(10, 6), dpi=200)

    # ==================== 实验数据绘图 ====================

    error_plot = ax.errorbar(I, V,
                            yerr=V_sem,
                            fmt='o',
                            markersize=9,
                            markerfacecolor="#220DE4",  
                            markeredgecolor='black',
                            markeredgewidth=2,
                            color='#2E86AB',
                            ecolor="#220DE4",  
                            elinewidth=2,
                            capsize=5,
                            capthick=2,
                            alpha=0.85,
                            label='Experimental data',
                            zorder=4)

    sim_line = ax.plot(I1, V1,
                      color='#F18F01',
                      linewidth=3.5,
                      label='Simulation',
                      alpha=0.95,
                      zorder=3,
                      solid_capstyle='round')

    ax.plot([I1[-1], I1[0]], [V1[-1], V1[0]],
           color='#F18F01',
           linewidth=3.5,
           alpha=0.95,
           zorder=3)

    # ==================== 图表美化 ====================

    # 6. 坐标轴标签和标题
    ax.set_xlabel(r'$\mathbf{I_q}$', fontweight='bold', fontname='Arial', labelpad=5, color='black')
    ax.set_ylabel(r'$\mathbf{V_q}$', fontweight='bold', fontname='Arial', labelpad=5, color='black')


    # 7. 纵坐标使用科学计数法并增大分度值
    # 设置科学计数法格式
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-2, 2))
    ax.yaxis.set_major_formatter(formatter)

    # 增大分度值 - 调整y轴刻度间隔
    ax.yaxis.set_major_locator(MaxNLocator(6))  # 最多显示6个主要刻度
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))  # 每个主刻度间有2个次刻度

    # 8. 网格设置
    ax.grid(True, alpha=0.4, linestyle='-', linewidth=0.8)
    ax.grid(True, which='minor', alpha=0.2, linestyle='--', linewidth=0.5)

    # 9. 设置背景色
    ax.set_facecolor('#f8f9fa')
    fig.patch.set_facecolor('white')

    # 10. 调整刻度参数
    ax.tick_params(axis='both', which='major', length=6, width=2, color='black')
    ax.tick_params(axis='both', which='minor', length=4, width=1, color='black')

    # 11. 设置刻度标签字体为Arial
    for label in ax.get_xticklabels():
        label.set_fontname('Arial')
        label.set_color('black')
    for label in ax.get_yticklabels():
        label.set_fontname('Arial')
        label.set_color('black')

    # 12. 确保科学计数法标签显示完整
    ax.ticklabel_format(style='sci', axis='y', scilimits=(-2, 2))

    for text in ax.yaxis.get_majorticklabels():
        text.set_fontname('Arial')
        text.set_color('black')

    # 恢复图表黑边（坐标轴边框）
    ax.spines['left'].set_visible(True)
    ax.spines['left'].set_color('black')
    ax.spines['left'].set_linewidth(2)

    ax.spines['bottom'].set_visible(True)
    ax.spines['bottom'].set_color('black')
    ax.spines['bottom'].set_linewidth(2)

    ax.spines['right'].set_visible(True)
    ax.spines['right'].set_color('black')
    ax.spines['right'].set_linewidth(2)

    ax.spines['top'].set_visible(True)
    ax.spines['top'].set_color('black')
    ax.spines['top'].set_linewidth(2)


    # 调整布局
    plt.tight_layout()


    plt.show()

    return fig, ax

# 主程序
if __name__ == "__main__":


   
    fig, ax = create_enhanced_iv_plot(I, V, I_sem, V_sem, I1, V1)







