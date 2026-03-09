import numpy as np
from qutip import *
import matplotlib.pyplot as plt
from scipy.signal import argrelmax, argrelmin
from matplotlib.ticker import ScalarFormatter, AutoMinorLocator, MaxNLocator
import matplotlib as mpl
# ====================== 全局参数 ========================
total_t=20
measure_times=21
tlist2 = np.linspace(0, total_t, measure_times)
dim = 7  # 七能级系统
Delta = -2.615 * 2 * np.pi  # 角频率

dt = total_t / (measure_times-1)
Gamma = 19.6 * 2 * np.pi  # 线频率
options = Options(nsteps=1500000)  # 增加积分步数
basis_states = [basis(dim, i) for i in range(dim)]

Dt=np.linspace(0,dt,100)
gamma_eff = [0]
delta=1
V = []
I = []
t = []

Sigx = [1]
Sigy = [0]

p_up = [1/2]
p_down = []
p2=[1/3]


I_th=0.2
bias=0.001


gamma_therory=bias/I_th

# 算子定义

Sx =  basis_states[1] * basis_states[0].dag() + basis_states[0] * basis_states[1].dag()
Sy =- 1j * basis_states[1] * basis_states[0].dag() + 1j * basis_states[0] * basis_states[1].dag()
Sz =  basis_states[1] * basis_states[1].dag() -basis_states[0] * basis_states[0].dag()
J1 = [0]
Iden= basis_states[1] * basis_states[1].dag() + basis_states[0] * basis_states[0].dag()
Iden_excited=basis_states[2] * basis_states[2].dag() + basis_states[3] * basis_states[3].dag()+basis_states[4] * basis_states[4].dag()+basis_states[5] * basis_states[5].dag()+basis_states[6] * basis_states[6].dag()




# ==========================================


H_structure = (basis_states[1] * basis_states[5].dag() + basis_states[5] * basis_states[1].dag())


collapse_ops = [
    *[np.sqrt(Gamma / 3) * basis_states[k] * basis_states[5].dag() for k in [2, 1, 3]],
    *[np.sqrt(Gamma / 3) * basis_states[k] * basis_states[4].dag() for k in [1, 2, 0]],
    *[np.sqrt(Gamma / 3) * basis_states[k] * basis_states[6].dag() for k in [3, 0, 1]]
]
def U(theta , phi):
    return (np.cos(theta/2)*Iden-1j*np.sin(theta/2)*(np.cos(phi)*Sx+np.sin(phi)*Sy)+Iden_excited)

def build_Hamiltonian(J):
    return  J * H_structure

states = (basis_states[1] + basis_states[0]).unit()

for i in range(measure_times-1):    #增加归一化
    H2 = build_Hamiltonian(J1[i])
    result = mesolve(
        H2,
        states,
        Dt,
        collapse_ops,
        [
            Sx,
            Sy,
            Sz,
            basis_states[0] * basis_states[0].dag(),
            basis_states[1] * basis_states[1].dag(),
           
           
        ],
        options=options,
    )
    result1 = mesolve(H2, states, Dt, collapse_ops, [], options=options)
    
    

    # #微波脉冲测量Sy，Sx
    rho_y= U(np.pi/2,-delta*4*(i+1))*result1.states[-1]*U(np.pi/2,-delta*4*(i+1)).dag()

    rho_x= U(np.pi/2,-delta*4*(i+1)-np.pi/2)*result1.states[-1]*U(np.pi/2,-delta*4*(i+1)-np.pi/2).dag()



    Sigx.append(float(np.real(rho_x[1,1] - rho_x[0,0])))
    Sigy.append(float(np.real(rho_y[1,1] - rho_y[0,0])))


    
    
    
    I.append(-0.75 * Sigy[i])
   

    y_true=np.sign(I[i]-I_th)
    y_pred=np.sign(gamma_eff[i]*I[i]-bias)
    error=y_true-y_pred   
    
    gamma_eff.append(gamma_eff[i] + error * I[i] * dt*0.0021) 
    
    

    J1.append(
        (3*(Gamma)*gamma_eff[i+1]/8)**0.5

        
    )
    
    
    states=result1.states[-1]


gamma_train=gamma_eff[:measure_times]





t=tlist2[:(measure_times-1)]


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
    'legend.fontsize': 20,
    'figure.titlesize': 20,

    'axes.linewidth': 2,
    'grid.alpha': 0.3,
})

# 创建图形和坐标轴
fig, ax = plt.subplots(figsize=(10, 5), dpi=200)


# 4. 模拟数据连线 - 使用对比色，不画点
sim_line = ax.plot(tlist2,gamma_train, 
                    color="#0870D1",  # 专业橙色
                    linewidth=3.5, 
                    label='Simulation ', 
                    alpha=0.95, 
                    zorder=3,
                    solid_capstyle='round')

# plt.subplot(4,1,2)

ax.axhline(y=gamma_therory, color="#F16406",
            linestyle='--',
            linewidth=3.5, 
            label=r'Theoretical $\mathbf{\gamma}$')
# plt.plot(t, gamma_e, 'b--', label=r'gamma_e ')


final_gamma=gamma_train[-1]

final_error=[]
# 7. 纵坐标使用科学计数法并增大分度值
# 设置科学计数法格式
formatter = ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((-2, 2))
ax.yaxis.set_major_formatter(formatter)

# 增大分度值 - 调整y轴刻度间隔
ax.yaxis.set_major_locator(MaxNLocator(6))  # 最多显示6个主要刻度
ax.yaxis.set_minor_locator(AutoMinorLocator(2))  # 每个主刻度间有2个次刻度
ax.ticklabel_format(style='sci', axis='y', scilimits=(-2,2))
ax.xaxis.set_major_locator(MaxNLocator(4))
ax.xaxis.set_minor_locator(AutoMinorLocator(2))
# 6. 坐标轴标签和标题
ax.set_xlabel('Time(μs)', fontweight='bold', labelpad=5, fontname='Arial')
ax.set_ylabel(r'$\mathbf{\gamma}_\mathbf{eff}$(MHz)', fontweight='bold', labelpad=5, fontname='Arial')
# ax.set_title(r'$\mathbf{\gamma}$ Evolution', 
#             fontweight='bold', 
#             pad=5,
#             fontsize=30,
#             fontname='Arial')

for label in ax.get_xticklabels():
    label.set_fontname('Arial')
for label in ax.get_yticklabels():
    label.set_fontname('Arial')

# 8. 网格设置
ax.grid(True, alpha=0.4, linestyle='-', linewidth=0.8)
ax.grid(True, which='minor', alpha=0.2, linestyle='--', linewidth=0.5)

# 9. 图例美化
# legend = ax.legend(loc='upper right', 
#                 prop={'weight': 'bold', 'family': 'Arial'},
#                 frameon=True, 
#                 fancybox=True, 
#                 shadow=True, 
#                 framealpha=0.95,
#                 edgecolor='gray',
#                 facecolor='white',
#                 borderpad=1,
#                 labelspacing=0.7)
    
# 11. 设置背景色
ax.set_facecolor('#f8f9fa')
fig.patch.set_facecolor('white')

# 12. 调整刻度参数
ax.tick_params(axis='both', which='major', length=6, width=2)
ax.tick_params(axis='both', which='minor', length=4, width=1)
plt.tight_layout()
# plt.savefig('Train gamma process.svg', format='svg', bbox_inches='tight')
plt.show()
