import numpy as np
from qutip import *
import matplotlib.pyplot as plt
from scipy.signal import argrelmax, argrelmin
from matplotlib.ticker import ScalarFormatter, AutoMinorLocator, MaxNLocator
from matplotlib.ticker import FixedLocator
# ====================== 全局参数 ========================
total_t=160
measure_times=161
tlist2 = np.linspace(0, total_t, measure_times)
dim = 7  # 七能级系统
Delta = -5.23 * 2 * np.pi  # 角频率
beta=0
dt = total_t / (measure_times-1)
Gamma = 19.6 * 2 * np.pi  # 线频率
options = Options(nsteps=1500000)  # 增加积分步数
basis_states = [basis(dim, i) for i in range(dim)]
I_th=0.2
bias=-0.001
###################算符定义####################
Sx =  basis_states[1] * basis_states[0].dag() + basis_states[0] * basis_states[1].dag()
Sy =- 1j * basis_states[1] * basis_states[0].dag() + 1j * basis_states[0] * basis_states[1].dag()
Sz =basis_states[1] * basis_states[1].dag() - basis_states[0] * basis_states[0].dag()
Iden= basis_states[1] * basis_states[1].dag() + basis_states[0] * basis_states[0].dag()
Iden_excited=basis_states[2] * basis_states[2].dag() + basis_states[3] * basis_states[3].dag()+basis_states[4] * basis_states[4].dag()+basis_states[5] * basis_states[5].dag()+basis_states[6] * basis_states[6].dag()

#微波旋转算符定义  (theta 旋转角度  phi 旋转轴)
def U(theta , phi):
    return (np.cos(theta/2)*Iden-1j*np.sin(theta/2)*(np.cos(phi)*Sx+np.sin(phi)*Sy)+Iden_excited)

gamma_train=-bias/I_th
Dt = np.linspace(0, dt, 130)
delta1=1
delta2=0
delta3=0

#####################定义初始状态#####################
a=1
b=1
c=1
d=1

theta=0

psi0= (a*basis_states[1] + b*basis_states[0]+c*basis_states[2]+d*basis_states[3]).unit()

p0_0=b**2/(a**2+b**2+c**2+d**2)
p1_0=a**2/(a**2+b**2+c**2+d**2)

p2_0=c**2/(a**2+b**2+c**2+d**2)
p3_0=d**2/(a**2+b**2+c**2+d**2)

# ====================== 七能级系统定义 ====================
H_offset = (
    
    + (-Delta) * basis_states[6] * basis_states[6].dag()
    + (Delta) * basis_states[4] * basis_states[4].dag()
    
)

H_structure1 = basis_states[1] * basis_states[5].dag() + basis_states[5] * basis_states[1].dag()
H_structure2 = (
    basis_states[2] * basis_states[4].dag()
    + basis_states[4] * basis_states[2].dag()
    + basis_states[3] * basis_states[6].dag()
    + basis_states[6] * basis_states[3].dag()
)


# 耗散项 (示例，根据实际情况调整)
collapse_ops = [
    *[np.sqrt(Gamma / 3) * basis_states[k] * basis_states[5].dag() for k in [2, 1, 3]],
    *[np.sqrt(Gamma / 3) * basis_states[k] * basis_states[4].dag() for k in [1, 2, 0]],
    *[np.sqrt(Gamma / 3) * basis_states[k] * basis_states[6].dag() for k in [3, 0, 1]]
]

def build_Hamiltonian_J2(J1,J2):
    return H_offset + J1 *H_structure1+J2*H_structure2                                                                                         


def build_Hamiltonian(J1):
    return  J1 *H_structure1
Sigx = [expect(Sx, psi0)]
Sigy = [expect(Sy, psi0)]

#常规模式

import csv
import pandas as pd
##################列表初始化#######################
p2=[p2_0]
p0=[p0_0]
p1=[p1_0]
p3=[p3_0]
V1 = []
I1 = [-0.75*expect(Sy, psi0)]


Sigx = [expect(Sx, psi0)]
Sigy = [expect(Sy, psi0)]


gamma2=[]
gamma1=[]


states=psi0
high_res_t=[]
v_spiking=[]
signal_spiking=[]
gamma1=[]
sampling_factor=4  # 高采样率因子

#(3 * gamma_train * (1/3) / (8 * Gamma *( (1/3) / (Gamma**2 + delta1**2) -beta**2*(1/3)/(Gamma**2 + 4*(Delta+delta2)**2))))**0.5
J1=[(3*(Gamma)*gamma_train/8)**0.5]   #第一段时间内已经训练好的gamma1
J2=[beta*J1[0]]
def compare(I):
    if I>0:
        return(1)
    else:
        return(0)
##########################################################
for i in range(measure_times-1):
   

    

    # Sigy2.append(result.expect[7][-1]/(result.expect[5][-1]+result.expect[4][-1]))  # 每次8us测量一次
    # Sigx2.append(result.expect[6][-1]/(result.expect[5][-1]+result.expect[4][-1]))
    # Sigy3.append(result.expect[9][-1]/(result.expect[8][-1]+result.expect[4][-1]))  # 每次8us测量一次



    t_measure = np.array([
    0,
    0.25 * dt,
    0.5 * dt,
    0.75 * dt 
    ])
    
    H2 = build_Hamiltonian_J2(J1[i],J2[i])
    
    # 高采样率测量
    result = mesolve(H2, states, Dt, collapse_ops, 
                    [Sx, Sy, Sz, basis_states[0] * basis_states[0].dag(), 
                     basis_states[1] * basis_states[1].dag(),
                     basis_states[2] * basis_states[2].dag()], 
                    options=options)
    
    # 状态演化（保持原有步长）
    result1 = mesolve(H2, states, Dt, collapse_ops, [], options=options)
    
    rho_y= U(np.pi/2,-delta1*(i+1))*result1.states[-1]*U(np.pi/2,-delta1*(i+1)).dag()

    rho_x= U(np.pi/2,-delta1*(i+1)-np.pi/2)*result1.states[-1]*U(np.pi/2,-delta1*(i+1)-np.pi/2).dag()



    Sigx.append(rho_x[1,1]-rho_x[0,0])
    Sigy.append(rho_y[1,1]-rho_y[0,0])
    # 处理高采样率数据
    for j in range(sampling_factor):
        current_time = tlist2[i] + t_measure[j] if i > 0 else t_measure[j]
        high_res_t.append(current_time)
        
        # 计算高采样率的物理量
        current_I1 = -0.75 * result.expect[1][j]
        current_p1 = result.expect[4][j]
        
        # 计算gamma1（需要小心处理，因为这是导数）
        if j > 0:
            dt_high_res = t_measure[j] - t_measure[j-1]
            p1_prev = result.expect[4][j-1]
            current_gamma1 = -(current_p1 - p1_prev) / (dt_high_res * p1_prev) if p1_prev != 0 else 0
        else:
            # 对于第一个点，使用上一个时间步的值或设为0
            current_gamma1 = gamma1[-1] if gamma1 else 0
        
        current_v = current_I1 * current_gamma1
        current_signal = compare(current_v + bias) 
        
        v_spiking.append(current_v)
        signal_spiking.append(current_signal)

    p0.append(result.expect[3][-1])
    p1.append(result.expect[4][-1])
    p2.append(result.expect[5][-1])
        
    
    

    # Sigy.append(result.expect[1][-1])  # 每次8us测量一次
    # Sigx.append(result.expect[0][-1])
    # gamma1.append(-(p0[i+1]-p0[i])/(dt*p0[i]))
    # gamma2.append(-(p2[i+1]-p2[i])/(dt*p2[i]))  
    # J1.append(
    #         (3 * gamma_train * p0[i+1] / (8 * Gamma *( p0[i+1] / (Gamma**2 + delta1**2) -beta**2*p2[i+1]/(Gamma**2 + 4*(Delta+delta2)**2))))**0.5
    #     )
    # J2.append(beta*J1[i+1])
    
    # gamma1.append(-(p1[i+1]-p1[i])/(dt*p1[i]))
    gamma1.append(-(p1[i+1]-p1[i])/(dt*p1[i]))
    gamma2.append(-(p2[i+1]-p2[i])/(dt*p2[i])) 
    def J2_active_value(idx):
        denom2 = (1 - p0[idx] - p1[idx])
        if denom2 <= 1e-12:
            denom2 = 1e-12
        return float((2 * J1[idx]**2 * p1[idx] * (Gamma**2 + 4*Delta**2) / (denom2 * Gamma**2))**0.5)
    J1.append(J1[i])
    if i <= 25:
        J2.append(0.0)
    elif 25 < i <= 35:
        J2.append(J2_active_value(i+1))
    elif 35 < i < 60:
        J2.append(0.0)
    elif 60 <= i < 70:
        J2.append(J2_active_value(i+1))
    elif 70 <= i < 95:
        J2.append(0.0)
    elif 95 <= i < 105:
        J2.append(J2_active_value(i+1))
    elif 105 <= i < 120:
        J2.append(0.0)
    else:
        # 120 <= i < end
        if i < (measure_times - 2):
            J2.append(J2_active_value(i+1))
        else:
            J2.append(J2[-1])
    I1.append(-0.75 * Sigy[-1])
    
        
    
    
    
    
    
    states=result1.states[-1]
    # print("Iteration {}: J1={}, gamma1={}".format(i, J1[i], gamma1[i]))
v_test=[]
v_m=[]
v_err=[]
J_train=J1[:measure_times-1]  
J2_active=J2[:measure_times-1]



t=tlist2[:(measure_times-1)]    
v_m=[]  
signal=[]
signal_fit=[]
v_err=[]
v_test=[]



J2=J2[:measure_times-1]







desired_width_mm = 240  # 例如，期望的宽度（毫米）

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


# 创建图形和坐标轴
fig, ax = plt.subplots(figsize=(12, 7), dpi=200)


# plt.rcParams.update({
#     'font.size': 23,             # 基础字体大小
#     'axes.titlesize': 23,        # 标题大小
#     'axes.labelsize': 25,        # 坐标轴标签大小
#     'xtick.labelsize': 25,       # X轴刻度大小 - 从23增加到25
#     'ytick.labelsize': 25,       # Y轴刻度大小 - 从23增加到25
#     'legend.fontsize': 22,       # 图例大小 - 从20增加到22
#     'axes.linewidth': 1.5,       # 坐标轴线宽
#     'lines.linewidth': 2.5,      # 数据线宽
#     'lines.markersize': 10,      # 数据点大小（如果有的话）
# })
ax.plot(t,gamma1, 
        color="#080808",  # 专业橙色
        linewidth=5, 
        label=r'γ of state $\mathbf{|1\rangle}$ ', 
        
        alpha=0.95, 
        zorder=3,
        solid_capstyle='round')



# 新增：右侧J2曲线
ax2 = ax.twinx()
ax2.plot(t, J2, 
         color="#EE0AE3", 
         linewidth=5, 
         label=r'$\mathbf{J_2}$ waveform', 
         alpha=0.95, 
         zorder=2,
         linestyle='--')
# ax.plot(t,signal, 
#         color="#F00B31",  # 专业橙色
#         linewidth=4, 
#         label='Sim ', 
#         alpha=0.95, 
#         zorder=3,
#         solid_capstyle='round')
# fig2, ax2 = plt.subplots(figsize=(14, 10), dpi=1300)
# ax2.plot(t,signal_fit, 
#         color="#044B13",  # 专业橙色
#         linewidth=4, 
#         label='EXp ', 
#         alpha=0.95, 
#         zorder=3,
#         solid_capstyle='round')
# ax.scatter(t, signal_fit,
#            marker='o', color="#DB1010", s=50, edgecolors="#D4159B", linewidths=1,
#            zorder=4,
#            label='Experiment')
# ax.plot(t,signal, 
#         color="#0D6412",  # 专业橙色
#         linewidth=4, 
#         label=r'$\mathbf{V=I·\gamma}$ ', 
#         alpha=0.95, 
#         zorder=3,
#         solid_capstyle='round')
# ax.scatter(t, signal_fit,
#            zorder=4,
#            label='Experiment', s=50, color="#F8083C",edgecolors='black' ,linewidths=2)


# ax.plot(t,v_m, 
#         color="#0D6412",  # 专业橙色
#                  linewidth=5, 
#          label=r'$\mathbf{V=I·\gamma}$ ', 
#          alpha=0.95, 
#          zorder=3,
#          solid_capstyle='round')
# ax.errorbar(t, v_test, yerr=v_err, 
#              fmt='o', color="#EE053F", markersize=7,  markeredgecolor="#8A1830",markeredgewidth=2, 
#              capsize=6, capthick=2, elinewidth=1,zorder=4,
#              label='Experiment')
# ax.axhline(y=-bias, color="#EE0B0B", linestyle='--', linewidth=5, label='Bias')
# plt.subplot(4,2,2)
# ax.plot(t, gamma1, 'b',linewidth=2.5, label=r'$\gamma \ of\  state\  |0\rangle$ ')
# plt.ylabel(r'$\gamma$(MHz) ')

# 6. 坐标轴标签和标题
ax.set_xlabel('Time(μs)', fontweight='bold', labelpad=5)
# ax.set_ylabel('Signal', fontweight='bold', labelpad=5)
ax.set_ylabel(r'γ(MHz)', fontweight='bold', labelpad=5)
# ax.set_title('Phasic Mode', 
#             fontweight='bold', 
#             pad=5,
#             fontsize=30)
# ax2.set_xlabel('Time(μs)', fontweight='bold', labelpad=5)
# ax.set_ylabel('Signal', fontweight='bold', labelpad=5)
# # plt.subplot(4,1,4)
# # plt.plot(tlist2, I, 'b--', label=r'I ')
# # plt.axhline(y=I_th, color='k', linestyle='--', label='Theoretical I_th')
# # plt.ylabel('I ')
# # plt.xlabel('Time/us')
# # plt.legend()
# 设置右侧y轴
ax2.set_ylabel(r'$\mathbf{J_2}$(MHz)', fontweight='bold', labelpad=5)
ax2.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
ax2.yaxis.set_major_locator(MaxNLocator(6))
ax2.yaxis.set_minor_locator(AutoMinorLocator(2))
ax2.ticklabel_format(style='sci', axis='y', scilimits=(-2,2))
# plt.subplot(4,2,3)
# # plt.plot(t, V1, 'r--', label=r'V1 ')
# ax.plot(t, v_m, 'b-', linewidth=2.5, label=r'V=I·$\gamma$')  # 使用实线而不是虚线
# ax.axhline(y=-bias, color='red', linestyle='--', linewidth=2.5, label='Theoretical bias')
# ax.set_ylabel(r'$\gamma$(MHz)', fontsize=28)  # 增加y轴标签字体大小
# ax.set_xlabel('Time(μs)', fontsize=28)  # 增加x轴标签字体大小
# ax.set_ylabel('V', fontsize=28)  # 增加y轴标签字体大小
# ax.set_title('Spiking Mode', fontsize=28, pad=20)
# 添加图例
# 图例合并
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc='lower right', 
           prop={'weight': 'bold'}, frameon=True, fancybox=True, shadow=True, 
           framealpha=0.95, edgecolor='gray', facecolor='white', borderpad=0.5, 
           labelspacing=0.5, fontsize=33)


# legend2 = ax2.legend(
#     loc='upper right',
#     prop={'weight': 'bold'},
#     frameon=True,
#     fancybox=True,
#     shadow=True,
#     framealpha=0.95,
#     edgecolor='gray',
#     facecolor='white',
#     borderpad=0.5,      # 边框与内容的距离，减小
#     labelspacing=0.5,   # label之间的垂直距离，减小
#     fontsize=25,        # 字体大小可适当减小
#     handlelength=1.2,   # 图例线条长度
#     handletextpad=0.5 )  # 线条和文字的距离
    # 7. 纵坐标使用科学计数法并增大分度值
# 9. 图例美化
# legend = ax.legend(
#     loc='lower right',
#     prop={'weight': 'bold'},
#     frameon=True,
#     fancybox=True,
#     shadow=True,
#     framealpha=0.95,
#     edgecolor='gray',
#     facecolor='white',
#     borderpad=0.5,      # 边框与内容的距离，减小
#     labelspacing=0.5,   # label之间的垂直距离，减小
#     fontsize=25,        # 字体大小可适当减小
#     handlelength=1.2,   # 图例线条长度
#     handletextpad=0.5 )  # 线条和文字的距离
    # 7. 纵坐标使用科学计数法并增大分度值
# 设置科学计数法格式
formatter = ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((-2, 2))
ax.yaxis.set_major_formatter(formatter)

# ax.yaxis.set_major_locator(FixedLocator([0, 1]))
ax.xaxis.set_major_locator(MaxNLocator(4))
ax.xaxis.set_minor_locator(AutoMinorLocator(2))
ax.ticklabel_format(style='sci', axis='y', scilimits=(-2,2))
ax.grid(True, alpha=0.4, linestyle='-', linewidth=0.8)
ax.grid(True, which='minor', alpha=0.2, linestyle='--', linewidth=0.5)
ax.set_facecolor('#f8f9fa')
fig.patch.set_facecolor('white')
ax.tick_params(axis='both', which='major', length=6, width=2)
ax.tick_params(axis='both', which='minor', length=4, width=1)
ax2.tick_params(axis='y', which='major', length=6, width=2)
ax2.tick_params(axis='y', which='minor', length=4, width=1)
plt.tight_layout()
# plt.savefig('phasic_signal.svg', format='svg', bbox_inches='tight')
# plt.savefig('phasic_voltage.svg', format='svg', bbox_inches='tight')
# plt.savefig('phasic_gamma_J2.svg', format='svg', bbox_inches='tight')
plt.show()