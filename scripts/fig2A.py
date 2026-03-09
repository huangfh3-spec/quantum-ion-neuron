#!/usr/bin/env python
# coding: utf-8




import numpy as np
from qutip import *
import matplotlib.pyplot as plt
from scipy.signal import argrelmax, argrelmin
import pandas as pd  # 新增：用于导出 CSV

# ====================== parameters ========================
dim = 7  
total_t = 640
delta = 0.040
Delta = -2.615 * 2 * np.pi  # detuning
dt =4
N=total_t//dt
measure_times =int(total_t // dt + 1)

Gamma = 19.6 * 2 * np.pi  # decay rate
options = {"nsteps": 1500000}
basis_states = [basis(dim, i) for i in range(dim)]
Dt = np.linspace(0, dt, 80)
tlist = np.linspace(0, total_t, measure_times)
gamma = [0.005]

beta = 0

V1 = []
I1 = []


t = []

Sx1 = [1]
Sy1 = [0]

states = (basis_states[1]+basis_states[0] ).unit()

bloch_points = [(1,0,0)]
P0 = [1/2]
P1 = [1/2]
P2 = [0]
P3 = [0]


# ===================== operators ========================
Sigx = basis_states[1] * basis_states[0].dag() + basis_states[0] * basis_states[1].dag()
Sigy = -1j * basis_states[1] * basis_states[0].dag() + 1j * basis_states[0] * basis_states[1].dag()
Sigz = basis_states[1] * basis_states[1].dag() - basis_states[0] * basis_states[0].dag()
Iden= basis_states[1] * basis_states[1].dag() + basis_states[0] * basis_states[0].dag()
Iden_excited=basis_states[2] * basis_states[2].dag() + basis_states[3] * basis_states[3].dag()+basis_states[4] * basis_states[4].dag()+basis_states[5] * basis_states[5].dag()+basis_states[6] * basis_states[6].dag()

#rotating operator
def U(theta , phi):
    return (np.cos(theta/2)*Iden-1j*np.sin(theta/2)*(np.cos(phi)*Sigx+np.sin(phi)*Sigy)+Iden_excited)




H_structure1 = basis_states[1] * basis_states[5].dag() + basis_states[5] * basis_states[1].dag()


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






from qutip import Bloch


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



    Sx1.append(float(np.real(rho_x[1,1] - rho_x[0,0])))
    Sy1.append(float(np.real(rho_y[1,1] - rho_y[0,0])))

    bloch_points.append([
    Sx1[-1],
    Sy1[-1],
    float(np.real(result.expect[2][-1]))
    ])



    
    P0.append(result.expect[3][-1])
    P1.append(result.expect[4][-1])
    P2.append(result.expect[5][-1])
    P3.append(result.expect[6][-1])

    # gamma.append(gamma[i] +0.5 * Sy1[i] * dt * 0.0004)   #dgamma=A*I*dt A=-1/3750
    gamma.append(gamma[i]) 
    # J1.append(
    #     (
    #         3* gamma_real1[i + 1]* P1[i + 1]/ (4* Gamma* (2 * P1[i + 1] / (Gamma**2 )
    #                - beta**2 * (P2[i + 1] + P3[i + 1]) / (Gamma**2 + 4 * Delta**2)
    #            ))
    #     )
    #     ** 0.5
    
    # J1.append(0)
    J1.append(

         (3*(Gamma)*gamma[i+1]/8)**0.5)    #next J1 value, gamma is updated with the current Sy1 value
    
    
    



    states = result1.states[-1]

points = np.array(bloch_points).T  # 转换为(3, N)形状
# b.add_points(points, meth='s')

b = Bloch()
b.zlabel = [r'$|1\rangle$', r'$|0\rangle$']


b.font_size = 18  
b.line_color = ['red']
b.line_width = [3]
for i in range(points.shape[1] - 1):
    b.add_line(points[:, i], points[:, i+1],color='red', linewidth=3)    
b.point_color = ['r']  # 统一颜色为蓝色

b.show()
b.point_size = [5]
b.make_sphere()
plt.show(block=True)















