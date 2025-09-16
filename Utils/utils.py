import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from datetime import timedelta
import os
import sys
from sympy import Function, pretty, init_printing, symbols, diff, Matrix, cos, sin, pi, tan, solve, Eq
x, y = symbols('x y')
# x, y, θ, ϕ_1, v_1, ω, v_ϕ_1, a_v_1, a_ω, a, b, β, d = symbols('x y θ ϕ_1 v_1 ω v_ϕ_1 a_v_1 a_ω a b β d')
init_printing(use_unicode=True)
# Get the directory containing the current file
project_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print("Current folder:", project_folder)

sys.path.append(project_folder)
from decimal import Decimal
from Utils.constants import *

def wrap_angle(theta):
    return np.arctan2(np.sin(theta), np.cos(theta))

def plot_path(random_tree, subplot, filename='all_trajectories.png', color=None):
    paths = []
    for node in random_tree:
        if node.parent == None:
            continue
        paths.append(node.trajectory)
    
    for path in paths:
        x = [node[0] for node in path]
        y = [node[1] for node in path]

        if color is not None:
            subplot.plot(x, y, color=color)  # 'marker=o' aggiunge un marcatore a forma di cerchio su ogni punto
        else:
            subplot.plot(x, y)  # 'marker=o' aggiunge un marcatore a forma di cerchio su ogni punto

    # Aggiunge titolo e etichette agli assi
    subplot.set_title('Optimal Path')
    subplot.set_xlabel('x')
    subplot.set_ylabel('y')

    # plt.gca().invert_yaxis()
    # Mostra il grafico
    # plt.savefig(filename)

    # plt.show()
    return

def single_plot_path(random_tree, color=None):
    paths = []
    for node in random_tree:
        if node.parent == None:
            continue
        paths.append(node.trajectory)
    
    for path in paths:
        x = [node[0] for node in path]
        y = [node[1] for node in path]

        if color is not None:
            plt.plot(x, y, color=color)  # 'marker=o' aggiunge un marcatore a forma di cerchio su ogni punto
        else:
            plt.plot(x, y)  # 'marker=o' aggiunge un marcatore a forma di cerchio su ogni punto

    # Aggiunge titolo e etichette agli assi
    plt.title('All trajectories')
    plt.xlabel('x')
    plt.ylabel('y')

    # plt.gca().invert_yaxis()
    # Mostra il grafico
    # plt.savefig(filename)

    # plt.show()
    return


def plot_trajectory(trajectory, color=None):
    
    x = [node[0] for node in trajectory]
    y = [node[1] for node in trajectory]

    if color is not None:
        plt.plot(x, y, color=color)  # 'marker=o' aggiunge un marcatore a forma di cerchio su ogni punto
    else:
        plt.plot(x, y)  # 'marker=o' aggiunge un marcatore a forma di cerchio su ogni punto

    # Aggiunge titolo e etichette agli assi
    plt.title('Optimal Path')
    plt.xlabel('x')
    plt.ylabel('y')

    # plt.gca().invert_yaxis()
    # Mostra il grafico
    # plt.savefig(filename)

    # plt.show()
    return


def plot_optimal_path(optimal_path):
        x = [node[0] for node in optimal_path]
        y = [node[1] for node in optimal_path]

        plt.plot(x, y, marker='o')  # 'marker=o' aggiunge un marcatore a forma di cerchio su ogni punto

        # Aggiunge titolo e etichette agli assi
        plt.title('Optimal Path')
        plt.xlabel('x')
        plt.ylabel('y')

        # plt.gca().invert_yaxis()
        # Mostra il grafico
        plt.savefig('optimal_path.png')

        # plt.show()
        return

def plot_in_time(x, freq, label='Variable', filename=None):
    x_values = np.arange(0, len(x)/freq, 1/freq)

    # Creazione del grafico
    plt.figure(figsize=(10, 4))  # Imposta le dimensioni del grafico
    plt.plot(x_values, x, '-b')  # '-b' sta per 'linea blu'
    plt.xlabel('Time (s)')
    plt.ylabel(label)

    if filename is not None:
        plt.savefig(os.path.join(project_folder ,f'./imgs/{filename}.png'))
    else:
        plt.show()
    return

def subplot_in_time(x, tau_step, subplot=None, label='Variable'):
    x_values = np.arange(0, len(x)*(1/tau_step), (1/tau_step))
    if len(x_values) > len(x):
        x_values = x_values[:len(x)]
    elif len(x_values) < len(x):
        x = x[:len(x_values)]

    # Creazione del grafico
    
    subplot.plot(x_values, x, '-b')  # '-b' sta per 'linea blu'
    if max(x) < 1e-10:
        subplot.set_ylim(-5, 5)
    subplot.set_xlabel('Time (s)')
    subplot.set_ylabel(label)

    return

def get_phi_n(state, P_1, P_n):
    x, y, th, phi_1, v_1, om  = state

    phi_n = np.arctan2(v_1 * np.sin(phi_1) + om * (P_n[0] - P_1[0]), v_1 * np.cos(phi_1) + om * (P_1[1] - P_n[1]))  

    return phi_n

def get_v_wheel_n(state, P_i):
    x, y, th, phi_1, v_1, om  = state 

    R = np.array([[np.cos(th), -np.sin(th)],
                                 [np.sin(th), np.cos(th)]])
    
    Rbeta = np.array([[np.cos(th + phi_1), -np.sin(th + phi_1)],
                                 [np.sin(th), np.cos(th)]])
    
    xy = np.array([x,y])

    Os_i = xy + R@P_i

    # Ow_1 = Os_i + 
    return

# def compute_jacobian(state, P_i):
#     Os1 = Matrix([x,y])
#     R = Matrix([[cos(θ), sin(θ)],
#                 [sin(θ), cos(θ)]])
    # jac_A = f.jacobian([x, y,θ, ϕ_1])
    
    return

def compute_line(x,y,phi):
    m = -1/np.tan(phi)
    b = y - m*x

    return (m,b)

def compute_ICR(phis, points):
    # Calcolo delle pendenze (m) e delle intercette (b) delle perpendicolari agli angoli di sterzata
    phis = [phi - np.pi/2 for phi in phis]
    # print(f'phis: {phis}')
    # print(f'points: {points}')
    m = {}
    b = {}
    for wheel, (x_pos, y_pos) in enumerate(points):
        angle = phis[wheel]
        if angle == 0:
            m[wheel] = float('inf')
            b[wheel] = x_pos
        else:
            m[wheel] = tan(angle)
            b[wheel] = y_pos - m[wheel]*x_pos

    # Creazione delle equazioni per le rette perpendicolari agli angoli di sterzata
    equations = []
    for wheel in m:
        if m[wheel] == float('inf'):
            # Per le rette verticali, l'equazione è semplicemente x = b (dove b è la posizione x della ruota)
            equations.append(Eq(x, b[wheel]))
        else:
            equations.append(Eq(y, m[wheel]*x + b[wheel]))
    
    print(f'equations: {equations}')
    solution = solve((equations[0], equations[1], equations[2], equations[3]), (x,y))

    if len(solution) > 0:
        print(f'solution found: {solution}')
    return

def coordinating_function(state, P1, Pi, previous_phi_i):
        
    v_1 = state[4]
    om = state[5]
    
    phi_1 = state[3]
    
    phi_i_1 = np.arctan2(v_1 * np.sin(phi_1) + om * (Pi[0] - P1[0] ), v_1 * np.cos(phi_1) + om * (P1[1] - Pi[1] ))  
    phi_i_2 = phi_i_1 + np.pi
    
    if abs(wrap_angle(phi_i_1 - previous_phi_i)) < abs(wrap_angle(phi_i_2 - previous_phi_i)) : 
        return phi_i_1
                
    else:
        return phi_i_2

def compute_all_phis(state, prev_phis = None):
    phis = []
    if prev_phis is None:
        phis.append(state[3])
        phis.append(get_phi_n(state, P1, P2))
        phis.append(get_phi_n(state, P1, P3))
        phis.append(get_phi_n(state, P1, P4))
    else:
        phis.append(state[3])
        phis.append(coordinating_function(state, P1, P2, prev_phis[1]))
        phis.append(coordinating_function(state, P1, P3, prev_phis[2]))
        phis.append(coordinating_function(state, P1, P4, prev_phis[3]))

    return phis

def compute_all_vs(state, phis, model):
    curr_vs = []
    curr_vs.append(model.get_v_joint(state,P1,phis[0]))
    curr_vs.append(model.get_v_joint(state,P2,phis[1]))
    curr_vs.append(model.get_v_joint(state,P3,phis[2]))
    curr_vs.append(model.get_v_joint(state,P4,phis[3]))
    return curr_vs


def wheel_steer_vel(state, input, P1, Pi):
    
    phi_1 = state[3]
    v_1 = state[4]
    om = state[5]
    v_phi_1 = input[0]
    a_v_1 = input[1]
    a_om = input[2]
        
    v_phi_i = ((om*(P1[1] - Pi[1]) + v_1*np.cos(phi_1))*(a_v_1*np.sin(phi_1) - a_om*(P1[0] - Pi[0]) + v_1*v_phi_1*np.cos(phi_1)) + (om*(P1[0] - Pi[0]) - v_1*np.sin(phi_1))*(a_om*(P1[1] - Pi[1]) + a_v_1*np.cos(phi_1) - v_1*v_phi_1*np.sin(phi_1)))/(np.power(om*(P1[1] - Pi[1]) + v_1*np.cos(phi_1),2) + np.power(om*(P1[0] - Pi[0]) - v_1*np.sin(phi_1),2))
    
    return v_phi_i

def wheel_drive_vel(state, dx, dy, phi_i, v_phi_i, Pi,d):
    
    th = state[2]
    om = state[5]
    Px = Pi[0]
    Py = Pi[1]
    
    
    v_i = [(om*np.cos(phi_i + th) + v_phi_i*np.cos(phi_i + th))*d + dx - om*(Py*np.cos(th) + Px*np.sin(th)),
    (om*np.sin(phi_i + th) + v_phi_i*np.sin(phi_i + th))*d + dy + om*(Px*np.cos(th) - Py*np.sin(th))]
    
    
    return v_i
    
def compute_extra_sub_plot(optimal_trajectory, rrt, optimal_inputs, initial_state, subfolder='nonlinear/', linear = True):

    previous_phi = [initial_state[3],initial_state[3],initial_state[3],initial_state[3]]
    
    # optimal_trajectory = optimal_trajectory
    # optimal_trajectory[0] = optimal_trajectory[0].tolist()
    # print("OPTIMAL_TRAJ", optimal_trajectory)
    phis = []
    v_phis = []
    v_ss = []
    v_ds = []
    v_ws = []

    for i,el in enumerate(optimal_trajectory):
        rrt.model.set_state(el)
        if i == len(optimal_trajectory) - 1:
            rrt.model.u = optimal_inputs[i]
        else:
            rrt.model.u = optimal_inputs[i+1]

        phi = rrt.model.coordinating_function(previous_phi)
        vs = compute_all_vs(el, previous_phi, rrt.model)

        v_phi = rrt.model.wheels_steer_vel()
        v_d = rrt.model.wheels_vel()

        theta = el[2]
        
        v_w1 = np.array([np.cos(theta + phi[0]), np.sin(theta + phi[0])]) @ v_d[0]
        v_w2 = np.array([np.cos(theta + phi[1]), np.sin(theta + phi[1])]) @ v_d[1]
        v_w3 = np.array([np.cos(theta + phi[2]), np.sin(theta + phi[2])]) @ v_d[2]
        v_w4 = np.array([np.cos(theta + phi[3]), np.sin(theta + phi[3])]) @ v_d[3]

        v_w = [v_w1, v_w2, v_w3, v_w4]
        
        if linear:
            optimal_trajectory[i].append(phi)
            optimal_trajectory[i].append(v_phi)
            optimal_trajectory[i].append(v_d)

        phis.append(phi)
        v_phis.append(v_phi)
        v_ds.append(v_d)
        v_ss.append(vs)
        v_ws.append(v_w)
        
        previous_phi = phi
        # print(f'el: {el}')

        # print(f'test_print: {optimal_trajectory[i]}, ')
    
    phis = np.array(phis)
    v_phis = np.array(v_phis)
    v_ws = np.array(v_ws)
    v_ss = np.array(v_ss)

    # print(f'phis: {phis}, ')
    # print(f'v_phis: {v_phis}, ')
    # print(f'v_ws: {v_ws}, ')
    # print(f'v_ss: {v_ss}, ')

    phi_1 = phis[:, 0]
    phi_2 = phis[:, 1]
    phi_3 = phis[:, 2]
    phi_4 = phis[:, 3]

    v_phi_1 = v_phis[:, 0]
    v_phi_2 = v_phis[:, 1]
    v_phi_3 = v_phis[:, 2]
    v_phi_4 = v_phis[:, 3]

    v_ss_1 = v_ss[:, 0]
    v_ss_2 = v_ss[:, 1]
    v_ss_3 = v_ss[:, 2]
    v_ss_4 = v_ss[:, 3]

    v_ws_1 = v_ws[:, 0]
    v_ws_2 = v_ws[:, 1]
    v_ws_3 = v_ws[:, 2]
    v_ws_4 = v_ws[:, 3]

    #test plt.close()
    plt.close()
    plt.figure()
    fig, axs = plt.subplots(4, 1, figsize=(10, 10))
    subplot_in_time(phi_1, FREQUENCY, axs[0], label='ϕ_1')
    subplot_in_time(phi_2, FREQUENCY, axs[1], label='ϕ_2')
    subplot_in_time(phi_3, FREQUENCY, axs[2], label='ϕ_3')
    subplot_in_time(phi_4, FREQUENCY, axs[3], label='ϕ_4')
    fig.tight_layout()

    plt.savefig(os.path.join(project_folder ,f'./imgs/{subfolder}phis.png'))

    fig, axs = plt.subplots(4, 1, figsize=(10, 10))
    subplot_in_time(v_phi_1, FREQUENCY, axs[0], label='v_ϕ_1')
    subplot_in_time(v_phi_2, FREQUENCY, axs[1], label='v_ϕ_2')
    subplot_in_time(v_phi_3, FREQUENCY, axs[2], label='v_ϕ_3')
    subplot_in_time(v_phi_4, FREQUENCY, axs[3], label='v_ϕ_4')
    fig.tight_layout()

    plt.savefig(os.path.join(project_folder ,f'./imgs/{subfolder}v_phis.png'))

    fig, axs = plt.subplots(4, 1, figsize=(10, 10))
    subplot_in_time(v_ss_1, FREQUENCY, axs[0], label='v_s_1')
    subplot_in_time(v_ss_2, FREQUENCY, axs[1], label='v_s_2')
    subplot_in_time(v_ss_3, FREQUENCY, axs[2], label='v_s_3')
    subplot_in_time(v_ss_4, FREQUENCY, axs[3], label='v_s_4')
    fig.tight_layout()

    plt.savefig(os.path.join(project_folder ,f'./imgs/{subfolder}v_ss.png'))

    fig, axs = plt.subplots(4, 1, figsize=(10, 10))
    subplot_in_time(v_ws_1, FREQUENCY, axs[0], label='v_w_1')
    subplot_in_time(v_ws_2, FREQUENCY, axs[1], label='v_w_2')
    subplot_in_time(v_ws_3, FREQUENCY, axs[2], label='v_w_3')
    subplot_in_time(v_ws_4, FREQUENCY, axs[3], label='v_w_4')
    fig.tight_layout()

    plt.savefig(os.path.join(project_folder ,f'./imgs/{subfolder}v_ws.png'))

    return optimal_trajectory

if __name__ == "__main__":
    # t = symbols('t')
    
    # x = Function('x')(t) 
    # y = Function('y')(t) 
    # θ = Function('θ')(t) 
    # ϕ = Function('ϕ')(t) 
    # β = Function('β')(t) 

    # θ_d = Function('\dot{θ}')(t)
    # alias = {diff(θ, t):θ_d,}
    
    # a, b, d = symbols('a b d')
    # Px, Py = symbols('Px Py')

    # Pos = Matrix([x,y])
    # print(Pos)

    # R = Matrix([[cos(θ), -sin(θ)],
    #             [sin(θ), cos(θ)]])
    # print(R)

    # Oj1 = Matrix([Px,Py])  
    # print(Oj1)

    # Os1 = Pos + R@Oj1 
    # print(Os1)

    # Rbeta = Matrix([[cos(θ + ϕ), -sin(θ + ϕ)],
    #             [sin(θ + ϕ), cos(θ + ϕ)]])
    
    # dis = Matrix([0,-d])

    # Ow1 = Os1 + Rbeta@dis
    # # print(Ow1)
    # # print(pretty(Ow1))

    # # dOw1 = Ow1.jacobian([x, y, θ, ϕ_1])
    # # print(pretty(dOw1))
    # dOw1 = diff(Ow1, t)
    # print(pretty(dOw1))

    # # from sympy.physics.vector import dynamicsymbols
    # # from sympy.physics.vector.printing import vpprint, vlatex
    # # f = dynamicsymbols('f')
    # # vpprint(f.diff())     # ḋ
    # # vlatex(f.diff())      # '\\dot{d}'
    
    state=[1,1,np.pi/3,np.pi/4,1,1]
    input=[1,1,1]
    dx=1
    dy=1
    
    d=1
    
    # P1 = [1,1]
    # P2 = [1,-1]
    
    phi_2 = coordinating_function(state,P1,P2)
    v_phi_2 = wheel_steer_vel(state,input,P1,P2)
    
    print("phi_2 : ",phi_2)
    print("v_phi_2 :",v_phi_2)
    print("v_2 : ",wheel_drive_vel(state,dx,dy,phi_2,v_phi_2,P2,d))



    