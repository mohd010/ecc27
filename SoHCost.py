#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 19:22:47 2024

@author: mohammad
"""
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
import time 
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import random

# Enable interactive mode
plt.ion()

# functions for Bidirectional charging
class module:
    def __init__(self, SoC, SoH, Imax, xmin, xmax, idP="M0"):
        self.SoC = SoC
        self.SoH = SoH
        self.Imax = Imax
        self.id = idP
        self.LF = np.array([])
        self.Traj = np.array([self.SoC])
        self.SoC0 = SoC
        self.SoCmin = xmin
        self.SoCmax = xmax
        
    def reset(self):
        self.SoC = self.SoC0
        self.LF = np.array([])
        self.Traj = np.array([self.SoC])
        
    def print_SoC(self):
        print(f"Module {self.idP} has state of charge: {self.SoC}")       

def MPCsession_v1(BatteryModules, dt, LF, N, T1, T2, T3):
    J = 6
    n_states = len(BatteryModules)
    n_controls = n_states
    x_traj = np.zeros((N+1, n_states))
    u_traj = np.zeros((N, n_controls))
    delta_traj = np.zeros((N, J))
    x_traj[0, :] = [battery.SoC for battery in BatteryModules]
    

    # Create a new Gurobi model
    model = gp.Model()
    
    # Define variables
    # Define decision variables
    u_vars = model.addVars(range(N), n_controls, lb=-GRB.INFINITY, ub=GRB.INFINITY, name='u')
    x_vars = model.addVars(range(N + 1), n_states, lb=-GRB.INFINITY, ub=GRB.INFINITY, name='x')
    delta_vars = model.addVars(range(N), J, vtype=GRB.BINARY, name='delta')
    SM = np.empty(J, dtype=object)
    SM[0] = np.array([[1,0,0],[0,1,0],[0,0,1]])
    SM[1] = np.array([[0,1,0],[1,0,0],[0,0,1]])
    SM[2] = np.array([[0,0,1],[1,0,0],[0,1,0]])
    SM[3] = np.array([[1,0,0],[0,0,1],[0,1,0]])
    SM[4] = np.array([[0,1,0],[0,0,1],[1,0,0]])
    SM[5] = np.array([[0,0,1],[0,1,0],[1,0,0]])
    
        
    R = [[0,0,0],[0,0,0],[0,0,0]]
    Q = [[1e0,0,0],[0,1e0,0],[0,0,1e0]]

    # Set objective function
    # objective = gp.quicksum(1*(x_vars[k, i] - x_ref[i]) *Q[i][l]* (x_vars[k, l] - x_ref[l]) for i in range(n_states) for l in range(n_states) for k in range(N + 1))
    objective = gp.quicksum(((u_vars[k, i] - BatteryModules[i].SoH) *Q[i][l]* (u_vars[k, l] - BatteryModules[i].SoH)+(u_vars[k, i]  *R[i][l]* u_vars[k, l] )) for i in range(n_states) for l in range(n_states) for k in range(N))
    # objective += gp.quicksum(1*(x_vars[N, i] - x_ref[i]) *Q[i][l]* (x_vars[N, l] - x_ref[l]) for i in range(n_states) for l in range(n_states) for k in range(N + 1))

    # Add state constraints using addConstrs()
    model.addConstrs((x_vars[k,i] <= BatteryModules[i].SoCmax for i in range(n_controls) for k in range(N)), name="constraint1_u")  # x[i+1] - x[i] <= 1
    model.addConstrs((BatteryModules[i].SoCmin <= x_vars[k,i] for i in range(n_controls) for k in range(N)), name="constraint2_u")  # x[i+1] - x[i] <= 1
        
    # model.addConstrs((diff_min[i] <= u_vars[k,i]-u_vars[k-1,i] for i in range(n_controls) for k in range(1,N)), name="constraint1_diff")  # x[i+1] - x[i] <= 1
    # model.addConstrs((u_vars[k,i]-u_vars[k-1,i] <= diff_max[i] for i in range(n_controls) for k in range(1,N)), name="constraint2_diff")  # x[i+1] - x[i] <= 1
    # model.addConstrs((y[i+1] - y[i] <= 1 for i in range(T-1)), name="constraint1_y")  # y[i+1] - y[i] <= 1

    # Add system dynamics as constraints
    SystemDynamics = model.addConstrs((x_vars[k + 1, j] == x_vars[k, j]  +
                     dt*u_vars[k,j] for j in range(n_states) for k in range(N)),
                    name='dynamics_constrs') #A[j, i] *
    ControlForm = model.addConstrs(
        (
            u_vars[k, j] == gp.quicksum(delta_vars[k, i] * np.dot(SM[i][j], LF) for i in range(J))
        ) 
        for j in range(n_states) 
        for k in range(N)
    )
    # Set initial state constraints
    InitialConstr = model.addConstrs((x_vars[0, i] == x_traj[0, i] for i in range(n_states)), name='initial_state_constrs')
    
    model.addConstrs(gp.quicksum(delta_vars[k,i] for i in range(J)) == 1 for k in range(N))        
    
    # Additional constraints on delta sums
    
    # Constraint group for T1
    model.addConstr(gp.quicksum(delta_vars[k,0] + delta_vars[k,3] for k in range(N)) <= T1[0], name="T1_0_constraint")
    model.addConstr(gp.quicksum(delta_vars[k,1] + delta_vars[k,2] for k in range(N)) <= T1[1], name="T1_1_constraint")
    model.addConstr(gp.quicksum(delta_vars[k,4] + delta_vars[k,5] for k in range(N)) <= T1[2], name="T1_2_constraint")
    
    # Constraint group for T2
    model.addConstr(gp.quicksum(delta_vars[k,1] + delta_vars[k,4] for k in range(N)) <= T2[0], name="T2_0_constraint")
    model.addConstr(gp.quicksum(delta_vars[k,0] + delta_vars[k,5] for k in range(N)) <= T2[1], name="T2_1_constraint")
    model.addConstr(gp.quicksum(delta_vars[k,2] + delta_vars[k,3] for k in range(N)) <= T2[2], name="T2_2_constraint")
    
    # Constraint group for T3
    model.addConstr(gp.quicksum(delta_vars[k,2] + delta_vars[k,5] for k in range(N)) <= T3[0], name="T3_0_constraint")
    model.addConstr(gp.quicksum(delta_vars[k,3] + delta_vars[k,4] for k in range(N)) <= T3[1], name="T3_1_constraint")
    model.addConstr(gp.quicksum(delta_vars[k,0] + delta_vars[k,1] for k in range(N)) <= T3[2], name="T3_2_constraint")
    
    ########################################################################
    model.setObjective(objective, GRB.MINIMIZE)

    # Set solver parameters
    model.Params.OutputFlag = 0  # Disable solver output
    
    # Set time limit
    # model.setParam('TimeLimit', 1)
    
    # Optimize the model
    model.optimize()

    if model.status == GRB.OPTIMAL:
        # Store the optimal state trajectory
        # x_vals.append([x[i].X for i in range(T)])
        # y_vals.append([y[i].X for i in range(T)])
        print('model feasible')
        x_traj = np.array([[x_vars[i,j].x for j in range(n_states)] for i in range(N+1)]) 
        u_traj = np.array([[u_vars[i,j].x for j in range(n_states)] for i in range(N)]) 
        delta_traj = np.array([[delta_vars[i,j].x for j in range(J)] for i in range(N)]) 
        initial_state = [x_vars[0,j].x for j in range(n_states)]
    elif model.status == 4:
        print('model infeasible')
    return x_traj, u_traj, delta_traj

def SortLoad(BatteryModules, SoC):
    N = len(BatteryModules)
    if SoC:
        for i in range(N):
            for j in range(i+1,N):
                if BatteryModules[i].SoC > BatteryModules[j].SoC:
                    T = BatteryModules[i]
                    BatteryModules[i] = BatteryModules[j]
                    BatteryModules[j] = T
    else:
        for i in range(N):
            for j in range(i+1,N):
                if BatteryModules[i].SoH < BatteryModules[j].SoH:
                    T = BatteryModules[i]
                    BatteryModules[i] = BatteryModules[j]
                    BatteryModules[j] = T
    return BatteryModules
    
def PlotBatteries(time, Tl, BatteryModules, flag):
      
    # Plot each with a predefined symbol and color
    plt.plot(time, BatteryModules[0].Traj, 'ro-', label=f"SoC {BatteryModules[0].id}")  # Red with circles
    plt.plot(time, BatteryModules[1].Traj, 'bs--', label=f"SoC {BatteryModules[1].id}")  # Blue with squares and dashed line
    plt.plot(time, BatteryModules[2].Traj, 'g^-.', label=f"SoC {BatteryModules[2].id}")  # Green with triangles and dash-dot line
    
    # Add labels, legend, and title
    plt.xlabel("Time (min)")
    plt.ylabel("SoC (%)")
    plt.title("State of Charge Over Time")
    plt.legend()
    plt.grid(True)
    
    # Prevent trimming
    plt.tight_layout()
    # Display the plot
    rand_num = random.randint(1000, 9999)
    filename = f"my_figure_{rand_num}.pdf"
    plt.savefig(filename, bbox_inches="tight")
    plt.show()
    
    # Define time array (x-axis)
    time = np.linspace(0, Tl/60, Tl)  # 100 points from 0 to 10 seconds
    # Plot each LF with a predefined symbol and color
    fig, ax = plt.subplots()
    ax.plot(time, BatteryModules[0].LF, 'r-', label=f"LF {BatteryModules[0].id}")  # Red with circles
    ax.plot(time, BatteryModules[1].LF, 'b-', label=f"LF {BatteryModules[1].id}")  # Blue with squares and dashed line
    ax.plot(time, BatteryModules[2].LF, 'g-.', label=f"LF {BatteryModules[2].id}")  # Green with triangles and dash-dot line
    
    # Add labels, legend, and title
    plt.xlabel("Time (min)")
    plt.ylabel("LF")
    plt.title("Load Factor Over Time")
    plt.legend(loc='upper left')
    plt.grid(True)
    
    if flag:
    # Zoomed inset
        axins = inset_axes(ax, width="30%", height="30%", loc="upper right")  # Inset position
        axins.step(time, BatteryModules[0].LF, 'r-', label=f"LF {BatteryModules[0].id}")  # Red with circles
        axins.step(time, BatteryModules[1].LF, 'b-', label=f"LF {BatteryModules[1].id}")  # Blue with squares and dashed line
        axins.step(time, BatteryModules[2].LF, 'g-', label=f"LF {BatteryModules[2].id}")  # Green with triangles and dash-dot line
        axins.set_xlim(7, 7.1)  # Zoomed range for x
        # axins.set_ylim(-0.5, 1)  # Zoomed range for y
        axins.grid(True)
        
        # Mark the zoomed area
        mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="red")
    
    # Prevent trimming
    plt.tight_layout()
    # Display the plot
    rand_num = random.randint(1000, 9999)
    filename = f"my_figure_{rand_num}.pdf"
    plt.savefig(filename, bbox_inches="tight")
    plt.show()
    
def solve_system_v1(a_list, l1, l2, l3, P):
    N = len(a_list)

    # Create model
    model = gp.Model("equation_system")

    # Variables: T1_i, T2_i, T3_i for i in 1,...,N
    T1 = model.addVars(N, name="T1", lb=0)
    T2 = model.addVars(N, name="T2", lb=0)
    T3 = model.addVars(N, name="T3", lb=0)

    # Constraints: a_i = T1_i*l1 + T2_i*l2 + T3_i*l3
    for i in range(N):
        model.addConstr(a_list[i] == T1[i]*l1*P + T2[i]*l2*P + T3[i]*l3*P, name=f"eq_{i}")

    # Constraint: sum of T1_i == sum of T2_i == sum of T3_i
    sum_T1 = gp.quicksum(T1[i] for i in range(N))
    sum_T2 = gp.quicksum(T2[i] for i in range(N))
    sum_T3 = gp.quicksum(T3[i] for i in range(N))
    model.addConstr(sum_T1 == sum_T2, name="sum_T1_eq_T2")
    model.addConstr(sum_T1 == sum_T3, name="sum_T1_eq_T3")

    sum_Ti1 = T1[0]+T2[0]+T3[0]
    sum_Ti2 = T1[1]+T2[1]+T3[1] 
    sum_Ti3 = T1[2]+T2[2]+T3[2]
    model.addConstr(sum_Ti1 == sum_Ti2, name="sum_T1_eq_T4")
    model.addConstr(sum_Ti1 == sum_Ti3, name="sum_T1_eq_T5")

    # Objective: Maximize T1_0 + T2_1 + ... + T3_{N-1}
    objective = 60*T1[0]+40*T2[1]+20*T3[2]+5*T2[0]+3*T3[1]+1*T2[2]

    model.setObjective(objective, GRB.MAXIMIZE)

    # Optimize
    model.optimize()

    # Collect results
    if model.status == GRB.OPTIMAL:
        solution = {
            "T1": [T1[i].X for i in range(N)],
            "T2": [T2[i].X for i in range(N)],
            "T3": [T3[i].X for i in range(N)],
            "ObjectiveValue": model.ObjVal
        }
        return solution
    else:
        return {"status": model.status}
    

def solve_system_v3(a_list, alpha1, alpha2, alpha3, I_G, lambda1, lambda2, f_type="quadratic", f_param=1.0):
    """
    Solve

        min sum_i [ lambda1 * f(C_i) + lambda2 * (T1_i + T2_i + T3_i) ]

    subject to
        a_i = T1_i*alpha1*I_G + T2_i*alpha2*I_G + T3_i*alpha3*I_G
        sum_i T1_i = sum_i T2_i = sum_i T3_i
        T1_i + T2_i + T3_i equal across modules
        C_i * (T1_i + T2_i + T3_i) = I_G * (alpha1*T1_i + alpha2*T2_i + alpha3*T3_i)
        T1_i, T2_i, T3_i >= 0
        C_i >= 0

    Parameters
    ----------
    a_list : list[float]
        Required charge increments per module.
    alpha1, alpha2, alpha3 : float
        Position/load factors.
    I_G : float
        Charging current.
    lambda1, lambda2 : float
        Objective weights.
    f_type : str
        Type of degradation function for C_i.
        Supported:
            - "linear"    -> f(C) = k*C
            - "quadratic" -> f(C) = k*C^2
    f_param : float
        Parameter k used in f(C).

    Returns
    -------
    dict
    """
    N = len(a_list)
    model = gp.Model("equation_system_v2")

    # Decision variables
    T1 = model.addVars(N, name="T1", lb=0.0)
    T2 = model.addVars(N, name="T2", lb=0.0)
    T3 = model.addVars(N, name="T3", lb=0.0)
    C = model.addVars(N, name="C", lb=0.0)

    # Energy / charge balance constraints
    for i in range(N):
        model.addConstr(
            a_list[i] == T1[i] * alpha1 * I_G + T2[i] * alpha2 * I_G + T3[i] * alpha3 * I_G,
            name=f"eq_{i}"
        )

    # Balanced assignment across positions
    sum_T1 = gp.quicksum(T1[i] for i in range(N))
    sum_T2 = gp.quicksum(T2[i] for i in range(N))
    sum_T3 = gp.quicksum(T3[i] for i in range(N))
    model.addConstr(sum_T1 == sum_T2, name="sum_T1_eq_T2")
    model.addConstr(sum_T1 == sum_T3, name="sum_T1_eq_T3")

    # Equal total charging time across modules
    total_time = {}
    for i in range(N):
        total_time[i] = T1[i] + T2[i] + T3[i]

    for i in range(1, N):
        model.addConstr(total_time[0] == total_time[i], name=f"equal_total_time_{i}")

    # g_i constraint:
    # C_i * sum_j T_j^[i] = sum_j alpha_j T_j^[i] I_G
    # This is bilinear, so we must enable nonconvex optimization in Gurobi.
    for i in range(N):
        model.addConstr(
            C[i] * total_time[i] ==
            I_G * (alpha1 * T1[i] + alpha2 * T2[i] + alpha3 * T3[i]),
            name=f"g_{i}"
        )

    # Objective construction
    degradation_terms = []
    for i in range(N):
        if f_type == "linear":
            degradation_terms.append(f_param * C[i])
        elif f_type == "quadratic":
            degradation_terms.append(f_param * C[i] * C[i])
        else:
            raise ValueError("Unsupported f_type. Use 'linear' or 'quadratic'.")

    objective = gp.quicksum(
        lambda1 * degradation_terms[i] + lambda2 * total_time[i]
        for i in range(N)
    )

    model.setObjective(objective, GRB.MINIMIZE)

    # Needed because of bilinear terms C[i] * total_time[i]
    model.Params.NonConvex = 2

    # Optimize
    model.optimize()

    # Collect results
    if model.status == GRB.OPTIMAL:
        solution = {
            "T1": [T1[i].X for i in range(N)],
            "T2": [T2[i].X for i in range(N)],
            "T3": [T3[i].X for i in range(N)],
            "C":  [C[i].X for i in range(N)],
            "ObjectiveValue": model.ObjVal
        }
        return solution
    else:
        return {"status": model.status}
    
def solve_system_v2(a_list, l1, l2, l3, P):
    N = len(a_list)

    # Create model
    model = gp.Model("equation_system")

    # Variables: T1_i, T2_i, T3_i for i in 1,...,N
    T1 = model.addVars(N, name="T1", lb=0)
    T2 = model.addVars(N, name="T2", lb=0)
    T3 = model.addVars(N, name="T3", lb=0)

    # Constraints: a_i = T1_i*l1 + T2_i*l2 + T3_i*l3
    for i in range(N):
        model.addConstr(a_list[i] == T1[i]*l1*P + T2[i]*l2*P + T3[i]*l3*P, name=f"eq_{i}")

    # Constraint: sum of T1_i == sum of T2_i == sum of T3_i
    sum_T1 = gp.quicksum(T1[i] for i in range(N))
    sum_T2 = gp.quicksum(T2[i] for i in range(N))
    sum_T3 = gp.quicksum(T3[i] for i in range(N))
    model.addConstr(sum_T1 == sum_T2, name="sum_T1_eq_T2")
    model.addConstr(sum_T1 == sum_T3, name="sum_T1_eq_T3")

    sum_Ti1 = T1[0]+T2[0]+T3[0]
    sum_Ti2 = T1[1]+T2[1]+T3[1] 
    sum_Ti3 = T1[2]+T2[2]+T3[2]
    model.addConstr(sum_Ti1 == sum_Ti2, name="sum_T1_eq_T4")
    model.addConstr(sum_Ti1 == sum_Ti3, name="sum_T1_eq_T5")

    # Objective: Maximize T1_0 + T2_1 + ... + T3_{N-1}
    objective = 60*T1[0]+40*T2[1]+20*T3[2]+5*T2[0]+3*T3[1]+1*T2[2]

    model.setObjective(objective, GRB.MAXIMIZE)

    # Optimize
    model.optimize()

    # Collect results
    if model.status == GRB.OPTIMAL:
        solution = {
            "T1": [T1[i].X for i in range(N)],
            "T2": [T2[i].X for i in range(N)],
            "T3": [T3[i].X for i in range(N)],
            "ObjectiveValue": model.ObjVal
        }
        return solution
    else:
        return {"status": model.status}

def UpdateLimits(T1,T2,T3,delta_traj):
    
    delta_step = delta_traj[0,:]
    if delta_step[0] >= 1:
        T1[0] = max(0, T1[0]-1)
        T2[1] = max(0, T2[1]-1)
        T3[2] = max(0, T3[2]-1)
    if delta_step[1] >= 1:
        T2[0] = max(0, T2[0]-1)
        T1[1] = max(0, T1[1]-1)
        T3[2] = max(0, T3[2]-1)
    if delta_step[2] >= 1:
        T3[0] = max(0, T3[0]-1)
        T1[1] = max(0, T1[1]-1)
        T2[2] = max(0, T2[2]-1)
    if delta_step[3] >= 1:
        T1[0] = max(0, T1[0]-1)
        T3[1] = max(0, T3[1]-1)
        T2[2] = max(0, T2[2]-1)
    if delta_step[4] >= 1:
        T2[0] = max(0, T2[0]-1)
        T3[1] = max(0, T3[1]-1)
        T1[2] = max(0, T1[2]-1)
    if delta_step[5] >= 1:
        T3[0] = max(0, T3[0]-1)
        T2[1] = max(0, T2[1]-1)
        T1[2] = max(0, T1[2]-1)
    
    return T1, T2, T3
    
N = 3
BatteryModules = np.empty(N, dtype=object)
# BatteryModules[0] = module(0.6, 1, 1, 0.2, 0.9, "M1")
# BatteryModules[1] = module(0.3, 0.7, 1, 0.2, 0.8, "M2")
# BatteryModules[2] = module(0.5, 0.4, 1, 0.2, 0.7, "M3")
BatteryModules[0] = module(0.4, 1, 1, 0.2, 0.9, "M1")
BatteryModules[1] = module(0.5, 0.7, 1, 0.2, 0.8, "M2")
BatteryModules[2] = module(0.3, 0.4, 1, 0.2, 0.6, "M3")

J = 6
SM = np.empty(J, dtype=object)
SM[0] = np.array([[1,0,0],[0,1,0],[0,0,1]])
SM[1] = np.array([[0,1,0],[1,0,0],[0,0,1]])
SM[2] = np.array([[0,0,1],[1,0,0],[0,1,0]])
SM[3] = np.array([[1,0,0],[0,0,1],[0,1,0]])
SM[4] = np.array([[0,1,0],[0,0,1],[1,0,0]])
SM[5] = np.array([[0,0,1],[0,1,0],[1,0,0]])

M = 7
level = np.empty(M, dtype=object)
level[0] = np.array([1,1,1])
level[1] = np.array([0,1,1])
level[2] = np.array([0,0,1])
level[3] = np.array([0,0,0])
level[4] = np.array([0,0,-1])
level[5] = np.array([0,-1,-1])
level[6] = np.array([-1,-1,-1])
Imax = 35

Igrid = 30
PowerSec = 1/(15*60)
LF = np.array([0.6,0.3,0.1])
LF = np.sort(LF)[::-1]

AList = [BatteryModules[i].SoCmax-BatteryModules[i].SoC for i in range(N)]
result = solve_system_v1(AList, LF[0], LF[1], LF[2], PowerSec)
T1 = result['T1']
T2 = result['T2']
T3 = result['T3']

Th = 50
Tsim = 9*60
plt.rcParams.update({'font.size': 17})   # change 14 → any size you prefer


for tl in range(Tsim):
    print(f"Classical control iteration {tl}")
    BatteryModules = SortLoad(BatteryModules, True)
    for battery, lf in zip(BatteryModules, LF):
        battery.LF = np.append(battery.LF,lf)
        battery.SoC = battery.SoC+lf*PowerSec
        battery.Traj = np.append(battery.Traj,battery.SoC)
        

timeD = np.linspace(0, Tsim/60, Tsim+1)  # 100 points from 0 to 10 seconds
PlotBatteries(timeD, Tsim, BatteryModules, True)
####### Section on MPC
Tsim = 17*60
timeD = np.linspace(0, Tsim/60, Tsim+1)  # 100 points from 0 to 10 seconds
for battery in BatteryModules:
    battery.reset()
N = 5
BatteryModules = SortLoad(BatteryModules, False)
startTsim = time.time()
for tl in range(Tsim):
    print(f"MPC iteration {tl} out of {Tsim}")
    startdT = time.time()
    x_traj, u_traj, delta_traj = MPCsession_v1(BatteryModules, PowerSec, LF, N, T1, T2, T3)
    T1, T2, T3 = UpdateLimits(T1,T2,T3,delta_traj)
    enddT = time.time()
    Elapsed = enddT-startdT
    print(f"Elapsed time is: {Elapsed}")
    u = u_traj[0,:]
    for battery, lf in zip(BatteryModules, u):
        battery.LF = np.append(battery.LF,lf)
        battery.SoC = battery.SoC+lf*PowerSec
        battery.Traj = np.append(battery.Traj,battery.SoC)
endTsim = time.time()
ElapsedTsim = endTsim-startTsim  
print(f"Overall elapsed time is: {ElapsedTsim}")      
PlotBatteries(timeD, Tsim, BatteryModules, False)

# Disable interactive mode after plotting (optional)
plt.ioff()