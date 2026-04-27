#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import time
from dataclasses import dataclass, field

import numpy as np
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt


@dataclass
class SoHModelParams:
    x1: float = 0.010
    x2: float = 0.020
    x3: float = 0.100
    x4: float = 0.150
    x5: float = 0.015
    x6: float = 0.200
    x7: float = 0.050
    x8: float = 0.010
    x9: float = 0.400
    x10: float = 0.70

    def as_array(self):
        return np.array(
            [
                self.x1, self.x2, self.x3, self.x4, self.x5,
                self.x6, self.x7, self.x8, self.x9, self.x10
            ],
            dtype=float,
        )


@dataclass
class module:
    SoC: float
    SoH: float          # SoH is now in [0, 1]
    Imax: float
    xmin: float
    xmax: float
    idP: str = "M0"
    capacity_nominal_ah: float = 50.0

    LF: np.ndarray = field(default_factory=lambda: np.array([]))
    Traj: np.ndarray = field(default_factory=lambda: np.array([]))
    SoHTraj: np.ndarray = field(default_factory=lambda: np.array([]))
    FCETraj: np.ndarray = field(default_factory=lambda: np.array([]))

    throughput_ah: float = 0.0
    FCE: float = 0.0
    current_direction: int = 0
    soc_cycle_start: float = None

    def __post_init__(self):
        self.id = self.idP
        self.SoC0 = self.SoC

        self.SoH = max(0.0, min(1.0, self.SoH))
        self.SoH0 = self.SoH

        self.SoCmin = self.xmin
        self.SoCmax = self.xmax

        self.Traj = np.array([self.SoC])
        self.SoHTraj = np.array([self.SoH])
        self.FCETraj = np.array([self.FCE])

        self.soc_cycle_start = self.SoC

    def reset(self):
        self.SoC = self.SoC0
        self.SoH = self.SoH0
        self.LF = np.array([])
        self.Traj = np.array([self.SoC])
        self.SoHTraj = np.array([self.SoH])
        self.FCETraj = np.array([0.0])
        self.throughput_ah = 0.0
        self.FCE = 0.0
        self.current_direction = 0
        self.soc_cycle_start = self.SoC

    def clone(self):
        return copy.deepcopy(self)

    def short_state(self):
        return f"{self.id}: SoC={self.SoC:.3f}, SoH={self.SoH:.3f}, FCE={self.FCE:.4f}"


def get_switching_matrices():
    J = 6
    SM = np.empty(J, dtype=object)

    SM[0] = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    SM[1] = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
    SM[2] = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    SM[3] = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
    SM[4] = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
    SM[5] = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])

    return SM


def SortLoad(BatteryModules, mode="charge"):
    """
    Classical SoC-balancing sorting.

    charge:
        lowest SoC gets highest positive current

    discharge:
        highest SoC gets highest negative current magnitude

    soh:
        highest SoH first
    """
    BatteryModules = BatteryModules.copy()

    if mode == "charge":
        BatteryModules = sorted(BatteryModules, key=lambda b: b.SoC)

    elif mode == "discharge":
        BatteryModules = sorted(BatteryModules, key=lambda b: b.SoC, reverse=True)

    elif mode == "soh":
        BatteryModules = sorted(BatteryModules, key=lambda b: b.SoH, reverse=True)

    else:
        raise ValueError("mode must be 'charge', 'discharge', or 'soh'")

    return np.array(BatteryModules, dtype=object)


def stress_factor(current, dod, soc_avg, params):
    x = params.as_array()
    C = abs(current)

    return abs(
        (x[0] * np.exp(x[1] * C) + x[2])
        * (x[3] * np.exp(x[4] * dod) + x[5])
        * (x[6] * np.exp(x[7] * soc_avg) + x[8])
    )


def update_soh_module(battery, current, dt_hours, soh_params):
    """
    Battery.SoH is stored in [0, 1].

    The empirical degradation equation is evaluated internally in percent,
    because the original formula uses 100 - SoH.
    """
    prev_soc = battery.SoC
    direction = 0 if abs(current) < 1e-12 else int(np.sign(current))

    if battery.current_direction == 0 and direction != 0:
        battery.current_direction = direction
        battery.soc_cycle_start = prev_soc

    elif direction != 0 and direction != battery.current_direction:
        battery.soc_cycle_start = prev_soc
        battery.current_direction = direction

    delta_ah = current * dt_hours
    battery.throughput_ah += abs(delta_ah)
    battery.FCE = battery.throughput_ah / (2.0 * battery.capacity_nominal_ah)

    battery.SoC = battery.SoC + delta_ah / battery.capacity_nominal_ah
    battery.SoC = min(battery.SoCmax, max(battery.SoCmin, battery.SoC))

    soc0 = battery.soc_cycle_start
    soc = battery.SoC

    soc_avg = ((abs(soc - soc0) / 2.0) + min(soc, soc0)) * 100.0
    dod = abs(soc - soc0) * 100.0

    stress = stress_factor(current, dod, soc_avg, soh_params)
    stress = max(stress, 1e-12)

    x10 = soh_params.x10

    soh_percent = battery.SoH * 100.0
    fce0 = ((100.0 - soh_percent) / stress) ** (1.0 / x10)

    soh_percent_new = 100.0 - abs(stress * ((battery.FCE + fce0) ** x10))
    soh_percent_new = max(0.0, min(100.0, soh_percent_new))

    battery.SoH = soh_percent_new / 100.0

    battery.Traj = np.append(battery.Traj, battery.SoC)
    battery.SoHTraj = np.append(battery.SoHTraj, battery.SoH)
    battery.FCETraj = np.append(battery.FCETraj, battery.FCE)


def MPCsession_v1(BatteryModules, dt, LF, N, T1, T2, T3, solver_threads=1, verbose=False):
    """
    Original MPC function kept unchanged in structure.
    """
    J = 6
    n_states = len(BatteryModules)
    n_controls = n_states

    x_traj = np.zeros((N + 1, n_states))
    u_traj = np.zeros((N, n_controls))
    delta_traj = np.zeros((N, J))
    x_traj[0, :] = [battery.SoC for battery in BatteryModules]

    model = gp.Model()
    model.Params.OutputFlag = 0
    model.Params.Threads = solver_threads

    u_vars = model.addVars(range(N), n_controls, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="u")
    x_vars = model.addVars(range(N + 1), n_states, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="x")
    delta_vars = model.addVars(range(N), J, vtype=GRB.BINARY, name="delta")

    SM = get_switching_matrices()

    R = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    Q = [[1e0, 0, 0], [0, 1e0, 0], [0, 0, 1e0]]

    objective = gp.quicksum(
        (
            (u_vars[k, i] - BatteryModules[i].SoH) * Q[i][l] * (u_vars[k, l] - BatteryModules[i].SoH)
            + (u_vars[k, i] * R[i][l] * u_vars[k, l])
        )
        for i in range(n_states)
        for l in range(n_states)
        for k in range(N)
    )

    model.addConstrs(
        (x_vars[k, i] <= BatteryModules[i].SoCmax for i in range(n_controls) for k in range(N)),
        name="soc_upper",
    )

    model.addConstrs(
        (BatteryModules[i].SoCmin <= x_vars[k, i] for i in range(n_controls) for k in range(N)),
        name="soc_lower",
    )

    model.addConstrs(
        (
            x_vars[k + 1, j] == x_vars[k, j] + dt * u_vars[k, j]
            for j in range(n_states)
            for k in range(N)
        ),
        name="dynamics",
    )

    model.addConstrs(
        (
            u_vars[k, j] == gp.quicksum(delta_vars[k, i] * np.dot(SM[i][j], LF) for i in range(J))
            for j in range(n_states)
            for k in range(N)
        ),
        name="control_form",
    )

    model.addConstrs(
        (x_vars[0, i] == x_traj[0, i] for i in range(n_states)),
        name="initial_state",
    )

    model.addConstrs(
        (gp.quicksum(delta_vars[k, i] for i in range(J)) == 1 for k in range(N)),
        name="delta_sum",
    )

    model.addConstr(gp.quicksum(delta_vars[k, 0] + delta_vars[k, 3] for k in range(N)) <= T1[0])
    model.addConstr(gp.quicksum(delta_vars[k, 1] + delta_vars[k, 2] for k in range(N)) <= T1[1])
    model.addConstr(gp.quicksum(delta_vars[k, 4] + delta_vars[k, 5] for k in range(N)) <= T1[2])

    model.addConstr(gp.quicksum(delta_vars[k, 1] + delta_vars[k, 4] for k in range(N)) <= T2[0])
    model.addConstr(gp.quicksum(delta_vars[k, 0] + delta_vars[k, 5] for k in range(N)) <= T2[1])
    model.addConstr(gp.quicksum(delta_vars[k, 2] + delta_vars[k, 3] for k in range(N)) <= T2[2])

    model.addConstr(gp.quicksum(delta_vars[k, 2] + delta_vars[k, 5] for k in range(N)) <= T3[0])
    model.addConstr(gp.quicksum(delta_vars[k, 3] + delta_vars[k, 4] for k in range(N)) <= T3[1])
    model.addConstr(gp.quicksum(delta_vars[k, 0] + delta_vars[k, 1] for k in range(N)) <= T3[2])

    model.setObjective(objective, GRB.MINIMIZE)
    model.optimize()

    if model.status == GRB.OPTIMAL:
        if verbose:
            print("model feasible")

        x_traj = np.array([[x_vars[i, j].X for j in range(n_states)] for i in range(N + 1)])
        u_traj = np.array([[u_vars[i, j].X for j in range(n_states)] for i in range(N)])
        delta_traj = np.array([[delta_vars[i, j].X for j in range(J)] for i in range(N)])

    elif model.status == GRB.INFEASIBLE:
        if verbose:
            print("model infeasible")

    return x_traj, u_traj, delta_traj


def solve_system_v1(a_list, l1, l2, l3, P, solver_threads=1, verbose=False):
    N = len(a_list)

    model = gp.Model("equation_system")
    model.Params.OutputFlag = 0
    model.Params.Threads = solver_threads

    T1 = model.addVars(N, name="T1", lb=0)
    T2 = model.addVars(N, name="T2", lb=0)
    T3 = model.addVars(N, name="T3", lb=0)

    for i in range(N):
        model.addConstr(a_list[i] == T1[i] * l1 * P + T2[i] * l2 * P + T3[i] * l3 * P)

    sum_T1 = gp.quicksum(T1[i] for i in range(N))
    sum_T2 = gp.quicksum(T2[i] for i in range(N))
    sum_T3 = gp.quicksum(T3[i] for i in range(N))

    model.addConstr(sum_T1 == sum_T2)
    model.addConstr(sum_T1 == sum_T3)

    model.addConstr(T1[0] + T2[0] + T3[0] == T1[1] + T2[1] + T3[1])
    model.addConstr(T1[0] + T2[0] + T3[0] == T1[2] + T2[2] + T3[2])

    objective = 60 * T1[0] + 40 * T2[1] + 20 * T3[2] + 5 * T2[0] + 3 * T3[1] + 1 * T2[2]
    model.setObjective(objective, GRB.MAXIMIZE)
    model.optimize()

    if model.status == GRB.OPTIMAL:
        return {
            "T1": [T1[i].X for i in range(N)],
            "T2": [T2[i].X for i in range(N)],
            "T3": [T3[i].X for i in range(N)],
            "ObjectiveValue": model.ObjVal,
        }

    raise RuntimeError(f"solve_system_v1 failed. Gurobi status: {model.status}")


def UpdateLimits(T1, T2, T3, delta_traj):
    delta_step = delta_traj[0, :]

    if delta_step[0] >= 1:
        T1[0] = max(0, T1[0] - 1)
        T2[1] = max(0, T2[1] - 1)
        T3[2] = max(0, T3[2] - 1)
    if delta_step[1] >= 1:
        T2[0] = max(0, T2[0] - 1)
        T1[1] = max(0, T1[1] - 1)
        T3[2] = max(0, T3[2] - 1)
    if delta_step[2] >= 1:
        T3[0] = max(0, T3[0] - 1)
        T1[1] = max(0, T1[1] - 1)
        T2[2] = max(0, T2[2] - 1)
    if delta_step[3] >= 1:
        T1[0] = max(0, T1[0] - 1)
        T3[1] = max(0, T3[1] - 1)
        T2[2] = max(0, T2[2] - 1)
    if delta_step[4] >= 1:
        T2[0] = max(0, T2[0] - 1)
        T3[1] = max(0, T3[1] - 1)
        T1[2] = max(0, T1[2] - 1)
    if delta_step[5] >= 1:
        T3[0] = max(0, T3[0] - 1)
        T2[1] = max(0, T2[1] - 1)
        T1[2] = max(0, T1[2] - 1)

    return T1, T2, T3


def build_pack_current_segments():
    return [
        (+15.0, 3 * 60),
        (+30.0, 3 * 60),
        (+40.0, 3 * 60),
        (0.0, 1 * 60),
        (-20.0, 3 * 60),
        (-35.0, 3 * 60),
        (-45.0, 3 * 60),
        (0.0, 1 * 60),
    ]


def create_random_battery_modules(seed=None):
    rng = np.random.default_rng(seed)
    sohs = rng.uniform(0.5, 1.0, size=3)

    BatteryModules = np.empty(3, dtype=object)
    BatteryModules[0] = module(0.4, float(sohs[0]), 50.0, 0.2, 0.9, "M1", 50.0)
    BatteryModules[1] = module(0.5, float(sohs[1]), 50.0, 0.2, 0.8, "M2", 50.0)
    BatteryModules[2] = module(0.3, float(sohs[2]), 50.0, 0.2, 0.6, "M3", 50.0)

    return BatteryModules


def run_classical_controller(BatteryModules, dt_hours, n_cycles, soh_params):
    LF_base = np.array([0.6, 0.3, 0.1])
    segments = build_pack_current_segments()

    for _ in range(n_cycles):
        for pack_current, duration_steps in segments:
            for _ in range(duration_steps):

                if pack_current > 0:
                    BatteryModules = SortLoad(BatteryModules, mode="charge")
                elif pack_current < 0:
                    BatteryModules = SortLoad(BatteryModules, mode="discharge")

                currents = LF_base * pack_current

                for battery, current in zip(BatteryModules, currents):
                    battery.LF = np.append(battery.LF, current)
                    update_soh_module(battery, current, dt_hours, soh_params)

    return BatteryModules


def run_mpc_controller(
    BatteryModules,
    dt_hours,
    n_cycles,
    horizon,
    soh_params,
    solver_threads=1,
    verbose_segments=False,
):
    LF_base = np.array([0.6, 0.3, 0.1])
    LF_base = np.sort(LF_base)[::-1]

    segments = build_pack_current_segments()
    capacity_ah = BatteryModules[0].capacity_nominal_ah
    dt_soc_per_amp = dt_hours / capacity_ah

    solve_system_calls = 0

    for cycle in range(n_cycles):
        for segment_id, (pack_current, duration_steps) in enumerate(segments):
            solve_system_calls += 1

            if verbose_segments:
                print(
                    f"[MPC] Cycle {cycle + 1}/{n_cycles}, "
                    f"segment {segment_id + 1}/8, "
                    f"Ipack={pack_current:.1f} A"
                )

            if pack_current > 0:
                AList = [b.SoCmax - b.SoC for b in BatteryModules]
                P_segment = abs(pack_current) * dt_soc_per_amp

            elif pack_current < 0:
                AList = [b.SoC - b.SoCmin for b in BatteryModules]
                P_segment = abs(pack_current) * dt_soc_per_amp

            else:
                AList = [0.0, 0.0, 0.0]
                P_segment = 1.0

            result = solve_system_v1(
                AList,
                LF_base[0],
                LF_base[1],
                LF_base[2],
                P_segment,
                solver_threads=solver_threads,
                verbose=False,
            )

            T1 = result["T1"]
            T2 = result["T2"]
            T3 = result["T3"]

            for _ in range(duration_steps):
                if abs(pack_current) < 1e-12:
                    u = np.zeros(3)
                    delta_traj = np.zeros((horizon, 6))
                else:
                    LF = LF_base * pack_current

                    _, u_traj, delta_traj = MPCsession_v1(
                        BatteryModules,
                        dt_soc_per_amp,
                        LF,
                        horizon,
                        T1,
                        T2,
                        T3,
                        solver_threads=solver_threads,
                        verbose=False,
                    )

                    T1, T2, T3 = UpdateLimits(T1, T2, T3, delta_traj)
                    u = u_traj[0, :]

                for battery, current in zip(BatteryModules, u):
                    battery.LF = np.append(battery.LF, current)
                    update_soh_module(battery, current, dt_hours, soh_params)

    return BatteryModules, solve_system_calls


def run_single_test(
    test_id,
    seed,
    n_cycles,
    horizon,
    dt_sec,
    soh_params,
    solver_threads=1,
    keep_trajectories=False,
):
    dt_hours = dt_sec / 3600.0

    base_modules = create_random_battery_modules(seed)
    init_sohs = np.array([b.SoH for b in base_modules], dtype=float)

    classical_modules = np.array([b.clone() for b in base_modules], dtype=object)
    mpc_modules = np.array([b.clone() for b in base_modules], dtype=object)

    start = time.time()

    classical_modules = run_classical_controller(
        classical_modules,
        dt_hours,
        n_cycles,
        soh_params,
    )

    mpc_modules, solve_system_calls = run_mpc_controller(
        mpc_modules,
        dt_hours,
        n_cycles,
        horizon,
        soh_params,
        solver_threads=solver_threads,
        verbose_segments=False,
    )

    elapsed = time.time() - start

    classical_final_soh = np.array([b.SoH for b in classical_modules], dtype=float)
    mpc_final_soh = np.array([b.SoH for b in mpc_modules], dtype=float)

    result = {
        "test_id": test_id,
        "seed": seed,
        "init_sohs": init_sohs,
        "classical_final_soh": classical_final_soh,
        "mpc_final_soh": mpc_final_soh,
        "classical_mean_soh": float(np.mean(classical_final_soh)),
        "mpc_mean_soh": float(np.mean(mpc_final_soh)),
        "classical_min_soh": float(np.min(classical_final_soh)),
        "mpc_min_soh": float(np.min(mpc_final_soh)),
        "solve_system_calls": solve_system_calls,
        "runtime": elapsed,
    }

    if keep_trajectories:
        result["classical_modules"] = classical_modules
        result["mpc_modules"] = mpc_modules

    return result


def aggregate_results(results):
    classical_mean = np.array([r["classical_mean_soh"] for r in results])
    mpc_mean = np.array([r["mpc_mean_soh"] for r in results])

    classical_min = np.array([r["classical_min_soh"] for r in results])
    mpc_min = np.array([r["mpc_min_soh"] for r in results])

    runtime = np.array([r["runtime"] for r in results])
    solve_calls = np.array([r["solve_system_calls"] for r in results])

    return {
        "num_tests": len(results),
        "classical_mean_soh_avg": float(np.mean(classical_mean)),
        "mpc_mean_soh_avg": float(np.mean(mpc_mean)),
        "classical_min_soh_avg": float(np.mean(classical_min)),
        "mpc_min_soh_avg": float(np.mean(mpc_min)),
        "mpc_gain_mean_soh": float(np.mean(mpc_mean - classical_mean)),
        "mpc_gain_min_soh": float(np.mean(mpc_min - classical_min)),
        "avg_test_runtime": float(np.mean(runtime)),
        "sum_test_runtime": float(np.sum(runtime)),
        "avg_solve_system_calls": float(np.mean(solve_calls)),
        "total_solve_system_calls": int(np.sum(solve_calls)),
    }


def plot_best_test_trajectories(
    classical_modules,
    mpc_modules,
    dt_sec,
    save_path="best_test_soc_soh.png",
):
    time_hours = np.arange(len(classical_modules[0].Traj)) * dt_sec / 3600.0

    plt.figure(figsize=(12, 9))

    plt.subplot(2, 1, 1)
    for i in range(3):
        plt.plot(time_hours, classical_modules[i].Traj, label=f"Classical {classical_modules[i].id}")
        plt.plot(time_hours, mpc_modules[i].Traj, "--", label=f"MPC {mpc_modules[i].id}")
    plt.ylabel("SoC")
    plt.title("Best Test: SoC over 10 Cycles")
    plt.grid(True)
    plt.legend(ncol=2)

    plt.subplot(2, 1, 2)
    for i in range(3):
        plt.plot(time_hours, classical_modules[i].SoHTraj, label=f"Classical {classical_modules[i].id}")
        plt.plot(time_hours, mpc_modules[i].SoHTraj, "--", label=f"MPC {mpc_modules[i].id}")
    plt.xlabel("Time [h]")
    plt.ylabel("SoH [-]")
    plt.title("Best Test: SoH over 10 Cycles")
    plt.grid(True)
    plt.legend(ncol=2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()