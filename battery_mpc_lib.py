#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import copy
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset


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

    def as_array(self) -> np.ndarray:
        return np.array(
            [self.x1, self.x2, self.x3, self.x4, self.x5,
             self.x6, self.x7, self.x8, self.x9, self.x10],
            dtype=float
        )


@dataclass
class Module:
    soc: float
    soh: float
    imax: float
    soc_min: float
    soc_max: float
    capacity_nominal_ah: float
    module_id: str = "M0"

    throughput_ah: float = 0.0
    fce: float = 0.0
    current_a_hist: list = field(default_factory=list)
    soc_hist: list = field(default_factory=list)
    soh_hist: list = field(default_factory=list)
    fce_hist: list = field(default_factory=list)
    lf_hist: list = field(default_factory=list)
    pack_cycle_start_soc: float = None
    last_current_sign: int = 0

    def __post_init__(self):
        self.soc0 = self.soc
        self.soh0 = self.soh
        self.pack_cycle_start_soc = self.soc
        self.soc_hist = [self.soc]
        self.soh_hist = [self.soh]
        self.fce_hist = [self.fce]

    def reset(self):
        self.soc = self.soc0
        self.soh = self.soh0
        self.throughput_ah = 0.0
        self.fce = 0.0
        self.current_a_hist = []
        self.soc_hist = [self.soc]
        self.soh_hist = [self.soh]
        self.fce_hist = [self.fce]
        self.lf_hist = []
        self.pack_cycle_start_soc = self.soc
        self.last_current_sign = 0

    def clone(self):
        return copy.deepcopy(self)

    def short_state(self) -> str:
        return f"{self.module_id}(SoC={self.soc:.3f}, SoH={self.soh:.3f}, FCE={self.fce:.3f})"


def get_switching_matrices():
    J = 6
    sm = np.empty(J, dtype=object)
    sm[0] = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    sm[1] = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
    sm[2] = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    sm[3] = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
    sm[4] = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
    sm[5] = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
    return sm


def sort_modules(modules, by_soc=True):
    modules_sorted = modules.copy()
    n = len(modules_sorted)
    for i in range(n):
        for j in range(i + 1, n):
            if by_soc:
                if modules_sorted[i].soc > modules_sorted[j].soc:
                    modules_sorted[i], modules_sorted[j] = modules_sorted[j], modules_sorted[i]
            else:
                if modules_sorted[i].soh < modules_sorted[j].soh:
                    modules_sorted[i], modules_sorted[j] = modules_sorted[j], modules_sorted[i]
    return modules_sorted


def stress_factor(current_a, dod_pct, soc_avg_pct, params: SoHModelParams):
    x = params.as_array()
    c = abs(current_a)

    term1 = x[0] * np.exp(x[1] * c) + x[2]
    term2 = x[3] * np.exp(x[4] * dod_pct) + x[5]
    term3 = x[6] * np.exp(x[7] * soc_avg_pct) + x[8]

    return abs(term1 * term2 * term3)


def fce_offset_from_soh(soh_pct, stress, params: SoHModelParams):
    x10 = params.x10
    eps = 1e-12
    stress = max(stress, eps)
    remaining = max(100.0 - soh_pct, 0.0)
    return (remaining / stress) ** (1.0 / x10)


def soh_from_stress_and_fce(stress, fce_total, fce_offset, params: SoHModelParams):
    x10 = params.x10
    soh = 100.0 - abs(stress * ((fce_total + fce_offset) ** x10))
    return max(0.0, min(100.0, soh))


def compute_dod_and_socavg(soc_start, soc_now):
    soc_avg = ((abs(soc_now - soc_start) / 2.0) + min(soc_now, soc_start)) * 100.0
    dod = abs(soc_now - soc_start) * 100.0
    return dod, soc_avg


def update_module_state_and_soh(module: Module, current_a, dt_hours, soh_params: SoHModelParams):
    prev_soc = module.soc
    sign_now = 0 if abs(current_a) < 1e-12 else int(np.sign(current_a))

    if module.last_current_sign == 0:
        module.last_current_sign = sign_now
        module.pack_cycle_start_soc = prev_soc
    elif sign_now != 0 and sign_now != module.last_current_sign:
        module.pack_cycle_start_soc = prev_soc
        module.last_current_sign = sign_now

    delta_ah = current_a * dt_hours
    module.throughput_ah += abs(delta_ah)
    module.fce = module.throughput_ah / (2.0 * module.capacity_nominal_ah)

    new_soc = prev_soc + delta_ah / module.capacity_nominal_ah
    new_soc = min(module.soc_max, max(module.soc_min, new_soc))
    module.soc = new_soc

    dod_pct, soc_avg_pct = compute_dod_and_socavg(module.pack_cycle_start_soc, module.soc)
    stress = stress_factor(current_a, dod_pct, soc_avg_pct, soh_params)
    fce_offset = fce_offset_from_soh(module.soh, stress, soh_params)
    module.soh = soh_from_stress_and_fce(stress, module.fce, fce_offset, soh_params)

    module.current_a_hist.append(current_a)
    module.soc_hist.append(module.soc)
    module.soh_hist.append(module.soh)
    module.fce_hist.append(module.fce)


def build_pack_current_profile():
    segments = [
        (+15.0, 10 * 60),
        (+30.0, 10 * 60),
        (+40.0, 10 * 60),
        (0.0,   3 * 60),
        (-20.0, 10 * 60),
        (-35.0, 10 * 60),
        (-45.0, 10 * 60),
        (0.0,   3 * 60),
    ]

    profile = []
    for current, duration_sec in segments:
        profile.extend([current] * duration_sec)
    return np.array(profile, dtype=float)


def classical_load_factors(pack_current_a):
    if abs(pack_current_a) < 1e-12:
        return np.array([0.0, 0.0, 0.0])
    return np.array([0.6, 0.3, 0.1], dtype=float)


def permutation_cost_for_step(modules, shares, pack_current_a, dt_hours, soh_params: SoHModelParams):
    cost = 0.0
    for m, share in zip(modules, shares):
        module_current = share * pack_current_a
        soc_pred = m.soc + (module_current * dt_hours / m.capacity_nominal_ah)
        soc_pred = min(m.soc_max, max(m.soc_min, soc_pred))

        dod_pct, soc_avg_pct = compute_dod_and_socavg(m.pack_cycle_start_soc, soc_pred)
        stress = stress_factor(module_current, dod_pct, soc_avg_pct, soh_params)
        weak_penalty = 1.0 / max(m.soh, 1e-6)

        soc_margin_penalty = 0.0
        if soc_pred < m.soc_min + 0.02:
            soc_margin_penalty += 50.0 * (m.soc_min + 0.02 - soc_pred)
        if soc_pred > m.soc_max - 0.02:
            soc_margin_penalty += 50.0 * (soc_pred - (m.soc_max - 0.02))

        cost += weak_penalty * stress + soc_margin_penalty

    return cost


def choose_best_permutation_step(modules, pack_current_a, dt_hours, soh_params: SoHModelParams):
    sm = get_switching_matrices()
    base_shares = classical_load_factors(pack_current_a)

    if np.allclose(base_shares, 0.0):
        return np.zeros(3), np.zeros(6)

    best_idx = None
    best_shares = None
    best_cost = np.inf

    for j in range(6):
        shares = sm[j] @ base_shares
        cost = permutation_cost_for_step(modules, shares, pack_current_a, dt_hours, soh_params)
        if cost < best_cost:
            best_cost = cost
            best_idx = j
            best_shares = shares

    delta = np.zeros(6)
    delta[best_idx] = 1.0
    return best_shares, delta


def mpc_session_v1(modules, dt_hours, pack_current_horizon, T1, T2, T3, soh_params: SoHModelParams, verbose=False):
    n_states = len(modules)
    n_horizon = len(pack_current_horizon)
    j_count = 6
    sm = get_switching_matrices()

    if verbose:
        print(f"    [MPC] Building optimization problem for horizon {n_horizon} ...")

    model = gp.Model("soh_aware_mpc")
    model.Params.OutputFlag = 0
    model.Params.Threads = max(1, os.cpu_count() or 1)

    delta = model.addVars(n_horizon, j_count, vtype=GRB.BINARY, name="delta")
    x = model.addVars(n_horizon + 1, n_states, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="x")

    for i in range(n_states):
        model.addConstr(x[0, i] == modules[i].soc)

    objective_terms = []
    base_shares_cache = []

    for k in range(n_horizon):
        pack_current = pack_current_horizon[k]
        base_shares = classical_load_factors(pack_current)
        base_shares_cache.append(base_shares)

        model.addConstr(gp.quicksum(delta[k, j] for j in range(j_count)) == 1)

        step_costs = []
        for j in range(j_count):
            shares = sm[j] @ base_shares
            step_costs.append(
                permutation_cost_for_step(modules, shares, pack_current, dt_hours, soh_params)
            )

        for i in range(n_states):
            next_soc_expr = x[k, i] + (dt_hours * pack_current / modules[i].capacity_nominal_ah) * gp.quicksum(
                delta[k, j] * (sm[j] @ base_shares)[i] for j in range(j_count)
            )

            model.addConstr(x[k + 1, i] == next_soc_expr)
            model.addConstr(x[k + 1, i] <= modules[i].soc_max)
            model.addConstr(x[k + 1, i] >= modules[i].soc_min)

        objective_terms.append(
            gp.quicksum(delta[k, j] * step_costs[j] for j in range(j_count))
        )

    model.addConstr(gp.quicksum(delta[k, 0] + delta[k, 3] for k in range(n_horizon)) <= T1[0])
    model.addConstr(gp.quicksum(delta[k, 1] + delta[k, 2] for k in range(n_horizon)) <= T1[1])
    model.addConstr(gp.quicksum(delta[k, 4] + delta[k, 5] for k in range(n_horizon)) <= T1[2])

    model.addConstr(gp.quicksum(delta[k, 1] + delta[k, 4] for k in range(n_horizon)) <= T2[0])
    model.addConstr(gp.quicksum(delta[k, 0] + delta[k, 5] for k in range(n_horizon)) <= T2[1])
    model.addConstr(gp.quicksum(delta[k, 2] + delta[k, 3] for k in range(n_horizon)) <= T2[2])

    model.addConstr(gp.quicksum(delta[k, 2] + delta[k, 5] for k in range(n_horizon)) <= T3[0])
    model.addConstr(gp.quicksum(delta[k, 3] + delta[k, 4] for k in range(n_horizon)) <= T3[1])
    model.addConstr(gp.quicksum(delta[k, 0] + delta[k, 1] for k in range(n_horizon)) <= T3[2])

    model.setObjective(gp.quicksum(objective_terms), GRB.MINIMIZE)

    if verbose:
        print("    [MPC] Solving ...")

    model.optimize()

    if model.status != GRB.OPTIMAL:
        if verbose:
            print("    [MPC] Solver did not return OPTIMAL. Falling back to greedy step.")
        shares, delta_now = choose_best_permutation_step(modules, pack_current_horizon[0], dt_hours, soh_params)
        x_traj = np.zeros((n_horizon + 1, n_states))
        u_traj = np.zeros((n_horizon, n_states))
        delta_traj = np.zeros((n_horizon, j_count))
        x_traj[0, :] = [m.soc for m in modules]
        u_traj[0, :] = shares * pack_current_horizon[0]
        delta_traj[0, :] = delta_now
        return x_traj, u_traj, delta_traj

    if verbose:
        print("    [MPC] Solve finished successfully.")

    x_traj = np.array([[x[k, i].X for i in range(n_states)] for k in range(n_horizon + 1)])
    delta_traj = np.array([[delta[k, j].X for j in range(j_count)] for k in range(n_horizon)])

    u_traj = np.zeros((n_horizon, n_states))
    for k in range(n_horizon):
        selected_j = int(np.argmax(delta_traj[k, :]))
        shares = sm[selected_j] @ base_shares_cache[k]
        u_traj[k, :] = shares * pack_current_horizon[k]

    return x_traj, u_traj, delta_traj


def update_limits(T1, T2, T3, delta_traj):
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


def solve_system_v1(a_list, l1, l2, l3, p, verbose=True):
    n = len(a_list)

    if verbose:
        print("[INIT] Solving initial allocation problem for T1, T2, T3 ...")

    model = gp.Model("equation_system")
    model.Params.OutputFlag = 0
    model.Params.Threads = max(1, os.cpu_count() or 1)

    T1 = model.addVars(n, name="T1", lb=0)
    T2 = model.addVars(n, name="T2", lb=0)
    T3 = model.addVars(n, name="T3", lb=0)

    for i in range(n):
        model.addConstr(a_list[i] == T1[i] * l1 * p + T2[i] * l2 * p + T3[i] * l3 * p)

    sum_T1 = gp.quicksum(T1[i] for i in range(n))
    sum_T2 = gp.quicksum(T2[i] for i in range(n))
    sum_T3 = gp.quicksum(T3[i] for i in range(n))

    model.addConstr(sum_T1 == sum_T2)
    model.addConstr(sum_T1 == sum_T3)

    model.addConstr(T1[0] + T2[0] + T3[0] == T1[1] + T2[1] + T3[1])
    model.addConstr(T1[0] + T2[0] + T3[0] == T1[2] + T2[2] + T3[2])

    objective = 60 * T1[0] + 40 * T2[1] + 20 * T3[2] + 5 * T2[0] + 3 * T3[1] + 1 * T2[2]
    model.setObjective(objective, GRB.MAXIMIZE)
    model.optimize()

    if model.status == GRB.OPTIMAL:
        result = {
            "T1": [T1[i].X for i in range(n)],
            "T2": [T2[i].X for i in range(n)],
            "T3": [T3[i].X for i in range(n)],
            "ObjectiveValue": model.ObjVal,
        }
        if verbose:
            print("[INIT] Initial allocation solved.")
            print(f"[INIT] Objective value: {result['ObjectiveValue']:.4f}")
        return result

    if verbose:
        print(f"[INIT] Initial allocation failed with status {model.status}.")
    return {"status": model.status}


def apply_currents_to_modules(modules, module_currents_a, dt_hours, soh_params):
    for m, current_a in zip(modules, module_currents_a):
        update_module_state_and_soh(m, current_a, dt_hours, soh_params)


def run_classical_controller(modules, pack_profile_a, dt_hours, n_cycles, soh_params, print_every_cycles=5):
    cycle_len = len(pack_profile_a)
    total_steps = n_cycles * cycle_len

    print("\n[CLASSICAL] Starting simulation ...")
    print(f"[CLASSICAL] Target cycles: {n_cycles}")
    print(f"[CLASSICAL] Steps per cycle: {cycle_len}")
    print(f"[CLASSICAL] Total simulation steps: {total_steps}")

    last_reported_cycle = -1

    for step in range(total_steps):
        pack_current = pack_profile_a[step % cycle_len]
        shares = classical_load_factors(pack_current)
        module_currents = shares * pack_current

        if abs(pack_current) > 1e-12:
            modules = sort_modules(modules, by_soc=(pack_current > 0))

        apply_currents_to_modules(modules, module_currents, dt_hours, soh_params)

        for m, lf in zip(modules, shares):
            m.lf_hist.append(lf)

        current_cycle = (step + 1) // cycle_len
        if current_cycle > last_reported_cycle and current_cycle % print_every_cycles == 0 and (step + 1) % cycle_len == 0:
            avg_soh = np.mean([m.soh for m in modules])
            min_soh = np.min([m.soh for m in modules])
            print(f"[CLASSICAL] Completed cycle {current_cycle}/{n_cycles} | avg SoH={avg_soh:.3f} | min SoH={min_soh:.3f}")
            last_reported_cycle = current_cycle

    print("[CLASSICAL] Simulation finished.")
    return modules


def run_mpc_controller(modules, pack_profile_a, dt_hours, n_cycles, horizon, T1, T2, T3, soh_params,
                       print_every_cycles=5, print_mpc_inner=False):
    cycle_len = len(pack_profile_a)
    total_steps = n_cycles * cycle_len

    print("\n[MPC] Starting simulation ...")
    print(f"[MPC] Target cycles: {n_cycles}")
    print(f"[MPC] Steps per cycle: {cycle_len}")
    print(f"[MPC] Total simulation steps: {total_steps}")
    print(f"[MPC] Prediction horizon: {horizon}")

    last_reported_cycle = -1

    for step in range(total_steps):
        idx0 = step % cycle_len
        horizon_currents = np.array(
            [pack_profile_a[(idx0 + k) % cycle_len] for k in range(horizon)],
            dtype=float
        )

        verbose_now = print_mpc_inner and (step % max(1, cycle_len // 8) == 0)

        x_traj, u_traj, delta_traj = mpc_session_v1(
            modules=modules,
            dt_hours=dt_hours,
            pack_current_horizon=horizon_currents,
            T1=T1,
            T2=T2,
            T3=T3,
            soh_params=soh_params,
            verbose=verbose_now,
        )

        T1, T2, T3 = update_limits(T1, T2, T3, delta_traj)

        module_currents = u_traj[0, :]
        pack_current = pack_profile_a[idx0]
        shares = np.zeros_like(module_currents) if abs(pack_current) < 1e-12 else module_currents / pack_current

        apply_currents_to_modules(modules, module_currents, dt_hours, soh_params)

        for m, lf in zip(modules, shares):
            m.lf_hist.append(lf)

        current_cycle = (step + 1) // cycle_len
        if current_cycle > last_reported_cycle and current_cycle % print_every_cycles == 0 and (step + 1) % cycle_len == 0:
            avg_soh = np.mean([m.soh for m in modules])
            min_soh = np.min([m.soh for m in modules])
            print(f"[MPC] Completed cycle {current_cycle}/{n_cycles} | avg SoH={avg_soh:.3f} | min SoH={min_soh:.3f}")
            last_reported_cycle = current_cycle

    print("[MPC] Simulation finished.")
    return modules


def plot_comparison(classical_modules, mpc_modules, dt_hours):
    print("\n[PLOT] Generating comparison plots ...")

    t_classical = np.arange(len(classical_modules[0].soc_hist)) * dt_hours
    t_mpc = np.arange(len(mpc_modules[0].soc_hist)) * dt_hours

    fig, axes = plt.subplots(3, 1, figsize=(12, 14), sharex=False)

    for i in range(3):
        axes[0].plot(t_classical, classical_modules[i].soc_hist, label=f"Classical {classical_modules[i].module_id}")
        axes[0].plot(t_mpc, mpc_modules[i].soc_hist, "--", label=f"MPC {mpc_modules[i].module_id}")
    axes[0].set_title("SoC comparison")
    axes[0].set_ylabel("SoC")
    axes[0].grid(True)
    axes[0].legend(ncol=2)

    for i in range(3):
        axes[1].plot(t_classical, classical_modules[i].soh_hist, label=f"Classical {classical_modules[i].module_id}")
        axes[1].plot(t_mpc, mpc_modules[i].soh_hist, "--", label=f"MPC {mpc_modules[i].module_id}")
    axes[1].set_title("SoH comparison")
    axes[1].set_ylabel("SoH [%]")
    axes[1].grid(True)
    axes[1].legend(ncol=2)

    for i in range(3):
        axes[2].plot(t_classical, classical_modules[i].fce_hist, label=f"Classical {classical_modules[i].module_id}")
        axes[2].plot(t_mpc, mpc_modules[i].fce_hist, "--", label=f"MPC {mpc_modules[i].module_id}")
    axes[2].set_title("FCE comparison")
    axes[2].set_ylabel("FCE")
    axes[2].set_xlabel("Time [h]")
    axes[2].grid(True)
    axes[2].legend(ncol=2)

    plt.tight_layout()
    plt.show()

    print("[PLOT] Plots displayed.")


def summarize_results(classical_modules, mpc_modules):
    print("\n[SUMMARY] Final results")
    print("=" * 70)
    for i in range(3):
        c = classical_modules[i]
        m = mpc_modules[i]
        print(
            f"{c.module_id}: "
            f"Classical SoH={c.soh:.4f}, MPC SoH={m.soh:.4f}, "
            f"Classical FCE={c.fce:.4f}, MPC FCE={m.fce:.4f}"
        )