#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import numpy as np
import matplotlib.pyplot as plt

from battery_mpc_lib import (
    Module,
    SoHModelParams,
    build_pack_current_profile,
    solve_system_v1,
    run_classical_controller,
    run_mpc_controller,
    plot_comparison,
    summarize_results,
)


def create_modules():
    modules = np.empty(3, dtype=object)

    modules[0] = Module(
        soc=0.40, soh=100.0, imax=50.0,
        soc_min=0.20, soc_max=0.90,
        capacity_nominal_ah=50.0,
        module_id="M1"
    )
    modules[1] = Module(
        soc=0.50, soh=90.0, imax=50.0,
        soc_min=0.20, soc_max=0.85,
        capacity_nominal_ah=50.0,
        module_id="M2"
    )
    modules[2] = Module(
        soc=0.30, soh=80.0, imax=50.0,
        soc_min=0.20, soc_max=0.80,
        capacity_nominal_ah=50.0,
        module_id="M3"
    )

    return modules


def print_run_header():
    print("=" * 72)
    print("Battery module cycling study: Classical controller vs SoH-aware MPC")
    print("=" * 72)


def print_configuration(dt_sec, n_cycles, horizon, profile, base_modules, soh_params):
    print("\n[CONFIG] Simulation settings")
    print(f"[CONFIG] Time step: {dt_sec:.2f} s")
    print(f"[CONFIG] Number of cycles: {n_cycles}")
    print(f"[CONFIG] MPC horizon: {horizon}")
    print(f"[CONFIG] Steps per cycle: {len(profile)}")
    print(f"[CONFIG] Total steps per controller: {len(profile) * n_cycles}")

    print("\n[CONFIG] Initial module states")
    for m in base_modules:
        print(f"[CONFIG] {m.short_state()} | bounds=({m.soc_min:.2f}, {m.soc_max:.2f}) | Cap={m.capacity_nominal_ah:.1f} Ah")

    print("\n[CONFIG] SoH model coefficients")
    print(
        "[CONFIG] "
        f"x = [{soh_params.x1}, {soh_params.x2}, {soh_params.x3}, {soh_params.x4}, {soh_params.x5}, "
        f"{soh_params.x6}, {soh_params.x7}, {soh_params.x8}, {soh_params.x9}, {soh_params.x10}]"
    )


def main():
    plt.ion()
    plt.rcParams.update({"font.size": 13})

    print_run_header()

    dt_sec = 1.0
    dt_hours = dt_sec / 3600.0
    n_cycles = 100
    horizon = 10

    soh_params = SoHModelParams(
        x1=0.010,
        x2=0.020,
        x3=0.100,
        x4=0.150,
        x5=0.015,
        x6=0.200,
        x7=0.050,
        x8=0.010,
        x9=0.400,
        x10=0.70,
    )

    print("\n[SETUP] Creating battery modules ...")
    base_modules = create_modules()

    print("[SETUP] Building pack current profile ...")
    pack_profile_a = build_pack_current_profile()

    print_configuration(dt_sec, n_cycles, horizon, pack_profile_a, base_modules, soh_params)

    print("\n[SETUP] Computing initial allocation limits ...")
    initial_lf = np.array([0.6, 0.3, 0.1], dtype=float)
    a_list = [m.soc_max - m.soc for m in base_modules]

    result = solve_system_v1(
        a_list=a_list,
        l1=initial_lf[0],
        l2=initial_lf[1],
        l3=initial_lf[2],
        p=dt_hours,
        verbose=True,
    )

    T1 = result["T1"]
    T2 = result["T2"]
    T3 = result["T3"]

    print(f"[SETUP] T1 = {np.round(T1, 3)}")
    print(f"[SETUP] T2 = {np.round(T2, 3)}")
    print(f"[SETUP] T3 = {np.round(T3, 3)}")

    print("\n[RUN] Launching classical controller simulation ...")
    classical_modules = np.array([m.clone() for m in base_modules], dtype=object)

    t0 = time.time()
    classical_modules = run_classical_controller(
        modules=classical_modules,
        pack_profile_a=pack_profile_a,
        dt_hours=dt_hours,
        n_cycles=n_cycles,
        soh_params=soh_params,
        print_every_cycles=5,
    )
    classical_time = time.time() - t0
    print(f"[RUN] Classical controller finished in {classical_time:.2f} s")

    print("\n[RUN] Launching MPC controller simulation ...")
    mpc_modules = np.array([m.clone() for m in base_modules], dtype=object)

    t0 = time.time()
    mpc_modules = run_mpc_controller(
        modules=mpc_modules,
        pack_profile_a=pack_profile_a,
        dt_hours=dt_hours,
        n_cycles=n_cycles,
        horizon=horizon,
        T1=T1.copy(),
        T2=T2.copy(),
        T3=T3.copy(),
        soh_params=soh_params,
        print_every_cycles=5,
        print_mpc_inner=False,
    )
    mpc_time = time.time() - t0
    print(f"[RUN] MPC controller finished in {mpc_time:.2f} s")

    print("\n[POST] Preparing summary and plots ...")
    summarize_results(classical_modules, mpc_modules)
    plot_comparison(classical_modules, mpc_modules, dt_hours)

    print("\n[DONE] All tasks completed.")
    plt.ioff()


if __name__ == "__main__":
    main()