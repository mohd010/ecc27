#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

from battery_mpc_lib import (
    SoHModelParams,
    run_single_test,
    aggregate_results,
    plot_best_test_trajectories,
)


def main():
    print("=" * 80)
    print("Parallel comparison: classical controller vs original Gurobi MPC")
    print("=" * 80)

    num_tests = 20
    n_cycles = 10
    horizon = 5
    dt_sec = 1.0

    cpu_count = os.cpu_count() or 8
    max_workers = min(num_tests, max(1, int(0.75 * cpu_count)))
    solver_threads = 1

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

    print("\n[CONFIG]")
    print(f"Tests:                         {num_tests}")
    print(f"Cycles per test:               {n_cycles}")
    print(f"Segments per cycle:            8")
    print(f"solve_system_v1 calls/test:    {8 * n_cycles}")
    print(f"MPC horizon:                   {horizon}")
    print(f"dt:                            {dt_sec} s")
    print(f"CPU count:                     {cpu_count}")
    print(f"Parallel workers:              {max_workers}")
    print(f"Gurobi threads per worker:     {solver_threads}")
    print(f"Initial SoH range:             random uniform [0.5, 1.0]")
    print(f"SoH unit:                      [-], not percent")
    print(f"MPC function:                  original MPCsession_v1 unchanged")

    print("\n[RUN] Starting parallel tests ...")

    global_start = time.time()
    seeds = [1000 + i for i in range(num_tests)]
    results = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                run_single_test,
                test_id=i + 1,
                seed=seeds[i],
                n_cycles=n_cycles,
                horizon=horizon,
                dt_sec=dt_sec,
                soh_params=soh_params,
                solver_threads=solver_threads,
                keep_trajectories=False,
            )
            for i in range(num_tests)
        ]

        completed = 0

        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            completed += 1

            print(
                f"[RUN] Test {result['test_id']:02d}/{num_tests} finished | "
                f"seed={result['seed']} | "
                f"init SoH={np.round(result['init_sohs'], 3)} | "
                f"classical mean SoH={result['classical_mean_soh']:.6f} | "
                f"MPC mean SoH={result['mpc_mean_soh']:.6f} | "
                f"gain={result['mpc_mean_soh'] - result['classical_mean_soh']:.6f} | "
                f"solve_system calls={result['solve_system_calls']} | "
                f"time={result['runtime']:.2f}s | "
                f"progress={completed}/{num_tests}"
            )

    total_wall_time = time.time() - global_start

    print("\n[POST] Aggregating results ...")
    summary = aggregate_results(results)

    print("\n[SUMMARY] Average over tests")
    print("-" * 80)
    print(f"Number of tests:                    {summary['num_tests']}")
    print(f"Average classical mean SoH:         {summary['classical_mean_soh_avg']:.6f}")
    print(f"Average MPC mean SoH:               {summary['mpc_mean_soh_avg']:.6f}")
    print(f"Average classical minimum SoH:      {summary['classical_min_soh_avg']:.6f}")
    print(f"Average MPC minimum SoH:            {summary['mpc_min_soh_avg']:.6f}")
    print(f"Average MPC gain in mean SoH:       {summary['mpc_gain_mean_soh']:.6f}")
    print(f"Average MPC gain in minimum SoH:    {summary['mpc_gain_min_soh']:.6f}")
    print(f"Average solve_system calls/test:    {summary['avg_solve_system_calls']:.1f}")
    print(f"Total solve_system calls:           {summary['total_solve_system_calls']}")
    print(f"Average runtime per test:           {summary['avg_test_runtime']:.2f} s")
    print(f"Sum of individual test runtimes:    {summary['sum_test_runtime']:.2f} s")
    print(f"Total parallel wall-clock runtime:  {total_wall_time:.2f} s")

    best = max(results, key=lambda r: r["mpc_mean_soh"] - r["classical_mean_soh"])

    print("\n[SUMMARY] Test with largest MPC mean-SoH advantage")
    print("-" * 80)
    print(f"Test ID:                 {best['test_id']}")
    print(f"Seed:                    {best['seed']}")
    print(f"Initial SoHs:            {np.round(best['init_sohs'], 6)}")
    print(f"Classical final SoHs:    {np.round(best['classical_final_soh'], 6)}")
    print(f"MPC final SoHs:          {np.round(best['mpc_final_soh'], 6)}")
    print(f"Classical mean SoH:      {best['classical_mean_soh']:.6f}")
    print(f"MPC mean SoH:            {best['mpc_mean_soh']:.6f}")
    print(f"MPC mean SoH advantage:  {best['mpc_mean_soh'] - best['classical_mean_soh']:.6f}")
    print(f"Solve-system calls:      {best['solve_system_calls']}")

    print("\n[POST] Re-running best test with full trajectories for plotting ...")

    best_detailed = run_single_test(
        test_id=best["test_id"],
        seed=best["seed"],
        n_cycles=n_cycles,
        horizon=horizon,
        dt_sec=dt_sec,
        soh_params=soh_params,
        solver_threads=solver_threads,
        keep_trajectories=True,
    )

    plot_best_test_trajectories(
        classical_modules=best_detailed["classical_modules"],
        mpc_modules=best_detailed["mpc_modules"],
        dt_sec=dt_sec,
        T1_traj=best_detailed["T1_hist"],
        T2_traj=best_detailed["T2_hist"],
        T3_traj=best_detailed["T3_hist"],
        save_path="best_test_soc_soh.png",
    )

    print("\n[DONE] Finished.")
    print("[DONE] Best-test plot saved as: best_test_soc_soh.png")


if __name__ == "__main__":
    main()